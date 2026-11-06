"""
IPCL Vault 予測ロジック（アンサンブルによる予測Vault ＋ LogReg と SVM のアンサンブルによる 750µm 以下の確率
                        ＋ ロジスティック回帰による 400〜600µm に入る確率 ＋ アンサンブルによる正しいサイズの確率）
====================================================================================================
Streamlit アプリと学習スクリプト（train_model.py）の共通部品です。
Streamlit がなくても、コマンドラインから予測できます。

    python ipcl_vault_predict.py --age 35 --LV -0.12 --ACD 3.25 --ACW 11.75

ACW より大きい最小の ICL サイズから 0.25mm 刻みで 5 サイズについて予測します。

モデル（全データ.ipynb と同じ設定）
  - 予測Vault        : Ensemble … セル20  RobustScaler → SVR (Linear) + Ridge + MLR の予測の平均
  - 750µm 以下の確率 : Ensemble … セル31  RobustScaler → LogisticRegression(C=1.0, max_iter=1000) と
                                   SVC(kernel='rbf', gamma=0.03, class_weight='balanced') の確率の平均
                                   ノートブックと同じく Vault > 0.75mm を陽性として学習したモデルの
                                   1 − 確率（750µm ちょうどは「以下」に入る）。基準値は
                                   bundle['p_within']['cut_mm'] に保存し、表示はそこから作る
  - 400〜600µm の確率: LogReg   … セル22  RobustScaler → LogisticRegression(C=1.0, max_iter=1000)
                                   目的変数 c400_600 == 1（400・600µm ちょうどを含む）。範囲は
                                   bundle['p_narrow']['range_mm'] に保存し、表示はそこから作る
  - 特徴量（上の3つ）  : LV, ACD, size, ACW, age（単位: mm / 歳）
  - 正しいサイズの確率: Ensemble … セル30  多項ロジスティック・多クラス XGBoost・多クラス CatBoost の確率の平均
                                   目的変数は段階 =（正しいサイズ − ACW より大きい最小サイズ）/ 0.25（0〜K）。
                                   段階 0〜4 が表示する 5 サイズにそのまま対応する（5 サイズの合計 100%）。
                                   特徴量は ACW, LV, ACD, age, gap（= ACW より大きい最小サイズ − ACW）。
                                   予測するサイズは確率分布の中央値（小さいサイズから足して 50% に達する段階）
                                   XGBoost・CatBoost は学習後に木（分岐の特徴量・しきい値・葉の値）だけを取り出し、
                                   numpy で同じ確率を計算する形（XGBTrees / CatTrees）で保存するので、
                                   アプリの実行に xgboost・catboost は不要（学習時だけ必要）
  モデルは sklearn の Pipeline（RobustScaler + 推定器）として保存しています。
  予測Vault の3モデルはどれも線形なので、係数・切片の平均（1つの線形式）として保存しています。
  確率は VotingClassifier（soft voting = 2モデルの確率の平均）です。
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, 'models', 'ipcl_vault_bundle.pkl')

FEATURES = ['LV', 'ACD', 'size', 'ACW', 'age']           # ノートブックと同じ列の順番
SIZE_FEATURES = ['ACW', 'LV', 'ACD', 'age', 'gap']       # 正しいサイズのモデル（セル30 と同じ列の順番）
SIZE_GRID = np.round(np.arange(11.00, 14.01, 0.25), 2)   # ICL サイズ候補
STEP_MM = 0.25
N_STEPS = 5                      # 表示するサイズ: ACW より大きい最小サイズ 〜 その +1.00mm
VAULT_RANGE = (0.25, 0.75)       # 広義の適正 Vault (mm)
VAULT_NARROW = (0.40, 0.60)      # 狭義の適正 Vault (mm)


def lower_size(acw):
    """ACW より大きい最小の候補サイズ"""
    acw = float(acw)
    above = SIZE_GRID[SIZE_GRID > acw]
    return float(above.min()) if len(above) else float(SIZE_GRID.max())


class StepClassifier:
    """セル30 の NominalClf と同じ多クラス分類器: RobustScaler → 分類器。
    学習データに無い段階の確率は 0 とし、段階 0〜k の k+1 列の確率を返す"""

    def __init__(self, base, k):
        self.base, self.k = base, k

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        self.present_ = np.unique(y)
        if len(self.present_) == 1:
            self.model_ = None
            return self
        enc = {c: i for i, c in enumerate(self.present_)}
        self.model_ = make_pipeline(RobustScaler(), clone(self.base)).fit(X, np.array([enc[v] for v in y]))
        return self

    def predict_proba(self, X):
        out = np.zeros((len(X), self.k + 1))
        if self.model_ is None:
            out[:, int(self.present_[0])] = 1.0
            return out
        P = self.model_.predict_proba(X)
        for j, c in enumerate(self.present_):
            out[:, int(c)] = P[:, j]
        s = out.sum(axis=1, keepdims=True)
        return np.where(s > 0, out / np.where(s > 0, s, 1), 1.0 / (self.k + 1))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class XGBTrees:
    """学習済み XGBoost（多クラス）の木だけを持ち、numpy で確率を計算する。
    各ノードで 値 < しきい値 なら左へ（XGBoost と同じく float32 で比較。欠測は default_left の向き）。
    確率 = softmax（クラスごとの base_margin + そのクラスの木の葉の値の合計）"""

    def __init__(self, trees, tree_class, base_margin, n_class):
        self.trees, self.tree_class = trees, np.asarray(tree_class, dtype=int)
        self.base_margin, self.n_class = np.asarray(base_margin, dtype=float), int(n_class)

    def leaf_sum(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((len(X), self.n_class))
        for t, c in zip(self.trees, self.tree_class):
            node = np.zeros(len(X), dtype=int)
            while True:
                r = np.where(t['left'][node] >= 0)[0]
                if len(r) == 0:
                    break
                n = node[r]
                x = X[r, t['feat'][n]]
                go_left = np.where(np.isnan(x), t['default_left'][n], x < t['thr'][n])
                node[r] = np.where(go_left, t['left'][n], t['right'][n])
            out[:, c] += t['leaf'][node]
        return out

    def predict_proba(self, X):
        return _softmax(self.leaf_sum(X) + self.base_margin)


class CatTrees:
    """学習済み CatBoost（多クラス・対称木）の木だけを持ち、numpy で確率を計算する。
    各段で 値 > しきい値 なら 1 として葉の番号を作る（float32 で比較）。
    確率 = softmax（scale × 葉の値の合計 + bias）"""

    def __init__(self, feats, borders, leaves, scale, bias):
        self.feats, self.borders, self.leaves = feats, borders, leaves
        self.scale, self.bias = float(scale), np.asarray(bias, dtype=float)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((len(X), len(self.bias)))
        for f, b, lv in zip(self.feats, self.borders, self.leaves):
            idx = np.zeros(len(X), dtype=int)
            for j in range(len(f)):
                idx |= (X[:, f[j]] > b[j]).astype(int) << j
            out += lv[idx]
        return _softmax(self.scale * out + self.bias)


class ScaledModel:
    """RobustScaler で変換してから、木だけのモデル（XGBTrees / CatTrees）で確率を計算する"""

    def __init__(self, scaler, model):
        self.scaler, self.model = scaler, model

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


def median_step(P):
    """確率分布の中央値の段階（小さい段階から足して 50% に達する段階。セル30 の sc_median_class）"""
    return np.argmax(np.cumsum(np.atleast_2d(P), axis=1) >= 0.5 - 1e-12, axis=1)


def size_features(ACW, LV, ACD, age):
    """正しいサイズのモデルの特徴量（gap = ACW より大きい最小サイズ − ACW）"""
    ACW = np.atleast_1d(np.asarray(ACW, dtype=float))
    gap = np.array([lower_size(a) for a in ACW]) - ACW
    return pd.DataFrame({'ACW': ACW, 'LV': LV, 'ACD': ACD, 'age': age, 'gap': gap})[SIZE_FEATURES].astype(float)


def predict_size_proba(bundle, X):
    """X（SIZE_FEATURES の列）→ 段階 0〜K の「正しいサイズ」である確率（アンサンブル = 各モデルの確率の平均）"""
    return np.mean([m.predict_proba(X) for m in bundle['p_size']['members'].values()], axis=0)


def load_bundle(path=DEFAULT_BUNDLE):
    if not os.path.exists(path):
        raise FileNotFoundError(f'モデルファイルが見つかりません: {path}\n'
                                f'train_model.py（または retrain）を実行して作成してください。')
    with open(path, 'rb') as f:
        return pickle.load(f)


def vault_label(v_mm):
    if VAULT_NARROW[0] <= v_mm <= VAULT_NARROW[1]:
        return '◎ 400-600µm'
    if VAULT_RANGE[0] <= v_mm <= VAULT_RANGE[1]:
        return '○ 250-750µm'
    return '△ 低い (<250µm)' if v_mm < VAULT_RANGE[0] else '△ 高い (>750µm)'


def prob_label(bundle):
    """確率モデルの基準（例: '750µm 以下'）"""
    return f"{bundle['p_within']['cut_mm'] * 1000:.0f}µm 以下"


def narrow_label(bundle):
    """400〜600µm の確率モデルの範囲（例: '400〜600µm'）"""
    lo, hi = bundle['p_narrow']['range_mm']
    return f'{lo * 1000:.0f}〜{hi * 1000:.0f}µm'


def predict_frame(bundle, X):
    """X（FEATURES の列を持つ DataFrame）→ 予測Vault (mm)、基準値（cut_mm）以下の確率 (0–1)、
    400〜600µm に入る確率 (0–1)"""
    X = X[FEATURES].astype(float)
    vault = np.asarray(bundle['vault']['model'].predict(X), dtype=float)
    clf = bundle['p_within']['model']   # Vault > cut_mm をクラス1として学習 → クラス0 が「以下」
    prob = clf.predict_proba(X)[:, list(clf.classes_).index(0)]
    nar = bundle['p_narrow']['model']   # c400_600 == 1（400〜600µm）がクラス1
    prob_narrow = nar.predict_proba(X)[:, list(nar.classes_).index(1)]
    return vault, prob, prob_narrow


def check_inputs(bundle, age, LV, ACD, ACW):
    """学習データの範囲外の入力を知らせる"""
    msgs = []
    for name, v in [('age', age), ('LV', LV), ('ACD', ACD), ('ACW', ACW)]:
        r = bundle['feature_ranges'][name]
        if not (r['min'] <= float(v) <= r['max']):
            msgs.append(f'{name} = {float(v):g} は学習データの範囲（{r["min"]:g}〜{r["max"]:g}）の外です。'
                        f'予測の信頼性が下がります。')
    return msgs


def predict_eye(bundle, age, LV, ACD, ACW):
    """1眼分の予測。ACW より大きい最小サイズから 5 サイズの予測Vaultと確率を返す"""
    lower = lower_size(ACW)
    sizes = np.round(lower + STEP_MM * np.arange(N_STEPS), 2)
    sizes = sizes[sizes <= SIZE_GRID.max()]
    grid = pd.DataFrame({'LV': LV, 'ACD': ACD, 'size': sizes, 'ACW': ACW, 'age': age})
    vaults, probs, probs_narrow = predict_frame(bundle, grid)
    P = predict_size_proba(bundle, size_features(ACW, LV, ACD, age))[0]   # 段階 0〜K = lower から 0.25mm 刻み
    pick = int(median_step(P)[0])
    probs_size = np.zeros(len(sizes))
    probs_size[:min(len(sizes), len(P))] = P[:len(sizes)]

    table = pd.DataFrame({
        'ICLサイズ (mm)': [f'{s:.2f}' for s in sizes],
        '予測Vault (µm)': np.round(vaults * 1000).astype(int),
        f'{prob_label(bundle)}の確率 (%)': np.round(probs * 100, 1),
        f'{narrow_label(bundle)}に入る確率 (%)': np.round(probs_narrow * 100, 1),
        '正しいサイズの確率 (%)': np.round(probs_size * 100, 1),
        '正しいサイズの予測': ['★' if i == pick else '' for i in range(len(sizes))],
        'Vault判定': [vault_label(v) for v in vaults],
    })

    warnings_ = check_inputs(bundle, age, LV, ACD, ACW)
    r = bundle['feature_ranges']['size']
    outside = [f'{s:.2f}' for s in sizes if s < r['min'] or s > r['max']]
    if outside:
        warnings_.append(f'サイズ {", ".join(outside)}mm は学習データに無いサイズです'
                         f'（学習データは {r["min"]:.2f}〜{r["max"]:.2f}mm）。参考程度にしてください。')
    if len(sizes) < len(P):
        warnings_.append(f'{SIZE_GRID.max():.2f}mm より大きいサイズは表示していません。正しいサイズの確率のうち'
                         f' {P[len(sizes):].sum() * 100:.1f}% はそれらのサイズの分です。')

    return {'table': table, 'sizes': sizes, 'vaults': vaults, 'probs': probs, 'probs_narrow': probs_narrow,
            'probs_size': probs_size, 'size_pick': lower + STEP_MM * pick,
            'lower_size': lower, 'cut_um': bundle['p_within']['cut_mm'] * 1000,
            'narrow_um': tuple(v * 1000 for v in bundle['p_narrow']['range_mm']), 'warnings': warnings_}


def make_figure(res, title=''):
    """サイズごとの予測Vault・基準値以下の確率・400〜600µm に入る確率・正しいサイズの確率
    （白背景・点プロット。文字化けを避けるためラベルは英語）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sizes, vaults, probs, probs_n = res['sizes'], res['vaults'] * 1000, res['probs'] * 100, res['probs_narrow'] * 100
    probs_s = res['probs_size'] * 100

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), facecolor='white')
    (a1, a2), (a3, a4) = axes
    for ax in axes.ravel():
        ax.set_facecolor('white')
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{s:.2f}' for s in sizes])
        ax.set_xlabel('ICL size (mm)', fontweight='bold')
        ax.set_xlim(sizes.min() - 0.15, sizes.max() + 0.15)

    a1.axhspan(VAULT_RANGE[0] * 1000, VAULT_RANGE[1] * 1000, color='#4CAF50', alpha=0.10, lw=0)
    a1.axhspan(VAULT_NARROW[0] * 1000, VAULT_NARROW[1] * 1000, color='#4CAF50', alpha=0.20, lw=0)
    a1.plot(sizes, vaults, '-', color='#90A4AE', lw=1, zorder=1)
    a1.scatter(sizes, vaults, s=55, color='#1565C0', edgecolor='black', lw=0.5, zorder=2)
    for s, v in zip(sizes, vaults):
        a1.annotate(f'{v:.0f}', (s, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    a1.set_ylim(min(0, vaults.min() - 100), max(1100, vaults.max() + 150))
    a1.set_ylabel('Predicted vault (um)', fontweight='bold')
    a1.set_title('Predicted vault (Ensemble)', fontweight='bold')

    lo, hi = res['narrow_um']
    for ax, pr, ttl in [(a2, probs, f"P(vault ≤ {res['cut_um']:.0f} um) (LogReg + SVM)"),
                        (a3, probs_n, f'P({lo:.0f}-{hi:.0f} um) (LogReg)'),
                        (a4, probs_s, 'P(correct size) (Ensemble, sum 100%)')]:
        ax.plot(sizes, pr, '-', color='#90A4AE', lw=1, zorder=1)
        ax.scatter(sizes, pr, s=55, color='#1565C0', edgecolor='black', lw=0.5, zorder=2)
        for s, p in zip(sizes, pr):
            ax.annotate(f'{p:.0f}%', (s, p), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)', fontweight='bold')
        ax.set_title(ttl, fontweight='bold')
    k = int(np.argmin(np.abs(sizes - res['size_pick'])))
    if abs(sizes[k] - res['size_pick']) < 1e-9:   # 正しいサイズの予測（確率分布の中央値）に印
        a4.scatter([sizes[k]], [probs_s[k]], s=170, facecolor='none', edgecolor='#C62828', lw=1.8, zorder=3)

    if title:
        fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    return fig


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description='IPCL 予測Vault（アンサンブル）/ 750µm 以下の確率（LogReg + SVM）/ '
                                             '400〜600µm に入る確率（LogReg）/ 正しいサイズの確率（アンサンブル）')
    ap.add_argument('--age', type=float, required=True)
    ap.add_argument('--LV', type=float, required=True)
    ap.add_argument('--ACD', type=float, required=True)
    ap.add_argument('--ACW', type=float, required=True)
    ap.add_argument('--bundle', default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    res = predict_eye(load_bundle(a.bundle), a.age, a.LV, a.ACD, a.ACW)
    print(f"ACW {a.ACW:.2f}mm → {res['lower_size']:.2f}mm から {len(res['sizes'])} サイズ")
    print(res['table'].to_string(index=False))
    print(f"正しいサイズの予測（確率分布の中央値）: {res['size_pick']:.2f}mm")
    for w in res['warnings']:
        print('注意:', w)


if __name__ == '__main__':
    main()
