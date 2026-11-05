"""
IPCL Vault 予測ロジック（アンサンブルによる予測Vault ＋ LogReg と SVM のアンサンブルによる 750µm 以下の確率
                        ＋ ロジスティック回帰による 400〜600µm に入る確率）
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
  - 特徴量（共通）    : LV, ACD, size, ACW, age（単位: mm / 歳）
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

    table = pd.DataFrame({
        'ICLサイズ (mm)': [f'{s:.2f}' for s in sizes],
        '予測Vault (µm)': np.round(vaults * 1000).astype(int),
        f'{prob_label(bundle)}の確率 (%)': np.round(probs * 100, 1),
        f'{narrow_label(bundle)}に入る確率 (%)': np.round(probs_narrow * 100, 1),
        'Vault判定': [vault_label(v) for v in vaults],
    })

    warnings_ = check_inputs(bundle, age, LV, ACD, ACW)
    r = bundle['feature_ranges']['size']
    outside = [f'{s:.2f}' for s in sizes if s < r['min'] or s > r['max']]
    if outside:
        warnings_.append(f'サイズ {", ".join(outside)}mm は学習データに無いサイズです'
                         f'（学習データは {r["min"]:.2f}〜{r["max"]:.2f}mm）。参考程度にしてください。')

    return {'table': table, 'sizes': sizes, 'vaults': vaults, 'probs': probs, 'probs_narrow': probs_narrow,
            'lower_size': lower, 'cut_um': bundle['p_within']['cut_mm'] * 1000,
            'narrow_um': tuple(v * 1000 for v in bundle['p_narrow']['range_mm']), 'warnings': warnings_}


def make_figure(res, title=''):
    """サイズごとの予測Vault・基準値以下の確率・400〜600µm に入る確率（白背景・点プロット。文字化けを避けるためラベルは英語）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sizes, vaults, probs, probs_n = res['sizes'], res['vaults'] * 1000, res['probs'] * 100, res['probs_narrow'] * 100

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.4), facecolor='white')
    for ax in (a1, a2, a3):
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
                        (a3, probs_n, f'P({lo:.0f}-{hi:.0f} um) (LogReg)')]:
        ax.plot(sizes, pr, '-', color='#90A4AE', lw=1, zorder=1)
        ax.scatter(sizes, pr, s=55, color='#1565C0', edgecolor='black', lw=0.5, zorder=2)
        for s, p in zip(sizes, pr):
            ax.annotate(f'{p:.0f}%', (s, p), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)', fontweight='bold')
        ax.set_title(ttl, fontweight='bold')

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
                                             '400〜600µm に入る確率（LogReg）')
    ap.add_argument('--age', type=float, required=True)
    ap.add_argument('--LV', type=float, required=True)
    ap.add_argument('--ACD', type=float, required=True)
    ap.add_argument('--ACW', type=float, required=True)
    ap.add_argument('--bundle', default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    res = predict_eye(load_bundle(a.bundle), a.age, a.LV, a.ACD, a.ACW)
    print(f"ACW {a.ACW:.2f}mm → {res['lower_size']:.2f}mm から {len(res['sizes'])} サイズ")
    print(res['table'].to_string(index=False))
    for w in res['warnings']:
        print('注意:', w)


if __name__ == '__main__':
    main()
