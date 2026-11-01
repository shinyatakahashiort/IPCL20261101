"""
IPCL Vault 予測ロジック（SVR による予測Vault ＋ ロジスティック回帰による 400–600µm 確率）
=====================================================================================
Streamlit アプリと学習スクリプト（train_model.py）の共通部品です。
Streamlit がなくても、コマンドラインから予測できます。

    python ipcl_vault_predict.py --age 35 --LV -0.12 --ACD 3.25 --ACW 11.75 --size 12.50

モデル（全データ.ipynb と同じ設定）
  - 予測Vault        : SVR (Linear)  … セル20  RobustScaler → SVR(C=1.0, epsilon=0.05, kernel='linear')
  - 400–600µm の確率 : LogReg        … セル22  RobustScaler → LogisticRegression(C=1.0, max_iter=1000)
  - 特徴量（共通）    : LV, ACD, size, ACW, age（単位: mm / 歳）
  モデルは sklearn の Pipeline（RobustScaler + 推定器）として保存しています。
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
N_STEPS = 5                      # 比較表: ACW より大きい最小サイズ 〜 その +1.00mm
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


def predict_frame(bundle, X):
    """X（FEATURES の列を持つ DataFrame）→ 予測Vault (mm) と 400–600µm の確率 (0–1)"""
    X = X[FEATURES].astype(float)
    vault = np.asarray(bundle['vault']['model'].predict(X), dtype=float)
    clf = bundle['p400_600']['model']
    prob = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    return vault, prob


def check_inputs(bundle, age, LV, ACD, ACW, size):
    """学習データの範囲外の入力を知らせる"""
    msgs = []
    for name, v in [('age', age), ('LV', LV), ('ACD', ACD), ('ACW', ACW), ('size', size)]:
        r = bundle['feature_ranges'][name]
        if not (r['min'] <= float(v) <= r['max']):
            msgs.append(f'{name} = {float(v):g} は学習データの範囲（{r["min"]:g}〜{r["max"]:g}）の外です。'
                        f'予測の信頼性が下がります。')
    if float(size) <= float(ACW):
        msgs.append(f'サイズ {float(size):.2f}mm が ACW {float(ACW):.2f}mm 以下です'
                    f'（学習データでは全眼で サイズ > ACW）。')
    return msgs


def predict_eye(bundle, age, LV, ACD, ACW, size):
    """1眼分の予測。入力サイズの結果と、ACW より大きい候補サイズごとの比較表を返す"""
    size = round(float(size), 2)
    x = pd.DataFrame([{'LV': LV, 'ACD': ACD, 'size': size, 'ACW': ACW, 'age': age}])
    vault, prob = predict_frame(bundle, x)

    lower = lower_size(ACW)
    sizes = np.round(lower + STEP_MM * np.arange(N_STEPS), 2)
    sizes = np.unique(np.append(sizes[sizes <= SIZE_GRID.max()], size))
    grid = pd.DataFrame({'LV': LV, 'ACD': ACD, 'size': sizes, 'ACW': ACW, 'age': age})
    vaults, probs = predict_frame(bundle, grid)

    table = pd.DataFrame({
        'ICLサイズ (mm)': [f'{s:.2f}' for s in sizes],
        '予測Vault (µm)': np.round(vaults * 1000).astype(int),
        '400–600µm の確率 (%)': np.round(probs * 100, 1),
        'Vault判定': [vault_label(v) for v in vaults],
        '入力サイズ': ['◀' if s == size else '' for s in sizes],
    })

    warnings_ = check_inputs(bundle, age, LV, ACD, ACW, size)
    return {'size': size, 'vault': float(vault[0]), 'prob': float(prob[0]),
            'table': table, 'sizes': sizes, 'vaults': vaults, 'probs': probs,
            'lower_size': lower, 'warnings': warnings_}


def make_figure(res, title=''):
    """サイズごとの予測Vaultと 400–600µm 確率（白背景・点プロット。文字化けを避けるためラベルは英語）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sizes, vaults, probs = res['sizes'], res['vaults'] * 1000, res['probs'] * 100
    sel = sizes == res['size']

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor='white')
    for ax in (a1, a2):
        ax.set_facecolor('white')
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{s:.2f}' for s in sizes])
        ax.set_xlabel('ICL size (mm)', fontweight='bold')
        ax.set_xlim(sizes.min() - 0.15, sizes.max() + 0.15)

    a1.axhspan(VAULT_RANGE[0] * 1000, VAULT_RANGE[1] * 1000, color='#4CAF50', alpha=0.10, lw=0)
    a1.axhspan(VAULT_NARROW[0] * 1000, VAULT_NARROW[1] * 1000, color='#4CAF50', alpha=0.20, lw=0)
    a1.plot(sizes, vaults, '-', color='#90A4AE', lw=1, zorder=1)
    a1.scatter(sizes[~sel], vaults[~sel], s=55, color='#1565C0', edgecolor='black', lw=0.5, zorder=2)
    a1.scatter(sizes[sel], vaults[sel], s=110, color='#E91E63', edgecolor='black', lw=0.8, zorder=3)
    for s, v in zip(sizes, vaults):
        a1.annotate(f'{v:.0f}', (s, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    a1.set_ylim(min(0, vaults.min() - 100), max(1100, vaults.max() + 150))
    a1.set_ylabel('Predicted vault (um)', fontweight='bold')
    a1.set_title('Predicted vault (SVR)', fontweight='bold')

    a2.plot(sizes, probs, '-', color='#90A4AE', lw=1, zorder=1)
    a2.scatter(sizes[~sel], probs[~sel], s=55, color='#1565C0', edgecolor='black', lw=0.5, zorder=2)
    a2.scatter(sizes[sel], probs[sel], s=110, color='#E91E63', edgecolor='black', lw=0.8, zorder=3)
    for s, p in zip(sizes, probs):
        a2.annotate(f'{p:.0f}%', (s, p), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    a2.set_ylim(0, 100)
    a2.set_ylabel('Probability (%)', fontweight='bold')
    a2.set_title('P(vault 400-600 um) (LogReg)', fontweight='bold')

    if title:
        fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    return fig


def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description='IPCL 予測Vault（SVR）/ 400–600µm 確率（LogReg）')
    ap.add_argument('--age', type=float, required=True)
    ap.add_argument('--LV', type=float, required=True)
    ap.add_argument('--ACD', type=float, required=True)
    ap.add_argument('--ACW', type=float, required=True)
    ap.add_argument('--size', type=float, required=True)
    ap.add_argument('--bundle', default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    res = predict_eye(load_bundle(a.bundle), a.age, a.LV, a.ACD, a.ACW, a.size)
    print(f"サイズ {res['size']:.2f}mm: 予測Vault {res['vault'] * 1000:.0f}µm"
          f" / 400–600µm に入る確率 {res['prob'] * 100:.1f}%")
    print(res['table'].to_string(index=False))
    for w in res['warnings']:
        print('注意:', w)


if __name__ == '__main__':
    main()
