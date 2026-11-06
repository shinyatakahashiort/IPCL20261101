"""
IPCL Vault 予測アプリ（予測Vault: アンサンブル / 750µm 以下の確率: LogReg と SVM のアンサンブル /
                      400〜600µm に入る確率: ロジスティック回帰 / 正しいサイズの確率: アンサンブル）
==========================================================================================
Streamlit Cloud: Main file path に streamlit_app.py を指定してデプロイ
      （手元で試すなら  python -m streamlit run streamlit_app.py ）
モデル: models/ipcl_vault_bundle.pkl（train_model.py で作成）
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ipcl_vault_predict import (DEFAULT_BUNDLE, N_STEPS, load_bundle, make_figure, narrow_label, predict_eye,
                                prob_label)

st.set_page_config(page_title='IPCL Vault Predictor', page_icon='👁️', layout='wide')


@st.cache_resource
def get_bundle(path, stamp):
    """stamp（更新時刻・サイズ）を引数に含め、pkl を差し替えたら自動で読み直す
    （Streamlit Cloud はファイル更新後もプロセスを続けるため、引数が同じだと古いモデルが残る）"""
    return load_bundle(path)


try:
    _st = os.stat(DEFAULT_BUNDLE)
    B = get_bundle(DEFAULT_BUNDLE, (_st.st_mtime_ns, _st.st_size))
except FileNotFoundError:
    st.error(f'モデルファイルが見つかりません: {DEFAULT_BUNDLE}\n'
             f'train_model.py（または retrain）を実行して作成してください。')
    st.stop()

if ('member_cv' not in B.get('p_within', {}) or 'p_narrow' not in B or 'p_size' not in B
        or 'pick' not in B['p_size']):   # モデルがそろう前、または★の付け方を変える前の pkl
    st.error('models/ipcl_vault_bundle.pkl が古い形式です（現在の確率モデルが入っていません）。'
             'streamlit_app.py と同じ時に作った models/ipcl_vault_bundle.pkl に差し替えてください。')
    st.stop()

V, P = B['vault'], B['p_within']
HL = prob_label(B)                       # 例: '750µm 以下'
CUT = f"{P['cut_mm'] * 1000:.0f}µm"      # 例: '750µm'
N = B['p_narrow']
NL = narrow_label(B)                     # 例: '400〜600µm'
S = B['p_size']

# ============================================================
# ヘッダー
# ============================================================
st.title('👁️ IPCL Vault Predictor')
st.caption(f'術前の測定値から、ACW より大きい最小の ICL サイズから {N_STEPS} サイズについて、'
           'アンサンブル（SVR・Ridge・重回帰の平均）による**予測Vault**と、'
           f'ロジスティック回帰と SVM の平均による **Vault が {CUT} 以下になる確率**、'
           f'ロジスティック回帰による **Vault が {NL} に入る確率**、'
           'アンサンブル（多項ロジスティック・XGBoost・CatBoost の平均）による **正しいサイズの確率**を表示します')
st.markdown(
    f"**学習データ**: 全症例 {B['n_eyes']}眼 / {B['n_patients']}人 ・ "
    f"**予測Vault**: {V['name']}（誤差の目安 RMSE {V['cv']['RMSE'] * 1000:.0f}µm / MAE {V['cv']['MAE'] * 1000:.0f}µm） ・ "
    f"**{HL}の確率**: {P['name']}（AUC {P['cv']['AUC']:.2f}） ・ "
    f"**{NL}に入る確率**: {N['name']}（AUC {N['cv']['AUC']:.2f}） ・ "
    f"**正しいサイズの確率**: アンサンブル（予測と正しいサイズの差 MAE {S['cv']['MAE_mm']:.2f}mm・"
    f"完全一致 {S['cv']['exact']:.0f}%）"
)

# 学習時と実行時のライブラリのバージョン違いを検出（pickle の互換性の問題を切り分けやすくする）
try:
    import scipy
    import sklearn
    _now = {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'scikit-learn': sklearn.__version__}
    _diff = {k: (v, _now[k]) for k, v in B['versions'].items() if k in _now and v != _now[k]}
    if _diff:
        st.warning('学習時とバージョンが異なるライブラリがあります: '
                   + ', '.join(f'{k} {a} → {b}' for k, (a, b) in _diff.items())
                   + '。予測結果が変わる可能性があります。requirements.txt の版で入れ直してください。')
except ImportError:
    pass

st.markdown('---')

# ============================================================
# 入力
# ============================================================
RNG = B['feature_ranges']


def num_input(label, name, key, step, fmt):
    r = RNG[name]
    pad = (r['max'] - r['min']) * 0.5
    return st.number_input(label, min_value=float(round(r['min'] - pad, 3)),
                           max_value=float(round(r['max'] + pad, 3)),
                           value=float(round(r['median'], 3)), step=step, format=fmt, key=key)


with st.sidebar:
    st.header('患者情報')
    age = st.number_input('年齢', min_value=10, max_value=90, value=int(round(RNG['age']['median'])), step=1)
    both = st.checkbox('両眼を入力する', value=True)
    st.markdown('---')
    st.caption(f"学習データの範囲: 年齢 {RNG['age']['min']:.0f}〜{RNG['age']['max']:.0f}歳 / "
               f"LV {RNG['LV']['min']:.2f}〜{RNG['LV']['max']:.2f} / "
               f"ACD {RNG['ACD']['min']:.2f}〜{RNG['ACD']['max']:.2f} / "
               f"ACW {RNG['ACW']['min']:.2f}〜{RNG['ACW']['max']:.2f} / "
               f"サイズ {RNG['size']['min']:.2f}〜{RNG['size']['max']:.2f} mm")

eyes = [('右眼 (OD)', 'R', 'Right eye (OD)'), ('左眼 (OS)', 'L', 'Left eye (OS)')] if both \
    else [('右眼 (OD)', 'R', 'Right eye (OD)')]
inputs = {}
for col, (name, k, _) in zip(st.columns(len(eyes)), eyes):
    with col:
        st.subheader(name)
        inputs[name] = {
            'LV': num_input('LV — Lens Vault (mm)', 'LV', f'lv{k}', 0.01, '%.3f'),
            'ACD': num_input('ACD — Anterior Chamber Depth (mm)', 'ACD', f'acd{k}', 0.01, '%.2f'),
            'ACW': num_input('ACW — Anterior Chamber Width (mm)', 'ACW', f'acw{k}', 0.01, '%.2f'),
        }

st.markdown('---')

# ============================================================
# 結果
# ============================================================
if st.button('予測する', type='primary'):
    st.caption(f"{HL}の確率の目安: 学習データ全体で {CUT} 以下だったのは {P['cv']['prevalence']:.0f}%"
               f"（{CUT} ちょうどを含みます。250µm 未満の低い Vault も「以下」に入ります）、"
               f"交差検証で決めた判定の目安（Youden 閾値）は {P['cv']['threshold'] * 100:.0f}% です。"
               f"{NL}に入る確率の目安: 学習データ全体で {NL} に入ったのは {N['cv']['prevalence']:.0f}%"
               f"（{NL.replace('〜', '・')} ちょうどを含みます）、Youden 閾値は {N['cv']['threshold'] * 100:.0f}% です。"
               "正しいサイズの確率: 5 サイズの合計が 100% で、★ は確率が最も高いサイズです。")
    for col, (name, _, name_en) in zip(st.columns(len(eyes)), eyes):
        with col:
            st.subheader(f'{name} の結果')
            v = inputs[name]
            res = predict_eye(B, age, v['LV'], v['ACD'], v['ACW'])
            sizes = res['sizes']
            st.markdown(f"ACW {v['ACW']:.2f} mm → **{sizes[0]:.2f}〜{sizes[-1]:.2f} mm** の {len(sizes)} サイズ")
            st.dataframe(res['table'], hide_index=True)
            st.markdown(f"正しいサイズの予測（確率が最も高いサイズ）: **{res['size_pick']:.2f} mm**")
            for w in res['warnings']:
                st.warning(w)
            fig = make_figure(res, name_en)
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info('左のサイドバーと上の入力欄を埋めて「予測する」を押してください。')

# ============================================================
# モデルの詳細
# ============================================================
st.markdown('---')
with st.expander('モデルの詳細と注意事項'):
    nb = B['notebook_split']

    def _nb_match(k):
        return '一致します' if nb['matches_notebook_output'][k] else \
            '異なります（データを修正したため。ノートブックを再実行すると更新されます）'

    st.markdown(f"""
**このアプリが使うモデル**: {B['data_file']} の **全症例 {B['n_eyes']}眼 / {B['n_patients']}人**をすべて使って学習した
最終モデルです（訓練群・ホールドアウト群には分けていません。作成: {B['created']}）。

**元の解析**: {B['source']}。特徴量は {', '.join(V['features'])}（単位 mm / 歳）、
どれも RobustScaler で標準化してから学習しています。

**性能の推定方法**: 全症例で{B['cv_method']}を行いました（1人を外して残りの全員で学習し、外した1人を予測する、を
全員について繰り返す）。両眼データのため、同じ患者の左右眼は必ず同じ側に入れて評価しています。

- **予測Vault** — {V['params']}：R² {V['cv']['R2']:.3f}、RMSE {V['cv']['RMSE'] * 1000:.0f}µm、
  MAE {V['cv']['MAE'] * 1000:.0f}µm、±100µm 以内 {V['cv']['within_100um']:.0f}%。
  3モデルはどれも線形なので、その平均であるアンサンブルは係数を平均した1つの線形式と同じ予測になります
- **{HL}の確率** — {P['params']}。{P['trained_as']}：AUC {P['cv']['AUC']:.3f}、
  Brier {P['cv']['Brier']:.3f}、Youden 閾値 {P['cv']['threshold'] * 100:.0f}% で感度 {P['cv']['sens']:.0f}% /
  特異度 {P['cv']['spec']:.0f}%（{HL}を陽性として計算）。単独では
  {'、'.join(f"{nm} AUC {c['AUC']:.3f}" for nm, c in P['member_cv'].items())}
- **{NL}に入る確率** — {N['params']}（ノートブック セル22 ①狭義。目的変数 {N['target']}）：AUC {N['cv']['AUC']:.3f}、
  Brier {N['cv']['Brier']:.3f}、Youden 閾値 {N['cv']['threshold'] * 100:.0f}% で感度 {N['cv']['sens']:.0f}% /
  特異度 {N['cv']['spec']:.0f}%
- **正しいサイズの確率** — {S['params']}（ノートブック セル30。目的変数 {S['target']}、特徴量 {', '.join(S['features'])}。
  gap = ACW より大きい最小サイズ − ACW）：予測（確率が最も高いサイズ）と正しいサイズの差 MAE {S['cv']['MAE_mm']:.3f}mm、
  完全一致 {S['cv']['exact']:.1f}%、±0.25mm 以内 {S['cv']['within025']:.1f}%、正解サイズの確率の平均 {S['cv']['p_true']:.1f}%。
  単独では {'、'.join(f"{nm} MAE {c['MAE_mm']:.3f}mm" for nm, c in S['member_cv'].items())}

**参考：ノートブックとの照合**（最終モデルには使っていません）: ノートブックは訓練群（{nb['n_train']}眼）で学習し、
ホールドアウト群（{nb['n_holdout']}眼）で検証しています。学習の手順がノートブックと同じであることを確かめるため、
同じ分割を再現して{nb['notebook_output_note']}と比べたのが下の表です。
予測Vaultは{_nb_match('vault')}。{CUT} 超の確率（ノートブックのモデル。アプリはその 1 − 確率の平均を表示）は、
LogReg が{_nb_match('high')}。SVM_RBF も{_nb_match('svm')}。{NL}に入る確率（セル22 LogReg）は{_nb_match('narrow')}。正しいサイズの確率（セル30 アンサンブル）は{_nb_match('size')}。
正しいサイズの照合だけは、ノートブックと同じく確率分布の中央値（小さいサイズから足して 50% に達するサイズ）を
予測サイズにしています（アプリの★と上の性能は、確率が最も高いサイズで計算しています）。
""")
    _groups = {'vault_train_cv': '予測Vault・訓練群CV', 'vault_holdout': '予測Vault・ホールドアウト',
               'high_train_cv': f'{CUT} 超 LogReg・訓練群CV', 'high_holdout': f'{CUT} 超 LogReg・ホールドアウト',
               'svm_train_cv': f'{CUT} 超 SVM_RBF・訓練群CV', 'svm_holdout': f'{CUT} 超 SVM_RBF・ホールドアウト',
               'narrow_train_cv': f'{NL} LogReg・訓練群CV', 'narrow_holdout': f'{NL} LogReg・ホールドアウト',
               'size_train_cv': '正しいサイズ・訓練群CV', 'size_holdout': '正しいサイズ・ホールドアウト'}
    _metric = {'R2': 'R²', 'RMSE': 'RMSE (mm)', 'MAE': 'MAE (mm)', 'AUC': 'AUC',
               'threshold': 'Youden 閾値', 'sens': '感度 (%)', 'spec': '特異度 (%)',
               'MAE_mm': 'MAE (mm)', 'exact': '完全一致 (%)', 'within025': '±0.25mm 以内 (%)',
               'logloss': 'LogLoss', 'p_true': '正解サイズの確率の平均 (%)'}
    st.dataframe(pd.DataFrame([
        {'区分': g, '指標': _metric[m], '同じ分割で再現': round(nb[k][m], 4), 'ノートブックの出力': ref}
        for k, g in _groups.items() for m, ref in nb['notebook_output'][k].items()
    ]), hide_index=True)
    st.markdown(f"""
**確率の読み方の注意**: 予測Vaultと {HL}の確率は別々に学習したモデルです。サイズを大きくすると予測Vaultは
上がり、{HL}の確率は下がります（係数の符号: 予測Vault size {V['coef']['size']:+.3f}、確率の LogReg size {P['coef']['size']:+.2f}。
SVM は非線形のため係数はありません）が、
予測Vaultが {CUT} ちょうどでも確率が 50% になるとは限りません。両方を合わせて見てください。
{NL}に入る確率も別に学習したモデルで、LogReg（線形）のため、サイズに対して一方向にしか変化しません
（係数 size {N['coef']['size']:+.2f}：サイズを大きくすると{'上がり' if N['coef']['size'] > 0 else '下がり'}ます）。

**正しいサイズの確率について**: 「正しいサイズ」は、術後 Vault などをふまえて判定した、本来選ぶべきサイズです。
表示する 5 サイズそれぞれが「正しいサイズ」である確率で、合計は 100% です（Vault が特定の範囲に入る確率ではありません）。
学習データでは、ACW より大きい最小サイズから {'・'.join(f"+{0.25 * a:.2f}mm が {b}眼" for a, b in S['stage_counts'].items())} でした。

---
⚠️ **本アプリは研究目的の参考情報です。** 予測Vaultの誤差は RMSE で {V['cv']['RMSE'] * 1000:.0f}µm、MAE で {V['cv']['MAE'] * 1000:.0f}µm 程度あり、
{HL}の確率の判別能は AUC {P['cv']['AUC']:.2f}、{NL}に入る確率は AUC {N['cv']['AUC']:.2f} です。
正しいサイズの予測が実際の正しいサイズと一致したのは {S['cv']['exact']:.0f}%（±0.25mm 以内は {S['cv']['within025']:.0f}%）です。
最終的なレンズ選択は必ず医師の判断で行ってください。
""")
