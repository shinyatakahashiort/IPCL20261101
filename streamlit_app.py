"""
IPCL Vault 予測アプリ（予測Vault: SVR / 400–600µm 確率: ロジスティック回帰）
=========================================================================
Streamlit Cloud: Main file path に streamlit_app.py を指定してデプロイ
      （手元で試すなら  python -m streamlit run streamlit_app.py ）
モデル: models/ipcl_vault_bundle.pkl（train_model.py で作成）
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ipcl_vault_predict import DEFAULT_BUNDLE, N_STEPS, load_bundle, make_figure, predict_eye

st.set_page_config(page_title='IPCL Vault Predictor', page_icon='👁️', layout='wide')


@st.cache_resource
def get_bundle():
    return load_bundle(DEFAULT_BUNDLE)


try:
    B = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

V, P = B['vault'], B['p400_600']

# ============================================================
# ヘッダー
# ============================================================
st.title('👁️ IPCL Vault Predictor')
st.caption(f'術前の測定値から、ACW より大きい最小の ICL サイズから {N_STEPS} サイズについて、'
           'SVR による**予測Vault**と、ロジスティック回帰による **Vault が 400–600µm に入る確率**を表示します')
st.markdown(
    f"**学習データ**: {B['n_eyes']}眼 / {B['n_patients']}人 ・ "
    f"**予測Vault**: {V['name']}（誤差の目安 RMSE {V['cv']['RMSE'] * 1000:.0f}µm） ・ "
    f"**400–600µm の確率**: {P['name']}（AUC {P['cv']['AUC']:.2f}）"
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
    st.caption(f"400–600µm の確率の目安: 学習データ全体で 400–600µm に入ったのは {P['cv']['prevalence']:.0f}%、"
               f"交差検証で決めた判定の目安（Youden 閾値）は {P['cv']['threshold'] * 100:.0f}% です。"
               f"確率はサイズが小さいほど高く出るため、予測Vaultと合わせて見てください。")
    for col, (name, _, name_en) in zip(st.columns(len(eyes)), eyes):
        with col:
            st.subheader(f'{name} の結果')
            v = inputs[name]
            res = predict_eye(B, age, v['LV'], v['ACD'], v['ACW'])
            sizes = res['sizes']
            st.markdown(f"ACW {v['ACW']:.2f} mm → **{sizes[0]:.2f}〜{sizes[-1]:.2f} mm** の {len(sizes)} サイズ")
            st.dataframe(res['table'], hide_index=True)
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
**学習データ**: {B['data_file']} — {B['n_eyes']}眼 / {B['n_patients']}人（作成: {B['created']}）。
全眼で学習し直した最終モデルです。

**元の解析**: {B['source']}。特徴量は {', '.join(V['features'])}（単位 mm / 歳）、
どちらも RobustScaler で標準化してから学習しています。

**性能の推定方法**: {B['cv_method']}。両眼データのため、同じ患者の左右眼は必ず同じ側に入れて評価しています。

- **予測Vault** — {V['params']}：R² {V['cv']['R2']:.3f}、RMSE {V['cv']['RMSE'] * 1000:.0f}µm、
  MAE {V['cv']['MAE'] * 1000:.0f}µm、±100µm 以内 {V['cv']['within_100um']:.0f}%
- **400–600µm の確率** — {P['params']}（目的変数 {P['target']}）：AUC {P['cv']['AUC']:.3f}、
  Brier {P['cv']['Brier']:.3f}、Youden 閾値 {P['cv']['threshold'] * 100:.0f}% で感度 {P['cv']['sens']:.0f}% /
  特異度 {P['cv']['spec']:.0f}%

**ノートブックとの対応**: ノートブックの訓練群（{nb['n_train']}眼）/ ホールドアウト群（{nb['n_holdout']}眼）の分割を
再現し、{nb['notebook_output_note']}と比べています。
予測Vaultは{_nb_match('vault')}。400–600µm の確率は{_nb_match('p400')}。
""")
    _groups = {'vault_train_cv': '予測Vault・訓練群CV', 'vault_holdout': '予測Vault・ホールドアウト',
               'p400_train_cv': '400–600µm・訓練群CV', 'p400_holdout': '400–600µm・ホールドアウト'}
    _metric = {'R2': 'R²', 'RMSE': 'RMSE (mm)', 'MAE': 'MAE (mm)', 'AUC': 'AUC',
               'threshold': 'Youden 閾値', 'sens': '感度 (%)', 'spec': '特異度 (%)'}
    st.dataframe(pd.DataFrame([
        {'区分': g, '指標': _metric[m], 'このモデルで再現': round(nb[k][m], 4), 'ノートブックの出力': ref}
        for k, g in _groups.items() for m, ref in nb['notebook_output'][k].items()
    ]), hide_index=True)
    st.markdown(f"""
**確率の読み方の注意**: ロジスティック回帰は各変数について単調なモデルです（係数の符号: size
{P['coef']['size']:+.2f}）。このため、サイズを小さくするほど 400–600µm に入る確率は上がり続け、
予測Vaultが 400µm を下回るサイズでも確率は下がりません。サイズの比較には予測Vaultと合わせて使ってください。

---
⚠️ **本アプリは研究目的の参考情報です。** 予測Vaultの誤差は RMSE で {V['cv']['RMSE'] * 1000:.0f}µm 程度あり、
400–600µm に入る確率の判別能は AUC {P['cv']['AUC']:.2f} です。最終的なレンズ選択は必ず医師の判断で行ってください。
""")
