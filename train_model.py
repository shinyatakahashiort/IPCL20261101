"""
予測Vault（アンサンブル）と 750µm 超の確率（ロジスティック回帰）の最終モデルを学習し、
models/ipcl_vault_bundle.pkl と models/metrics.json を作成する
=====================================================================
データ : IPCL9.xlsx（既定はこのフォルダ。--data で指定可）
モデル : 全データ.ipynb と同じ特徴量・前処理・ハイパーパラメータ
         - 予測Vault        : セル20 Ensemble（SVR (Linear) + Ridge + MLR の予測の平均 = VotingRegressor）
                               RobustScaler → SVR(C=1.0, epsilon=0.05, kernel='linear') / Ridge(alpha=1.0) /
                               LinearRegression。目的変数 Vault2 (mm)、特徴量の欠測は中央値で補完
         - 750µm 超の確率   : セル31 LogReg（Vault > 0.75mm, high075）
                               RobustScaler → LogisticRegression(C=1.0, max_iter=1000)
                               目的変数 Vault2 > 0.75（750µm ちょうどは陰性）、欠測のある眼は除外
         - 特徴量（共通）    : LV, ACD, size, ACW, age
評価   : 患者単位 LOOCV（Leave-One-Patient-Out）
         両眼データのため、同じ患者の左右眼は必ず同じ側（学習 or 検証）に入れる
確認   : ノートブック セル17 と同じ訓練群 / ホールドアウト群の分割を再現し、
         ノートブックの出力（訓練群 CV・ホールドアウトの指標）と一致するかを表示する
         （この分割は照合のためだけに使い、最終モデルの学習には使わない）
最終   : 全眼で学習し直して保存（pkl に個票データは保存しない。SVR はサポートベクター＝学習眼の
         特徴量を内部に持つため、3モデルとも線形であることを使い、アンサンブルを同じ予測になる
         係数・切片（3モデルの平均）だけの形に置き換えて保存する）

使い方:
    python train_model.py
    python train_model.py --data /path/to/IPCL9.xlsx
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (brier_score_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from ipcl_vault_predict import DEFAULT_BUNDLE, FEATURES, HERE

SEED = 2025              # ノートブック セル4 の GLOBAL_SEED と同じ
HOLDOUT_RATIO = 0.2      # ノートブック セル17 と同じ
NUMERIC_COLS = ['age', 'LV', 'ACD', 'ACW', 'size', 'Vault2']
HIGH_CUT = 0.75         # ノートブック セル31 high075: Vault2 > 0.75 を陽性
HIGH_LABEL = f'{HIGH_CUT * 1000:.0f}µm 超'
REQUIRED_COLS = ['ID', 'RL'] + NUMERIC_COLS


# ============================================================
# データ（ノートブック セル7）
# ============================================================
def build_patient_id(frame):
    """患者ID。ID欠測の区間は「連続2行 = 同一患者の両眼」として復元する"""
    pid = [('' if pd.isna(v) else f'P{int(v)}') for v in frame['ID']]
    miss = [i for i, v in enumerate(pid) if v == '']
    for k in range(0, len(miss), 2):
        for j in miss[k:k + 2]:
            pid[j] = f'X{k // 2}'
    return pid


def load_data(path):
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f'データに必要な列がありません: {missing}')

    problems = []
    for c in NUMERIC_COLS:
        bad = df[c].map(lambda v: pd.notna(v) and not isinstance(v, (int, float, np.integer, np.floating)))
        if bad.any():
            problems.append(f'  {c}: 数値でない値が {int(bad.sum())}件（Excel行 {[i + 2 for i in df.index[bad][:5]]}）')
    if problems:
        raise SystemExit('データに異常があります（Excelの列ズレ・貼り付けミスの可能性）:\n' + '\n'.join(problems))

    df['pid'] = build_patient_id(df)
    return df


def vault_xy(frame):
    """セル20 と同じ: Vault2 を目的変数に、特徴量の欠測は（その群の）中央値で補完"""
    frame = frame[frame['Vault2'].notna()]
    X = frame[FEATURES].fillna(frame[FEATURES].median())
    return X, frame['Vault2'].to_numpy(dtype=float), frame['pid'].to_numpy()


def high_xy(frame):
    """セル31 と同じ: 特徴量・Vault2 に欠測のある眼は除外し、Vault2 > HIGH_CUT を陽性とする"""
    frame = frame.dropna(subset=FEATURES + ['Vault2'])
    return frame[FEATURES], (frame['Vault2'] > HIGH_CUT).astype(int).to_numpy(), frame['pid'].to_numpy()


# ============================================================
# モデル（ノートブック セル20・セル31 と同じ設定）
# ============================================================
ENSEMBLE_MEMBERS = ['SVR (Linear)', 'Ridge', 'MLR']   # セル20 の Top3（訓練群 CV の R² 上位3）


def make_ensemble():
    """セル20 と同じ: RobustScaler → VotingRegressor（3モデルの予測の単純平均）"""
    return Pipeline([('scaler', RobustScaler()),
                     ('model', VotingRegressor([('m0', SVR(C=1.0, epsilon=0.05, kernel='linear')),
                                                ('m1', Ridge(alpha=1.0, random_state=SEED)),
                                                ('m2', LinearRegression())]))])


def make_logreg():
    return Pipeline([('scaler', RobustScaler()),
                     ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])


def compact_linear_ensemble(pipe):
    """3モデル（線形カーネル SVR・Ridge・MLR）はどれも 係数 · x + 切片 で予測するため、その平均である
    アンサンブルも 係数・切片 の平均だけで決まる。SVR 本体はサポートベクター（学習眼の特徴量）を
    内部に保持するため、配布用に係数・切片だけを持つ LinearRegression に置き換える（学習はしない・予測は同じ）"""
    members = pipe['model'].estimators_
    lin = LinearRegression()
    lin.coef_ = np.mean([np.ravel(m.coef_) for m in members], axis=0).astype(float)
    lin.intercept_ = float(np.mean([np.ravel(m.intercept_)[0] for m in members]))
    lin.n_features_in_ = len(lin.coef_)
    return Pipeline([('scaler', pipe['scaler']), ('model', lin)])


def lopo_oof(make, X, y, groups, proba):
    """患者単位 LOOCV の out-of-fold 予測"""
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups=groups):
        m = make().fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1] if proba else m.predict(X.iloc[te])
    return oof


# ============================================================
# 評価指標
# ============================================================
def reg_metrics(y, p):
    err = np.abs(y - p)
    return {'n': int(len(y)), 'R2': float(r2_score(y, p)),
            'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'MAE': float(mean_absolute_error(y, p)),
            'within_100um': float((err <= 0.100 + 1e-9).mean() * 100),
            'within_150um': float((err <= 0.150 + 1e-9).mean() * 100)}


def youden_threshold(y, p):
    """セル22 と同じ: 感度 + 特異度 − 1 が最大となる閾値"""
    fpr, tpr, thr = roc_curve(y, p)
    return float(min(max(thr[int(np.argmax(tpr - fpr))], 0.0), 1.0))


def sens_spec(y, p, thr):
    pred = (np.asarray(p) >= thr).astype(int)
    return (float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1) * 100),
            float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1) * 100))


def cls_metrics(y, p, thr=None):
    """thr を渡すとその閾値（ホールドアウト用）、渡さなければ Youden 閾値を求めて評価する"""
    thr = youden_threshold(y, p) if thr is None else thr
    sens, spec = sens_spec(y, p, thr)
    return {'n': int(len(y)), 'prevalence': float(y.mean() * 100), 'AUC': float(roc_auc_score(y, p)),
            'Brier': float(brier_score_loss(y, p)), 'threshold': thr, 'sens': sens, 'spec': spec}


# ============================================================
# ノートブックの訓練群 / ホールドアウト群で再現確認（セル17・20・31）
# ============================================================
NOTEBOOK_OUTPUT_NOTE = '全データ.ipynb 2026-10-30 実行時の出力（2026-11-01 に c400_600・Vault を修正する前の IPCL9.xlsx）'
NOTEBOOK_OUTPUT = {   # NOTEBOOK_OUTPUT_NOTE の出力（セル20 Ensemble・セル31 Vault > 0.75mm の LogReg）
    'vault_train_cv': {'R2': 0.2387, 'RMSE': 0.1448, 'MAE': 0.1067},
    'vault_holdout': {'R2': 0.2647, 'RMSE': 0.1433, 'MAE': 0.1101},
    'high_train_cv': {'AUC': 0.724, 'threshold': 0.29, 'sens': 69.8, 'spec': 71.5},
    'high_holdout': {'AUC': 0.699, 'sens': 62.5, 'spec': 66.7},
}


def notebook_split_check(df):
    strata = ((df['Vault2'] >= 0.25) & (df['Vault2'] <= 0.75)).astype(int).values
    sgk = StratifiedGroupKFold(n_splits=int(round(1 / HOLDOUT_RATIO)), shuffle=True, random_state=SEED)
    idx_tr, idx_ho = next(sgk.split(np.arange(len(df)), strata, groups=df['pid'].values))
    d_tr, d_ho = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_ho].reset_index(drop=True)

    Xv, yv, gv = vault_xy(d_tr)
    Xv_ho, yv_ho, _ = vault_xy(d_ho)
    Xp, yp, gp = high_xy(d_tr)
    Xp_ho, yp_ho, _ = high_xy(d_ho)

    p_cv = cls_metrics(yp, lopo_oof(make_logreg, Xp, yp, gp, proba=True))
    res = {
        'n_train': int(len(d_tr)), 'n_holdout': int(len(d_ho)),
        'patients_train': int(d_tr['pid'].nunique()), 'patients_holdout': int(d_ho['pid'].nunique()),
        'vault_train_cv': reg_metrics(yv, lopo_oof(make_ensemble, Xv, yv, gv, proba=False)),
        'vault_holdout': reg_metrics(yv_ho, make_ensemble().fit(Xv, yv).predict(Xv_ho)),
        'high_train_cv': p_cv,
        'high_holdout': cls_metrics(yp_ho, make_logreg().fit(Xp, yp).predict_proba(Xp_ho)[:, 1],
                                    thr=p_cv['threshold']),
    }

    print('\n' + '=' * 70)
    print('【確認】ノートブックの訓練群 / ホールドアウト群（セル17の分割）で再現 ※照合用。最終モデルには使わない')
    print('=' * 70)
    print(f"  訓練群 {res['n_train']}眼 / {res['patients_train']}人   "
          f"ホールドアウト {res['n_holdout']}眼 / {res['patients_holdout']}人")
    print(f'  比較対象: {NOTEBOOK_OUTPUT_NOTE}')
    ok = {'vault': True, 'high': True}
    for key, cols in [('vault_train_cv', ['R2', 'RMSE', 'MAE']), ('vault_holdout', ['R2', 'RMSE', 'MAE']),
                      ('high_train_cv', ['AUC', 'threshold', 'sens', 'spec']),
                      ('high_holdout', ['AUC', 'sens', 'spec'])]:
        line = []
        for c in cols:
            ref, got = NOTEBOOK_OUTPUT[key][c], res[key][c]
            tol = 0.5 * 10 ** -len(str(ref).split('.')[1]) + 1e-9   # ノートブックの表示桁の丸め分だけ許容
            same = abs(got - ref) <= tol
            ok[key.split('_')[0]] &= same
            line.append(f'{c}={got:.4f}（ノートブック {ref}）{"" if same else " ← 不一致"}')
        print(f'  {key:<15} ' + '  '.join(line))
    for k, lbl in [('vault', '予測Vault'), ('high', f'{HIGH_LABEL}の確率')]:
        print(f"  → {lbl}: {'✅ ノートブックの出力と一致' if ok[k] else '⚠️ ノートブックの出力と異なります（データが更新された場合はノートブックの再実行が必要）'}")
    res['matches_notebook_output'] = {k: bool(v) for k, v in ok.items()}
    res['notebook_output'] = NOTEBOOK_OUTPUT
    res['notebook_output_note'] = NOTEBOOK_OUTPUT_NOTE
    return res


# ============================================================
# main
# ============================================================
def main():
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=f'IPCL 予測Vault（アンサンブル）/ {HIGH_LABEL}の確率（LogReg）の学習')
    ap.add_argument('--data', default=os.path.join(HERE, 'IPCL9.xlsx'))
    ap.add_argument('--out', default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        raise SystemExit(f'データファイルが見つかりません: {a.data}\n'
                         f'IPCL9.xlsx をこのフォルダに置くか、--data で場所を指定してください。')

    df = load_data(a.data)
    print('=' * 70)
    print(f'IPCL 予測Vault（アンサンブル）/ {HIGH_LABEL}の確率（LogReg）の学習')
    print('=' * 70)
    print(f'データ: {os.path.basename(a.data)}  {len(df)}眼 / {df["pid"].nunique()}人')
    print(f'特徴量: {FEATURES}')
    print(f'評価  : 患者単位 LOOCV（Leave-One-Patient-Out, {df["pid"].nunique()} fold）')

    nb_check = notebook_split_check(df)

    # ---- 全データで交差検証 → 最終モデル ----
    Xv, yv, gv = vault_xy(df)
    Xp, yp, gp = high_xy(df)
    print('\n' + '=' * 70)
    print(f'【全症例】患者単位 LOOCV と最終モデル（全症例で学習: Vault {len(yv)}眼 / 確率 {len(yp)}眼）')
    print('=' * 70)
    v_cv = reg_metrics(yv, lopo_oof(make_ensemble, Xv, yv, gv, proba=False))
    p_cv = cls_metrics(yp, lopo_oof(make_logreg, Xp, yp, gp, proba=True))
    print(f"  予測Vault Ensemble    : R²={v_cv['R2']:.3f}  RMSE={v_cv['RMSE'] * 1000:.0f}µm  "
          f"MAE={v_cv['MAE'] * 1000:.0f}µm  ±100µm以内={v_cv['within_100um']:.1f}%")
    print(f"  {HIGH_LABEL}の確率 LogReg : AUC={p_cv['AUC']:.3f}  Brier={p_cv['Brier']:.3f}  "
          f"Youden閾値={p_cv['threshold']:.2f}（感度 {p_cv['sens']:.1f}% / 特異度 {p_cv['spec']:.1f}%）  "
          f"陽性割合={p_cv['prevalence']:.1f}%")

    ens_fitted = make_ensemble().fit(Xv, yv)
    ens = compact_linear_ensemble(ens_fitted)
    diff = float(np.abs(ens.predict(Xv) - ens_fitted.predict(Xv)).max())
    if diff > 1e-9:
        raise SystemExit(f'係数・切片だけのアンサンブルの予測が元のアンサンブルと一致しません（最大差 {diff:.2e} mm）')
    n_sv = len(ens_fitted['model'].estimators_[0].support_)
    print(f'  アンサンブルを係数・切片だけの形に置き換え（SVR のサポートベクター {n_sv}眼分を除外、'
          f'予測の最大差 {diff:.1e} mm）')
    members = {nm: dict(zip(FEATURES, map(float, np.ravel(m.coef_))))
               for nm, m in zip(ENSEMBLE_MEMBERS, ens_fitted['model'].estimators_)}
    logreg = make_logreg().fit(Xp, yp)
    coef = {'vault_ensemble': dict(zip(FEATURES, map(float, ens['model'].coef_.ravel()))),
            'high_logreg': dict(zip(FEATURES, map(float, logreg['model'].coef_.ravel())))}
    print('  係数（RobustScaler 後）')
    for k, c in {**{f'  {nm}': c for nm, c in members.items()}, **coef}.items():
        print(f'    {k:<15} ' + '  '.join(f'{f}={v:+.4f}' for f, v in c.items()))

    import joblib
    import scipy
    import sklearn
    bundle = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': os.path.basename(a.data),
        'n_eyes': int(len(df)),
        'n_patients': int(df['pid'].nunique()),
        'seed': SEED,
        'cv_method': '患者単位 LOOCV（Leave-One-Patient-Out）',
        'source': '全データ.ipynb セル20（Ensemble: SVR (Linear) + Ridge + MLR）・セル31（LogReg, Vault > 0.75mm）と同じ特徴量・前処理・設定',
        'versions': {'python': '.'.join(map(str, sys.version_info[:3])), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__},
        'vault': {'model': ens, 'name': 'Ensemble (SVR (Linear) + Ridge + MLR)', 'features': FEATURES,
                  'target': 'Vault2 (mm)',
                  'params': 'RobustScaler → 平均[SVR(C=1.0, epsilon=0.05, kernel="linear"), Ridge(alpha=1.0), '
                            'LinearRegression]',
                  'members': ENSEMBLE_MEMBERS, 'member_coef': members,
                  'cv': v_cv, 'coef': coef['vault_ensemble']},
        'p_high': {'model': logreg, 'name': 'LogReg', 'features': FEATURES,
                   'target': f'Vault2 > {HIGH_CUT:.2f}（Vault {HIGH_LABEL}）', 'cut_mm': HIGH_CUT,
                   'params': 'RobustScaler → LogisticRegression(C=1.0, max_iter=1000)',
                   'cv': p_cv, 'coef': coef['high_logreg']},
        'notebook_split': nb_check,
        # 入力チェック用の要約統計のみ（個票データは保存しない）
        'feature_ranges': {c: {'min': float(df[c].min()), 'max': float(df[c].max()),
                               'median': float(df[c].median())} for c in FEATURES},
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(bundle, f)

    metrics = {k: v for k, v in bundle.items() if k not in ('vault', 'p_high')}
    metrics['vault'] = {k: v for k, v in bundle['vault'].items() if k != 'model'}
    metrics['p_high'] = {k: v for k, v in bundle['p_high'].items() if k != 'model'}
    with open(os.path.join(os.path.dirname(a.out), 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print(f'保存: {a.out}  ({os.path.getsize(a.out) / 1024:.0f} KB)')
    print(f'      {os.path.join(os.path.dirname(a.out), "metrics.json")}')
    print('=' * 70)


if __name__ == '__main__':
    main()
