"""
予測Vault（SVR）と 400–600µm 確率（ロジスティック回帰）の最終モデルを学習し、
models/ipcl_vault_bundle.pkl と models/metrics.json を作成する
=====================================================================
データ : IPCL9.xlsx（既定はこのフォルダ。--data で指定可）
モデル : 全データ.ipynb と同じ特徴量・前処理・ハイパーパラメータ
         - 予測Vault        : セル20 SVR (Linear)  RobustScaler → SVR(C=1.0, epsilon=0.05, kernel='linear')
                               目的変数 Vault2 (mm)、特徴量の欠測は中央値で補完
         - 400–600µm の確率 : セル22 LogReg        RobustScaler → LogisticRegression(C=1.0, max_iter=1000)
                               目的変数 c400_600 == 1、欠測のある眼は除外
         - 特徴量（共通）    : LV, ACD, size, ACW, age
評価   : 患者単位 LOOCV（Leave-One-Patient-Out）
         両眼データのため、同じ患者の左右眼は必ず同じ側（学習 or 検証）に入れる
確認   : ノートブック セル17 と同じ訓練群 / ホールドアウト群の分割を再現し、
         ノートブックの出力（訓練群 CV・ホールドアウトの指標）と一致するかを表示する
最終   : 全眼で学習し直して保存（pkl に個票データは保存しない。SVR はサポートベクター＝学習眼の
         特徴量を内部に持つため、同じ予測になる 係数・切片 だけの形に置き換えて保存する）

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

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (brier_score_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from ipcl_vault_predict import DEFAULT_BUNDLE, FEATURES, HERE

SEED = 2025              # ノートブック セル4 の GLOBAL_SEED と同じ
HOLDOUT_RATIO = 0.2      # ノートブック セル17 と同じ
NUMERIC_COLS = ['age', 'LV', 'ACD', 'ACW', 'size', 'Vault2', 'c400_600']
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


def p400_xy(frame):
    """セル22 と同じ: 特徴量・c400_600・Vault2 に欠測のある眼は除外し、c400_600 == 1 を陽性とする"""
    frame = frame.dropna(subset=FEATURES + ['c400_600', 'Vault2'])
    return frame[FEATURES], (frame['c400_600'] == 1).astype(int).to_numpy(), frame['pid'].to_numpy()


# ============================================================
# モデル（ノートブック セル20・セル22 と同じ設定）
# ============================================================
def make_svr():
    return Pipeline([('scaler', RobustScaler()),
                     ('model', SVR(C=1.0, epsilon=0.05, kernel='linear'))])


def make_logreg():
    return Pipeline([('scaler', RobustScaler()),
                     ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])


def compact_linear_svr(pipe):
    """線形カーネル SVR の予測は 係数・切片 だけで決まる（coef_ · x + intercept_）。
    SVR 本体はサポートベクター（学習眼の特徴量）を内部に保持するため、配布用に
    係数・切片だけを持つ LinearRegression に置き換える（学習はしない・予測は同じ）"""
    svr = pipe['model']
    lin = LinearRegression()
    lin.coef_ = svr.coef_.ravel().astype(float)
    lin.intercept_ = float(svr.intercept_[0])
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
# ノートブックの訓練群 / ホールドアウト群で再現確認（セル17・20・22）
# ============================================================
NOTEBOOK_OUTPUT_NOTE = '全データ.ipynb 2026-10-30 実行時の出力（2026-11-01 に c400_600・Vault を修正する前の IPCL9.xlsx）'
NOTEBOOK_OUTPUT = {   # NOTEBOOK_OUTPUT_NOTE の出力
    'vault_train_cv': {'R2': 0.2547, 'RMSE': 0.1433, 'MAE': 0.1061},
    'vault_holdout': {'R2': 0.2891, 'RMSE': 0.1409, 'MAE': 0.1082},
    'p400_train_cv': {'AUC': 0.700, 'threshold': 0.23, 'sens': 86.4, 'spec': 50.3},
    'p400_holdout': {'AUC': 0.705, 'sens': 80.0, 'spec': 54.8},
}


def notebook_split_check(df):
    strata = ((df['Vault2'] >= 0.25) & (df['Vault2'] <= 0.75)).astype(int).values
    sgk = StratifiedGroupKFold(n_splits=int(round(1 / HOLDOUT_RATIO)), shuffle=True, random_state=SEED)
    idx_tr, idx_ho = next(sgk.split(np.arange(len(df)), strata, groups=df['pid'].values))
    d_tr, d_ho = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_ho].reset_index(drop=True)

    Xv, yv, gv = vault_xy(d_tr)
    Xv_ho, yv_ho, _ = vault_xy(d_ho)
    Xp, yp, gp = p400_xy(d_tr)
    Xp_ho, yp_ho, _ = p400_xy(d_ho)

    p_cv = cls_metrics(yp, lopo_oof(make_logreg, Xp, yp, gp, proba=True))
    res = {
        'n_train': int(len(d_tr)), 'n_holdout': int(len(d_ho)),
        'patients_train': int(d_tr['pid'].nunique()), 'patients_holdout': int(d_ho['pid'].nunique()),
        'vault_train_cv': reg_metrics(yv, lopo_oof(make_svr, Xv, yv, gv, proba=False)),
        'vault_holdout': reg_metrics(yv_ho, make_svr().fit(Xv, yv).predict(Xv_ho)),
        'p400_train_cv': p_cv,
        'p400_holdout': cls_metrics(yp_ho, make_logreg().fit(Xp, yp).predict_proba(Xp_ho)[:, 1],
                                    thr=p_cv['threshold']),
    }

    print('\n' + '=' * 70)
    print('【確認】ノートブックの訓練群 / ホールドアウト群（セル17の分割）で再現')
    print('=' * 70)
    print(f"  訓練群 {res['n_train']}眼 / {res['patients_train']}人   "
          f"ホールドアウト {res['n_holdout']}眼 / {res['patients_holdout']}人")
    print(f'  比較対象: {NOTEBOOK_OUTPUT_NOTE}')
    ok = {'vault': True, 'p400': True}
    for key, cols in [('vault_train_cv', ['R2', 'RMSE', 'MAE']), ('vault_holdout', ['R2', 'RMSE', 'MAE']),
                      ('p400_train_cv', ['AUC', 'threshold', 'sens', 'spec']),
                      ('p400_holdout', ['AUC', 'sens', 'spec'])]:
        line = []
        for c in cols:
            ref, got = NOTEBOOK_OUTPUT[key][c], res[key][c]
            tol = 0.06 if c in ('sens', 'spec') else (0.006 if c == 'threshold' else 0.00051)
            same = abs(round(got, 4) - ref) <= tol
            ok[key.split('_')[0]] &= same
            line.append(f'{c}={got:.4f}（ノートブック {ref}）{"" if same else " ← 不一致"}')
        print(f'  {key:<15} ' + '  '.join(line))
    for k, lbl in [('vault', '予測Vault'), ('p400', '400–600µm 確率')]:
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
    ap = argparse.ArgumentParser(description='IPCL 予測Vault（SVR）/ 400–600µm 確率（LogReg）の学習')
    ap.add_argument('--data', default=os.path.join(HERE, 'IPCL9.xlsx'))
    ap.add_argument('--out', default=DEFAULT_BUNDLE)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        raise SystemExit(f'データファイルが見つかりません: {a.data}\n'
                         f'IPCL9.xlsx をこのフォルダに置くか、--data で場所を指定してください。')

    df = load_data(a.data)
    print('=' * 70)
    print('IPCL 予測Vault（SVR）/ 400–600µm 確率（LogReg）の学習')
    print('=' * 70)
    print(f'データ: {os.path.basename(a.data)}  {len(df)}眼 / {df["pid"].nunique()}人')
    print(f'特徴量: {FEATURES}')
    print(f'評価  : 患者単位 LOOCV（Leave-One-Patient-Out, {df["pid"].nunique()} fold）')

    nb_check = notebook_split_check(df)

    # ---- 全データで交差検証 → 最終モデル ----
    Xv, yv, gv = vault_xy(df)
    Xp, yp, gp = p400_xy(df)
    print('\n' + '=' * 70)
    print(f'【全データ】患者単位 LOOCV と最終モデル（Vault {len(yv)}眼 / 確率 {len(yp)}眼）')
    print('=' * 70)
    v_cv = reg_metrics(yv, lopo_oof(make_svr, Xv, yv, gv, proba=False))
    p_cv = cls_metrics(yp, lopo_oof(make_logreg, Xp, yp, gp, proba=True))
    print(f"  予測Vault SVR (Linear): R²={v_cv['R2']:.3f}  RMSE={v_cv['RMSE'] * 1000:.0f}µm  "
          f"MAE={v_cv['MAE'] * 1000:.0f}µm  ±100µm以内={v_cv['within_100um']:.1f}%")
    print(f"  400–600µm確率 LogReg  : AUC={p_cv['AUC']:.3f}  Brier={p_cv['Brier']:.3f}  "
          f"Youden閾値={p_cv['threshold']:.2f}（感度 {p_cv['sens']:.1f}% / 特異度 {p_cv['spec']:.1f}%）  "
          f"陽性割合={p_cv['prevalence']:.1f}%")

    svr_fitted = make_svr().fit(Xv, yv)
    svr = compact_linear_svr(svr_fitted)
    diff = float(np.abs(svr.predict(Xv) - svr_fitted.predict(Xv)).max())
    if diff > 1e-9:
        raise SystemExit(f'係数・切片だけの SVR の予測が元の SVR と一致しません（最大差 {diff:.2e} mm）')
    print(f'  SVR を係数・切片だけの形に置き換え（サポートベクター {len(svr_fitted["model"].support_)}眼分を除外、'
          f'予測の最大差 {diff:.1e} mm）')
    logreg = make_logreg().fit(Xp, yp)
    coef = {'vault_svr': dict(zip(FEATURES, map(float, svr['model'].coef_.ravel()))),
            'p400_logreg': dict(zip(FEATURES, map(float, logreg['model'].coef_.ravel())))}
    print('  係数（RobustScaler 後）')
    for k, c in coef.items():
        print(f'    {k:<12} ' + '  '.join(f'{f}={v:+.4f}' for f, v in c.items()))

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
        'source': '全データ.ipynb セル20（SVR (Linear)）・セル22（LogReg, c400_600）と同じ特徴量・前処理・設定',
        'versions': {'python': '.'.join(map(str, sys.version_info[:3])), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__},
        'vault': {'model': svr, 'name': 'SVR (Linear)', 'features': FEATURES, 'target': 'Vault2 (mm)',
                  'params': 'RobustScaler → SVR(C=1.0, epsilon=0.05, kernel="linear")',
                  'cv': v_cv, 'coef': coef['vault_svr']},
        'p400_600': {'model': logreg, 'name': 'LogReg', 'features': FEATURES,
                     'target': 'c400_600 == 1（Vault 400–600µm）',
                     'params': 'RobustScaler → LogisticRegression(C=1.0, max_iter=1000)',
                     'cv': p_cv, 'coef': coef['p400_logreg']},
        'notebook_split': nb_check,
        # 入力チェック用の要約統計のみ（個票データは保存しない）
        'feature_ranges': {c: {'min': float(df[c].min()), 'max': float(df[c].max()),
                               'median': float(df[c].median())} for c in FEATURES},
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(bundle, f)

    metrics = {k: v for k, v in bundle.items() if k not in ('vault', 'p400_600')}
    metrics['vault'] = {k: v for k, v in bundle['vault'].items() if k != 'model'}
    metrics['p400_600'] = {k: v for k, v in bundle['p400_600'].items() if k != 'model'}
    with open(os.path.join(os.path.dirname(a.out), 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print(f'保存: {a.out}  ({os.path.getsize(a.out) / 1024:.0f} KB)')
    print(f'      {os.path.join(os.path.dirname(a.out), "metrics.json")}')
    print('=' * 70)


if __name__ == '__main__':
    main()
