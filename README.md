# IPCL Vault Predictor（SVR / LogReg）— Streamlit Cloud 版

術前の測定値（年齢・LV・ACD・ACW）から、ACW より大きい最小の ICL サイズから 0.25mm 刻みで 5 サイズについて、

- **予測Vault**（SVR (Linear)）
- **Vault が 400–600µm に入る確率**（ロジスティック回帰）

を表示するウェブアプリです。

既存の `IPCL_Streamlit_Cloud`（サイズ推奨アプリ）とは別のアプリです。**別の GitHub リポジトリ**にして
デプロイしてください（同じリポジトリに入れると `streamlit_app.py` が上書きされます）。

---

## ⚠️ 公開範囲について（先にお読みください）

Streamlit Community Cloud にデプロイすると、**URL を知っている人が誰でも開ける状態**が既定です。
院内だけで使う場合は、次のどちらかを必ず設定してください。

- デプロイ後に **Manage app → Settings → Sharing** で閲覧できる人を限定する（招待したメールアドレスのみ）
- または GitHub リポジトリを **Private** にしたうえで、上記の Sharing も限定する

このリポジトリに**患者データ（xlsx）は含めません**（`.gitignore` で除外済み）。
`models/ipcl_vault_bundle.pkl` には学習済みモデルと要約統計（各変数の最小・最大・中央値）だけが入っており、
患者ごとのデータは含まれていません（SVR は本来、学習眼の測定値の一部〈サポートベクター〉を内部に持つため、同じ予測になる係数・切片だけの形に置き換えて保存しています）。

---

## デプロイ手順

1. **GitHub に新しいリポジトリを作る**（院内利用なら Private を推奨）
2. **このフォルダの中身をすべてアップロードする**
   - `models/ipcl_vault_bundle.pkl` も必ず含めてください（アプリの動作に必要です）
   - GitHub の Web UI からアップロードする場合は、`models` フォルダを作ってからその中に入れてください
3. **https://share.streamlit.io** を開き、GitHub アカウントで連携する
4. 「Create app」→ リポジトリを選び、次を指定する
   - **Main file path**: `streamlit_app.py`
   - **Advanced settings → Python version**: 指定しなくても動きます（`3.11`〜`3.14` のどれでも可）
5. 「Deploy」を押す
6. 起動したら **Manage app → Settings → Sharing** で公開範囲を設定する

このアプリは scikit-learn だけで動きます（CatBoost・XGBoost・LightGBM は不要）。そのため `packages.txt` はありません。

---

## ファイル構成

```
├── streamlit_app.py       画面（Streamlit Cloud の Main file path に指定）
├── ipcl_vault_predict.py  予測の計算
├── train_model.py         モデルの学習（手元で実行するスクリプト）
├── requirements.txt       アプリの実行に必要なライブラリ（モデル読み込みに関わる分だけ固定）
├── requirements-train.txt 学習し直すときだけ必要なライブラリ（クラウドでは未使用）
├── .gitignore             患者データを誤って上げないための設定
├── .streamlit/config.toml 画面の配色
└── models/
    ├── ipcl_vault_bundle.pkl  学習済みモデル（全症例で学習）
    └── metrics.json           交差検証の結果
```

---

## モデルについて

`全データ.ipynb` の次の2モデルと**同じ特徴量・前処理・設定**で、**全データ（IPCL9.xlsx, 278眼 / 142人）**で学習し直したものです。

| | ノートブック | 設定 | 患者単位 LOOCV（全データ） |
|---|---|---|---|
| 予測Vault | セル20 `SVR (Linear)` | RobustScaler → SVR(C=1.0, epsilon=0.05, kernel='linear')、目的変数 Vault2 (mm) | R² 0.254 / RMSE 144µm / MAE 108µm |
| 400–600µm の確率 | セル22 `LogReg`（狭義） | RobustScaler → LogisticRegression(C=1.0, max_iter=1000)、目的変数 c400_600 == 1 | AUC 0.698 / Youden 閾値 0.27（感度 76% / 特異度 58%） |

- 特徴量（共通）: LV, ACD, size, ACW, age（単位 mm / 歳）。乱数シードはノートブックの GLOBAL_SEED（2025）と同じ
- 2026-11-01 に修正した IPCL9.xlsx で学習しています（c400_600: no 129・130・133・206 を 1、no 144 を 2、no 121 を 0 に変更。
  Vault: no 3 を 602→600µm、no 44 を 601→600µm に変更）。これで c400_600 と Vault2 の判定は全眼で一致します。
  修正前の値に戻して訓練群だけで学習すると、ノートブック（10/30 実行）が保存した `vault_ml_final.pkl`・
  `vault_binary_prediction_v2.pkl` と係数・予測値が完全に一致することを確認済みです。
  アプリは全データで学習し直したモデルなので、ノートブックの（訓練群で学習した）モデルとは予測値が少し異なります
- モデルは **scikit-learn 1.7.2**（Python 3.13）で作成しています。Mac 版・Windows 版（scikit-learn 1.6.1）と
  全 278 眼で予測値が完全に一致することを確認済みです

**確率の読み方の注意**: ロジスティック回帰は各変数について単調なモデルなので、サイズを小さくするほど
400–600µm に入る確率は上がり続けます（予測Vaultが 400µm を下回るサイズでも下がりません）。
サイズの比較には予測Vaultと合わせて使ってください。

### データを更新してモデルを作り直す場合

クラウド上では学習できません（患者データを置かないため）。手元のパソコンで作り直してください。

```bash
pip install -r requirements-train.txt
python train_model.py --data /path/to/IPCL9.xlsx
```

`models/ipcl_vault_bundle.pkl` と `models/metrics.json` が更新されるので、その2ファイルを GitHub に push すると
Streamlit Cloud が自動で再起動して新しいモデルになります。学習は requirements.txt と同じ版
（scikit-learn 1.7.2）で行ってください。

---

## うまく動かないとき

| 症状 | 対処 |
|---|---|
| 「モデルファイルが見つかりません」 | `models/ipcl_vault_bundle.pkl` がリポジトリに含まれているか確認（GitHub 上でフォルダ構成を確認） |
| インストールに何十分もかかる | `requirements.txt` の scikit-learn が 1.7.2 以上か確認（古い版は Python 3.14 用のパッケージが無く、ソースからのビルドが走ります） |
| `.gitignore` や `.streamlit` が GitHub に無い | 先頭がドットのファイルは Finder やブラウザで見えないため、アップロード時に漏れがちです。`.gitignore` は患者データを誤って上げないための設定なので、追加をおすすめします |
| 「Oh no. Error running app.」と表示される | `requirements.txt` で Streamlit のバージョンを固定していないか確認。固定している場合は `streamlit` とだけ書いて Commit し、Reboot app |
| 「バージョンが異なるライブラリがあります」と表示される | モデル読み込みに関わる行（numpy〜threadpoolctl）は変更せずにそのまま使ってください |

---

⚠️ **本アプリは研究目的の参考情報です。** 予測Vaultの誤差は RMSE で約140µm あり、
400–600µm に入る確率の判別能は AUC 0.70 程度です。最終的なレンズ選択は必ず医師の判断で行ってください。
