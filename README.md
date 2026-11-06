# IPCL Vault Predictor（Ensemble / LogReg + SVM）— Streamlit Cloud 版

術前の測定値（年齢・LV・ACD・ACW）から、ACW より大きい最小の ICL サイズから 0.25mm 刻みで 5 サイズについて、

- **予測Vault**（アンサンブル: SVR (Linear)・Ridge・重回帰の予測の平均）
- **Vault が 750µm 以下になる確率**（ロジスティック回帰と SVM の確率の平均）
- **Vault が 400〜600µm に入る確率**（ロジスティック回帰）
- **正しいサイズの確率**（アンサンブル: 多項ロジスティック・XGBoost・CatBoost の確率の平均。5 サイズの合計 100%）

を表示するウェブアプリです。

既存の `IPCL_Streamlit_Cloud`（サイズ推奨アプリ）とは別のアプリです。**別の GitHub リポジトリ**にして
デプロイしてください（同じリポジトリに入れると `streamlit_app.py` が上書きされます）。

---

## ⚠️ 公開範囲について（先にお読みください）

Streamlit Community Cloud にデプロイすると、**URL を知っている人が誰でも開ける状態**が既定です。
院内だけで使う場合は、次のどちらかを必ず設定してください。

- デプロイ後に **Manage app → Settings → Sharing** で閲覧できる人を限定する（招待したメールアドレスのみ）
- または GitHub リポジトリを **Private** にしたうえで、上記の Sharing も限定する

このリポジトリに**元データ（xlsx）は含めません**（`.gitignore` で除外済み）。
`models/ipcl_vault_bundle.pkl` には学習済みモデルと要約統計（各変数の最小・最大・中央値）が入っています。
予測Vault のアンサンブルに含まれる SVR は、3モデルとも線形なので係数・切片〈3モデルの平均〉だけの形に置き換えて保存しています。
**ただし、確率のモデルに含まれる SVM（RBF カーネル）は、学習眼の一部〈サポートベクター, 224眼分〉の LV・ACD・ACW・サイズ・年齢（標準化後の値。元の値に戻せます）と、その眼の Vault が 750µm を超えたかどうかを内部に持ちます。ID や Vault の値そのものは含みません。リポジトリを公開のままにすると、これらの値も誰でも読める状態になります**（2026-11-02 にこの前提で公開のままとすることを決定）。
正しいサイズの確率に含まれる XGBoost・CatBoost は、学習後に木（分岐に使う変数・しきい値・葉の値）だけを取り出し、numpy で同じ確率を計算する形にして保存しています（元のモデルとの差は 10⁻⁶ 未満で、アプリの実行に xgboost・catboost は不要です）。

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

`全データ.ipynb` の次の4つのモデルと**同じ特徴量・前処理・設定**で、**全症例（IPCL9.xlsx, 278眼 / 142人）**で学習し直したものです（訓練群・ホールドアウト群には分けず、全症例を学習に使っています。性能は全症例での患者単位 LOOCV で推定）。

| | ノートブック | 設定 | 患者単位 LOOCV（全データ） |
|---|---|---|---|
| 予測Vault | セル20 `Ensemble (SVR (Linear)+Ridge+MLR)` | RobustScaler → SVR(C=1.0, epsilon=0.05, kernel='linear')・Ridge(alpha=1.0)・LinearRegression の予測の平均、目的変数 Vault2 (mm) | R² 0.244 / RMSE 145µm / MAE 107µm |
| 750µm 以下の確率 | セル31 `LogReg` と `SVM_RBF`（Vault > 0.75mm）の確率の平均 | RobustScaler → LogisticRegression(C=1.0, max_iter=1000) と SVC(C=1.0, kernel='rbf', gamma=0.03, class_weight='balanced', probability=True)。ノートブックと同じく Vault2 > 0.75 を陽性として学習し、1 − その確率（= 750µm 以下になる確率。750µm ちょうどは「以下」）を表示 | AUC 0.715 / Youden 閾値 0.68（750µm 以下を陽性として感度 73% / 特異度 69%）。単独では LogReg 0.717・SVM_RBF 0.707 |
| 400〜600µm に入る確率 | セル22 `LogReg`（①狭義 c400_600） | RobustScaler → LogisticRegression(C=1.0, max_iter=1000)、目的変数 c400_600 == 1（Vault 400〜600µm。400µm・600µm ちょうどを含む） | AUC 0.698 / Youden 閾値 0.27（感度 76% / 特異度 58%） |
| 正しいサイズの確率 | セル30 `アンサンブル（多項ロジスティック＋多クラス XGBoost＋多クラス CatBoost）` | 各モデル RobustScaler → 分類器（LogisticRegression(C=1.0, max_iter=2000)・XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)・CatBoostClassifier(iterations=100, depth=3, learning_rate=0.05, l2_leaf_reg=3.0)）の確率の平均。目的変数は段階 =（正しいサイズ − ACW より大きい最小サイズ）/ 0.25、特徴量 ACW・LV・ACD・age・gap（= 最小サイズ − ACW） | 予測（確率分布の中央値）と正しいサイズの差 MAE 0.145mm / 完全一致 51% / ±0.25mm 以内 91%。単独では多項ロジスティック 0.143・XGBoost 0.153・CatBoost 0.146mm |

- 特徴量（正しいサイズ以外の3つ）: LV, ACD, size, ACW, age（単位 mm / 歳）。乱数シードはノートブックの GLOBAL_SEED（2025）と同じ
- 2026-11-01 に修正した IPCL9.xlsx で学習しています（Vault: no 3 を 602→600µm、no 44 を 601→600µm に変更。
  どちらも 750µm 以下なので、750µm 超かどうかは変わりません）。c400_600 列も同じ日に no 121・129・130・133・144・206 を修正しています。
  400〜600µm に入る確率（セル22 LogReg）はこの修正の影響を受けるため、ノートブックを再実行するまでは「不一致」と表示されます
  （修正前の値に戻すと、訓練群 CV・ホールドアウトの AUC・閾値・感度・特異度がノートブックの出力と一致することを確認済みです）。
  750µm 以下の確率の2モデル（セル31 LogReg・SVM_RBF, 750µm 超を陽性として学習）は、どちらも現在のデータのままでノートブック（10/30 実行）の出力
  （訓練群 CV・ホールドアウトの AUC・閾値・感度・特異度）と一致します。予測Vault（セル20 Ensemble）は
  修正前の値に戻すと出力（R²・RMSE・MAE）と一致し、アンサンブル内の SVR は `vault_ml_final.pkl` の SVR と
  予測値が完全に一致することを確認済みです。
  正しいサイズの確率（セル30 アンサンブル）は、現在のデータのままでノートブックの出力（訓練群 CV の MAE・完全一致・±0.25mm 以内・
  LogLoss・正解サイズの確率の平均、ホールドアウトの MAE・完全一致・±0.25mm 以内）と一致し、ノートブックの使用例
  （ホールドアウト1眼目の 5 サイズの確率）も一致することを確認済みです。
  アプリは全データで学習し直したモデルなので、ノートブックの（訓練群で学習した）モデルとは予測値が少し異なります
- モデルは **scikit-learn 1.7.2**（Python 3.13）で作成しています。Mac 版・Windows 版（scikit-learn 1.6.1）と
  全 278 眼で予測値が完全に一致することを確認済みです

**確率の読み方の注意**: 予測Vaultと 750µm 以下の確率は別々に学習したモデルです。サイズを大きくすると予測Vaultは上がり、750µm 以下の確率は下がりますが、
予測Vaultが 750µm ちょうどでも確率が 50% になるとは限りません。両方を合わせて見てください。
400〜600µm に入る確率も別に学習したモデルで、ロジスティック回帰（線形）のため、サイズに対して一方向にしか変化しません（サイズを大きくすると下がります）。
正しいサイズの確率は、Vault が特定の範囲に入る確率ではなく、表示する 5 サイズのうちどれが「正しいサイズ」かの確率です（合計 100%）。

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

⚠️ **本アプリは研究目的の参考情報です。** 予測Vaultの誤差は RMSE で約145µm、MAE で約107µm あり、
750µm 以下になる確率の判別能は AUC 0.72、400〜600µm に入る確率は AUC 0.70 程度です。
正しいサイズの予測が実際の正しいサイズと一致したのは約 51%（±0.25mm 以内は約 91%）です。最終的なレンズ選択は必ず医師の判断で行ってください。
