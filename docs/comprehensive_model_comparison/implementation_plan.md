# 実装計画書: 本格モデル比較（3種 Loss × 2種 Seed による十分な Epoch 学習と STFT/Waveform 統合評価）

Residual Attention U-Net において、3 種類の損失関数（`L1`, `L1+SSIM`, `L1+SSIM+Frequency`）と 2 種類の乱数シード（`seed=42`, `seed=43`）の全 6 条件に対し、十分なエポック数（`epochs=15`, `patience=5`）での系統的な訓練を実施し、STFT スペクトログラム領域および Oracle-Phase Waveform（時間軸波形）領域の統合比較評価を行います。

---

## ユーザー確認事項 (User Review Required)

> [!NOTE]
> 現在の環境は CPU 実行環境です。先行タスクで実装された `--cache-stft`（STFT インメモリキャッシュ）を活用することで、各エポックのデータロードと STFT 変換のオーバーヘッドを最小化し、CPU 上で効率的に 6 パターンの訓練を完了させます。

---

## 提案する変更内容 (Proposed Changes)

### 1. 総合評価・比較スクリプトの作成
#### [NEW] [comprehensive_comparison.py](file:///f:/guard-connection-ai/scripts/comprehensive_comparison.py)
- [checkpoints/](file:///f:/guard-connection-ai/checkpoints) 配下に保存される全ベストチェックポイント（`resunet_attention_*_best.pt`）を走査・自動読み込み。
- モデルごとに対応するシードで Validation Dataset を構築。
- **STFT 領域評価**:
  - Validation Loss, MAE, RMSE, PRD, 相関係数 (Correlation) の算出。
- **Waveform 領域評価**:
  - Ground Truth Phase を組み合わせた Oracle-Phase Inverse STFT による波形復元指標（MAE, RMSE, PRD, 相関係数）の全体平均 (Mean)、標準偏差 (Std)、全体中央値 (Median)、四分位範囲 (IQR)、および被験者別中央値 (Subject Median) を算出。
- 全結果を結合した統合比較表を [outputs/evaluation/comprehensive_model_comparison.csv](file:///f:/guard-connection-ai/outputs/evaluation/comprehensive_model_comparison.csv) に保存・整形出力。

### 2. 6パターンの本格学習実行
- 以下の 6 条件で [train.py](file:///f:/guard-connection-ai/scripts/train.py) を順次実行:
  1. `L1` (seed 42): `--epochs 15 --patience 5 --cache-stft --seed 42`
  2. `L1` (seed 43): `--epochs 15 --patience 5 --cache-stft --seed 43`
  3. `L1 + SSIM` (seed 42): `--epochs 15 --patience 5 --cache-stft --ssim-weight 0.1 --seed 42`
  4. `L1 + SSIM` (seed 43): `--epochs 15 --patience 5 --cache-stft --ssim-weight 0.1 --seed 43`
  5. `L1 + SSIM + Freq` (seed 42): `--epochs 15 --patience 5 --cache-stft --ssim-weight 0.1 --frequency-weight 0.001 --seed 42`
  6. `L1 + SSIM + Freq` (seed 43): `--epochs 15 --patience 5 --cache-stft --ssim-weight 0.1 --frequency-weight 0.001 --seed 43`

---

## 検証計画 (Verification Plan)

### 1. 自動テストとコード品質
- pytest によるテスト実行:
  ```powershell
  python -m pytest
  ```
- Ruff による静的コード解析:
  ```powershell
  python -m ruff check scripts/comprehensive_comparison.py
  ```

### 2. 評価結果の検証
- 全 6 種のチェックポイント（`resunet_attention_*_best.pt`）が正常に更新・保存されていることを確認。
- `scripts/comprehensive_comparison.py` を実行し、全指標が網羅された比較表 CSV が出力されることを確認。
- 結果の考察（L1 Loss の優位性やシード依存性、各損失関数の特徴）をまとめ、`docs/comprehensive_model_comparison/walkthrough.md` を作成。
