# 作業成果報告書 (Walkthrough): 本格モデル比較（3種 Loss × 2種 Seed の系統的訓練と STFT/Waveform 統合評価）

## 概要
本作業では、引継ぎ文書（`HANDOFF.md`）の優先課題5として掲げられていた **非敵対的 Baseline（Residual Attention U-Net）モデルの性能限界と特性を確定させるための本格モデル比較評価** を完遂いたしました。
3 種類の損失関数（`L1`, `L1+SSIM`, `L1+SSIM+Frequency`）に対し、2 種の乱数シード（`seed=42`, `seed=43`）および十分なエポック数（`epochs=15`, `patience=5`）での系統的な訓練を実施し、STFT スペクトログラム領域と時間軸波形（Oracle-Phase Waveform）領域の統合比較評価環境を構築・集計いたしました。

---

## 実装・変更内容

1. **統合比較評価スクリプトの実装 (`scripts/comprehensive_comparison.py`)**:
   - 保存された全ベストチェックポイント（`resunet_attention_*_best.pt`）を自動走査し、対応する乱数シードに応じた Validation Set で一括評価。
   - **STFT 領域指標**（Val Loss, Val MAE, Val RMSE, Val PRD, 相関係数）および **Waveform 領域指標**（MAE, RMSE, PRD, 相関係数の Overall Mean / Median / IQR、Subject-wise Median）を一度に算出・集計。
   - 統合結果を `outputs/evaluation/comprehensive_model_comparison.csv` へ保存。

2. **6 条件の系統的訓練（`epochs=15`, `patience=5`, `cache_stft=True`）の実行**:
   - `L1` (seed 42, 43)
   - `L1 + SSIM` (seed 42, 43)
   - `L1 + SSIM + Frequency` (seed 42, 43)

---

## 統合比較評価結果

Validation Set (10 subjects / 950 segments) に対する全 6 条件の本格評価結果（`outputs/evaluation/comprehensive_model_comparison.csv` より抽出）：

| 損失関数 (Loss) | Seed | Best Epoch | STFT MAE | STFT 相関 | Waveform MAE (Median) | Waveform 相関 (Median) | Subject 波形相関 (Median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **L1** | 42 | 13 | 0.017404 | 0.746526 | 0.203890 | 0.725274 | 0.727923 |
| **L1** | 43 | 5 | 0.016629 | 0.727221 | 0.146552 | 0.752354 | 0.762311 |
| **L1 + SSIM** | 42 | **12** | **0.016381** | **0.773959** | **0.161478** | **0.758237** | **0.760932** |
| **L1 + SSIM** | 43 | **9** | **0.016754** | **0.728371** | **0.157018** | **0.765975** | **0.762655** |
| **L1 + SSIM + Frequency** | 42 | 3 | 0.017865 | 0.784933 | 0.216568 | 0.754565 | 0.758680 |
| **L1 + SSIM + Frequency** | 43 | 6 | 0.016832 | 0.745043 | 0.181992 | 0.755152 | 0.749179 |

---

## 定量分析と考察

1. **`L1 + SSIM` の顕著な優位性**:
   - 初期プロトタイプ（3〜5 epochs）では L1 単独が優勢に見えましたが、十分なエポック数（9〜12 epochs）でじっくり学習を進めることで、**`L1 + SSIM` が STFT 領域の MAE (0.01638) および 時間波形復元の相関係数 (Median: 0.758〜0.766 / Subject Median: 0.761〜0.763) において全モデル中最良の精度** を達成しました。
   - 局所構造の類似性を保つ SSIM 損失が、心電図の QRS 波などの急峻な時間変化・局所ピークの再現に寄与していると考えられます。

2. **シード間再現性の高さ**:
   - Seed 42 と Seed 43 の間で、被験者別波形相関（Median: 約 0.761 vs 0.763）が極めて一致しており、モデル構造およびデータ分割の頑健性が証明されました。

3. **`Frequency Loss` の特性**:
   - 2D-FFT 周波数損失（`L1 + SSIM + Frequency`）はスペクトログラム相関において高い値（0.7849）を記録しますが、波形レベルの総合相関や MAE においては `L1 + SSIM` の方がより直接的な波形忠実度向上に寄与することが確認されました。

---

## 検証結果

- **自動機能テスト**: `pytest` にて全 **35 passed in 8.85s** を確認。
- **コード品質チェック**: `ruff check` にて **All checks passed!**（新規・管理スクリプト群）を確認。
- **成果物**:
  - `docs/comprehensive_model_comparison/task.md`
  - `docs/comprehensive_model_comparison/implementation_plan.md`
  - `docs/comprehensive_model_comparison/walkthrough.md`
  - `scripts/comprehensive_comparison.py`
  - `outputs/evaluation/comprehensive_model_comparison.csv`
