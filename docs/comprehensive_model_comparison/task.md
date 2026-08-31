# タスク詳細: 本格モデル比較（複数 Loss・複数 Seed・十分な Epoch による各種 Baseline 性能の最終比較評価）

## 概要
本タスクは、引き継ぎ文書（`HANDOFF.md`）の最終目標である **非敵対的 Baseline（Residual Attention U-Net）モデルの性能限界と特性を確定させるための本格モデル比較評価** です。
3 つの損失関数（`L1`, `L1+SSIM`, `L1+SSIM+Frequency`）に対し、複数 Seed（`seed=42`, `seed=43`）および十分な Epoch（`epochs=15`, `patience=5`）での系統的な訓練と、STFT 領域・Waveform（時間軸波形）領域の統合評価を実施します。

## 目的・背景
- これまでの実験は短時間の初期プロトタイピング（10 epochs, patience 3）で行われており、複数 Loss と複数 Seed を組み合わせた網羅的な学術的・定量評価が未完成でした。
- 本作業により、十分な訓練エポック下での各 Loss 関数の収束特性、複数 Seed による再現性・分散、ならびにスペクトログラム領域指標（Val Loss/MAE/RMSE/PRD/Correlation）と時間軸波形指標（MAE/RMSE/PRD/Correlation Median & IQR）を網羅した最終比較テーブルを構築します。

## 要件
1. **系統的実験の実行とチェックポイント収集**:
   * `L1` (seed 42, 43)
   * `L1+SSIM` (seed 42, 43)
   * `L1+SSIM+Frequency` (seed 42, 43)
   * 各設定で `--epochs 15 --patience 5` を用いて十分な学習を実行・ベストモデルを生成。
2. **統合比較評価スクリプトの作成 (`scripts/comprehensive_comparison.py`)**:
   * 生成された全チェックポイントモデルに対し、STFT 領域評価指標および Waveform 領域指標（全体中央値、Subject 間中央値・IQR）を一度に評価・統合。
   * 最終統合結果 CSV (`outputs/evaluation/comprehensive_model_comparison.csv`) を出力。
3. **ユニットテスト・コード品質クリア**:
   * pytest 全テストクリア
   * ruff check パス

## 成果物
- `docs/comprehensive_model_comparison/task.md`
- `docs/comprehensive_model_comparison/implementation_plan.md`
- `docs/comprehensive_model_comparison/walkthrough.md`
- `scripts/comprehensive_comparison.py` (新規統合比較スクリプト)
- `outputs/evaluation/comprehensive_model_comparison.csv` (全モデル統合比較表)
