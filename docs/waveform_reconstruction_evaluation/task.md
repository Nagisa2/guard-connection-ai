# タスク詳細: Waveform 再構成評価の実装と全モデル比較

## 概要
本タスクは、GUARD Connection AI パイプラインにおける推論出力（ECG STFT Magnitude スペクトログラム）から、Ground Truth の STFT Phase 情報を用いた Oracle-Phase Inverse STFT によって計算される **時間軸波形（Waveform）復元精度** を評価・集計・テスト化する作業です。

## 目的・背景
- モデルの出力は 2D STFT Magnitude スペクトログラムですが、心電図（ECG）としての最終評価には時間軸における波形精度（MAE, RMSE, PRD, Correlation）が極めて重要です。
- 現状は Phase の予測モデルを含まないため、Ground Truth ECG の STFT Phase を組み合わせた **Oracle-Phase 再構成**を「復元精度の上限値 (Upper Bound)」として評価指標に組み込みます。

## 要件
1. **ユニットテストの追加**:
   - Oracle-Phase による波形再構成処理および評価指標計算 (`oracle_phase_reconstruction_metrics`) の自動テストを `tests/test_evaluate_waveform.py` として作成・実装。
2. **全チェックポイントの比較評価機能**:
   - `L1`, `L1+SSIM`, `L1+SSIM+Frequency` などの全チェックポイントモデルに対し、一括で Oracle-Phase 波形復元精度を評価する比較スクリプト (`scripts/compare_waveform_reconstruction.py`) の作成。
   - Validation Set における全 Segment 指標、および Subject ごとの中央値 (Median)・四分位範囲 (IQR) の集計・CSV 出力機能の実装。
3. **既存パイプライン・品質検査のパス**:
   - pytest (全テストパス)
   - Ruff (コードスタイル・リンターチェック)

## 成果物
- `docs/waveform_reconstruction_evaluation/task.md`
- `docs/waveform_reconstruction_evaluation/implementation_plan.md`
- `docs/waveform_reconstruction_evaluation/walkthrough.md`
- `tests/test_evaluate_waveform.py` (新規テスト)
- `scripts/compare_waveform_reconstruction.py` (新規比較評価スクリプト)
- `outputs/evaluation/waveform_comparison.csv` (全モデル評価比較)
