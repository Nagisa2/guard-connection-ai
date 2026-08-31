# タスク詳細: 位相非依存・位相推定波形復元（Griffin-Lim / PPG位相転用 / ゼロ位相）の実装と評価

## 概要
本タスクは、GUARD Connection AI において、モデルが出力する ECG STFT Magnitude スペクトログラムから、**正解位相（Oracle Phase）に依存しない実用的な時間軸波形（Waveform）復元手法** を実装・検証・定量比較する作業です。

## 目的・背景
- 現在の波形復元パイプラインは、正解 ECG の位相を組み合わせた「Oracle-Phase 復元（上限値）」に基づいています。
- 実際の臨床・実用フェーズでは正解 ECG の位相は未知であるため、PPG 入力のみから波形を再構成する手法（Phase-Free / Phase-Transfer 再構成）の確立が不可欠です。
- 本作業により、以下の 4 種の手法による波形復元パイプラインを実装し、実用環境における復元精度を網羅的に定量評価します：
  1. **Oracle-Phase 復元**: 正解 ECG 位相を使用（理論的上限制御）
  2. **PPG Phase Transfer 復元**: 入力 PPG の STFT 位相を転用
  3. **Griffin-Lim 反復復元**: 予測 Magnitude から STFT/iSTFT 反復により整合位相を推定
  4. **Zero Phase 復元**: 位相 0 によるベースライン

## 要件
1. **波形復元アルゴリズムの実装 (`src/guard_connection_ai/data/stft.py`)**:
   - `griffin_lim_reconstruction(magnitude, config, n_iter=32)`: Griffin-Lim 法による位相推定・波形復元。
   - `phase_transfer_reconstruction(magnitude, phase, config)`: 任意位相（PPG 位相や Zero 位相）を適用した波形復元。
2. **ユニットテストの追加 (`tests/test_phase_reconstruction.py`)**:
   - Griffin-Lim 法の収束性と合成信号復元テスト。
   - PPG 位相転用および Zero 位相復元の動作・エラーハンドリングテスト。
3. **位相復元比較評価スクリプトの実装 (`scripts/evaluate_phase_reconstruction.py`)**:
   - Validation Set (10 subjects / 950 segments) に対し、最良チェックポイントモデル（`L1 + SSIM` 等）を用いて 4 種の手法を一括評価。
   - 手法ごとの波形指標（MAE, RMSE, PRD, 相関係数）の Mean, Median, IQR, Subject Median を集計し、比較 CSV (`outputs/evaluation/phase_reconstruction_comparison.csv`) を出力。
4. **品質検査**:
   - pytest 全テストクリア
   - ruff check パス

## 成果物
- `docs/phase_reconstruction_evaluation/task.md`
- `docs/phase_reconstruction_evaluation/implementation_plan.md`
- `docs/phase_reconstruction_evaluation/walkthrough.md`
- `src/guard_connection_ai/data/stft.py` (機能拡張)
- `tests/test_phase_reconstruction.py` (新規テスト)
- `scripts/evaluate_phase_reconstruction.py` (新規評価スクリプト)
- `outputs/evaluation/phase_reconstruction_comparison.csv` (比較表)
