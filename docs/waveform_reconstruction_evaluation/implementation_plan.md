# 実装計画書: Waveform 再構成評価機能の実装と全モデル比較

GUARD Connection AI パイプラインにおける ECG STFT Magnitude 予測結果から Ground Truth Phase を用いた Oracle-Phase 波形復元（Inverse STFT）の評価機能をテスト化し、既存の学習済み全モデルに対する詳細な波形レベルの比較評価を自動化します。

## Proposed Changes

### Data / Metrics

#### [MODIFY] [stft.py](file:///e:/guard-connection-ai/src/guard_connection_ai/data/stft.py)
- `oracle_phase_reconstruction_metrics` や `inverse_stft` 周りのエラーハンドリング・入力型定義を確認し、テスト可能な堅牢性を確保します。

### Scripts

#### [MODIFY] [evaluate_waveform_reconstruction.py](file:///e:/guard-connection-ai/scripts/evaluate_waveform_reconstruction.py)
- 単一モデルの波形評価に加えて、Subject 単位での詳細統計量（中央値・IQR）を算出できるよう機能追加・出力フォーマット調整を行います。

#### [NEW] [compare_waveform_reconstruction.py](file:///e:/guard-connection-ai/scripts/compare_waveform_reconstruction.py)
- 存在する全チェックポイント（`resunet_attention_l1_best.pt`, `resunet_attention_l1_ssim_best.pt`, `resunet_attention_l1_ssim_frequency_best.pt` 等）を一元評価し、比較表 CSV (`outputs/evaluation/waveform_comparison.csv`) を生成します。

### Tests

#### [NEW] [test_evaluate_waveform.py](file:///e:/guard-connection-ai/tests/test_evaluate_waveform.py)
- `oracle_phase_reconstruction_metrics` および Waveform 評価パイプラインのユニットテストを新規追加し、短時間のダミー入力で MAE, RMSE, PRD, Correlation が正常に計算されることを検証します。

---

## Verification Plan

### Automated Tests
- pytest によるテスト実行:
  ```powershell
  $env:PYTHONPATH="e:\guard-connection-ai\src"; C:\Python\Python312\python.exe -m pytest
  ```
- Ruff による静的解析:
  ```powershell
  C:\Python\Python312\python.exe -m ruff check src/ tests/ scripts/
  ```

### Manual Verification
- `scripts/compare_waveform_reconstruction.py` を実行し、`outputs/evaluation/waveform_comparison.csv` が正しく生成されること、および全条件における Waveform 復元指標（MAE, RMSE, PRD, Correlation）の中央値および平均値を確認します。
