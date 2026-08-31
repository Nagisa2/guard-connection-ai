# 実装計画書: 位相非依存・位相推定波形復元（Griffin-Lim / PPG位相転用 / ゼロ位相）の実装と評価

ECG STFT Magnitude スペクトログラムから、正解 ECG 位相を必要としない実用的な時間軸波形（Waveform）復元手法（Griffin-Lim法、PPG位相転用、ゼロ位相復元）を実装し、Oracle-Phase（理論上限）と比較した復元精度を網羅的に定量評価します。

---

## ユーザー確認事項 (User Review Required)

> [!NOTE]
> Griffin-Lim アルゴリズムは STFT と iSTFT の反復処理（デフォルト 32 イテレーション）を行います。NumPy / SciPy のベクトル化処理により、高速かつ安定した収束を実現します。

---

## 提案する変更内容 (Proposed Changes)

### 1. STFT モジュールの拡張
#### [MODIFY] [stft.py](file:///f:/guard-connection-ai/src/guard_connection_ai/data/stft.py)
- `griffin_lim_reconstruction(magnitude: np.ndarray, config: STFTConfig | None = None, n_iter: int = 32, momentum: float = 0.99) -> np.ndarray`:
  - 予測 Magnitude のみから位相を反復推定し、時間波形を復元するアルゴリズム。
- `phase_transfer_reconstruction(magnitude: np.ndarray, phase: np.ndarray, config: STFTConfig | None = None) -> np.ndarray`:
  - 任意の位相情報（PPG 位相や Zero 位相）を適用して逆 STFT を行う汎用関数。

### 2. ユニットテストの追加
#### [NEW] [test_phase_reconstruction.py](file:///f:/guard-connection-ai/tests/test_phase_reconstruction.py)
- 合成信号に対する Griffin-Lim 法の再構成動作検証（相関 > 0.85）。
- PPG 位相転用および Zero 位相復元のテンソル整合性・エラー処理テスト。

### 3. 位相復元比較評価スクリプトの作成
#### [NEW] [evaluate_phase_reconstruction.py](file:///f:/guard-connection-ai/scripts/evaluate_phase_reconstruction.py)
- 最良チェックポイント（例: `resunet_attention_l1_ssim_seed42_best.pt`, `resunet_attention_l1_ssim_seed43_best.pt` 等）に対し、以下の 4 手法で時間波形を復元・評価：
  1. `Oracle-Phase` (正解 ECG 位相)
  2. `PPG-Phase Transfer` (入力 PPG 位相)
  3. `Griffin-Lim` (位相反復推定)
  4. `Zero-Phase` (位相 0)
- Validation Set (10 subjects / 950 segments) における MAE, RMSE, PRD, 相関係数の統計値（Overall Mean/Median/IQR, Subject Median）を算出し、比較 CSV (`outputs/evaluation/phase_reconstruction_comparison.csv`) を出力。

---

## 検証計画 (Verification Plan)

### 1. 自動テストとコード品質
- pytest によるテスト実行:
  ```powershell
  python -m pytest
  ```
- Ruff による静的コード解析:
  ```powershell
  python -m ruff check src/ tests/ scripts/evaluate_phase_reconstruction.py
  ```

### 2. 定量評価の実行
- `scripts/evaluate_phase_reconstruction.py` を実行し、全手法の比較結果表を生成。
- 正解位相なしでの波形復元の可能性・実用限界を分析し、`docs/phase_reconstruction_evaluation/walkthrough.md` を作成。
