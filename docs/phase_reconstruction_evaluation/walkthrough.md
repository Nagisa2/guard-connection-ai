# 作業成果報告書 (Walkthrough): 位相非依存・位相推定波形復元の実装と定量評価

## 概要
本作業では、GUARD Connection AI において、モデルが予測する ECG STFT Magnitude スペクトログラムから、**正解位相（Oracle Phase）を用いずに時間軸波形（Waveform）を復元する実用手法（PPG位相転用、Griffin-Lim反復推定、ゼロ位相復元）** を実装し、Oracle-Phase（理論上限）と比較した復元精度を網羅的に定量評価・分析いたしました。

---

## 実装・変更内容

1. **波形復元機能の拡張 (`src/guard_connection_ai/data/stft.py`)**:
   - `griffin_lim_reconstruction`: STFT Magnitude のみから位相を反復推定する Griffin-Lim アルゴリズムの実装（初期位相指定にも対応）。
   - `phase_transfer_reconstruction`: 任意の位相（PPG 位相や Zero 位相）を適用して逆 STFT を行う汎用関数。
   - `zero_phase_reconstruction`: ゼロ位相復元関数。
   - `waveform_reconstruction_metrics`: 任意の波形ペアに対する評価指標（MAE, RMSE, PRD, 相関）算出関数。

2. **ユニットテストの追加 (`tests/test_phase_reconstruction.py`)**:
   - 正解位相による完全復元（相関 > 0.99, MAE < 1e-10）テスト。
   - Griffin-Lim の出力形状・収束性・初期位相指定テスト。
   - 形状不整合や不正パラメータに対するエラーハンドリングテスト（計 6 テスト追加）。

3. **位相復元比較評価スクリプトの実装 (`scripts/evaluate_phase_reconstruction.py`)**:
   - 最良モデル（`L1 + SSIM`）を用いて、Validation Set (10 subjects / 950 segments) に対し 5 種の手法を一括評価。
   - 結果を `outputs/evaluation/phase_reconstruction_comparison.csv` へ出力。

---

## 定量比較評価結果

Validation Set（10 被験者 / 950 セグメント）に対する位相復元手法の比較結果：

| 復元手法 | 使用位相情報 | MAE (Median) | RMSE (Median) | PRD (Median) | 波形相関 (Median) | Subject 波形相関 (Median) |
|---|---|---:|---:|---:|---:|---:|
| **Oracle Phase** (上限評価) | 正解 ECG 位相 | **0.161478** | **0.209201** | **49.01%** | **0.758237** | **0.760932** |
| **PPG Phase Transfer** | 入力 PPG 位相 | 0.243662 | 0.295574 | 71.64% | -0.029284 | -0.046482 |
| **Griffin-Lim (PPG Init)** | PPG 位相初期化 + 反復 | 0.283222 | 0.352837 | 80.71% | -0.031103 | -0.064653 |
| **Griffin-Lim (Zero Init)** | ゼロ位相初期化 + 反復 | 0.316712 | 0.380538 | 85.71% | -0.004968 | 0.004017 |
| **Zero Phase** | 位相 0 | 0.263285 | 0.331521 | 91.30% | -0.009825 | -0.010066 |

---

## 定量分析と極めて重要な知見

1. **ECG 波形における「位相（Phase）」の決定的一意性**:
   - 正解 ECG 位相を使用した場合、波形相関 `0.76` の高い精度で心電図波形が復元されますが、位相情報がない場合（Zero Phase や Griffin-Lim）は相関がほぼ 0 付近（無相関）に低下します。
   - 心電図の QRS 群の極性、急峻な立ち上がり、ST/T 波などの臨床的特徴は、各周波数成分の「厳密な時間的位相配置」に依存しており、Magnitude（振幅強度）のみでは時間軸上の干渉を正しく再構成できないことが実証されました。

2. **PPG 位相と ECG 位相の非互換性**:
   - PPG（血流脈波）と ECG（心筋脱分極）は基本周波数（心拍数）こそ同期しているものの、波形形状や高調波の位相差（時間遅延・伝播特性）が大きく異なるため、PPG 位相をそのまま転用しても心電図波形には変換されません。

3. **今後の最重要開発方針の確立**:
   本実験により、実用的な PPG ➔ ECG 時間波形生成パイプラインを実現するための明確なロードマップが確定しました：
   - **方針 1: 複素 STFT 直接推定モデル**:
     - モデルの出力を Magnitude のみではなく、実部（Real）と虚部（Imaginary）、または Magnitude + Phase の 2 チャンネル出力とし、モデル内部で位相を直接回帰する。
   - **方針 2: 1D 波形エンドツーエンド直接回帰モデル (1D-UNet / WaveNet 等)**:
     - STFT 変換を介さず、1D PPG 波形から 1D ECG 波形を時間軸上で直接学習・推論するアーキテクチャの導入。

---

## 検証結果

- **全ユニットテスト**: `pytest` にて全 **41 passed in 9.12s** を達成。
- **コードスタイル**: `ruff check` にて **All checks passed!** を確認。
- **成果物**:
  - `docs/phase_reconstruction_evaluation/task.md`
  - `docs/phase_reconstruction_evaluation/implementation_plan.md`
  - `docs/phase_reconstruction_evaluation/walkthrough.md`
  - `src/guard_connection_ai/data/stft.py` (拡張)
  - `tests/test_phase_reconstruction.py` (新規テスト)
  - `scripts/evaluate_phase_reconstruction.py` (新規スクリプト)
  - `outputs/evaluation/phase_reconstruction_comparison.csv` (比較表)
