# GUARD Connection AI 引継ぎ文書

## 1. この文書の目的

このリポジトリは、GUARD Connection の前段 AI、つまり PPG から ECG の時間周波数表現を推定する研究用コードです。

次の担当者は、この文書を読んだ後、既存コードと成果物を確認してから作業を再開してください。既存設計を大きく変更せず、まずテストを実行してください。

## 2. 現在の到達点

以下の一連の処理が実装され、BIDMC 実データで動作確認済みです。

```text
BIDMC CSV
  -> subject-wise split
  -> 10 秒 segment / 50% overlap
  -> configurable preprocessing
  -> STFT magnitude
  -> PyTorch Dataset / DataLoader
  -> Residual Attention U-Net
  -> L1 / L1+SSIM / Frequency loss
  -> MLflow logging
  -> checkpoint / validation evaluation
```

現在の全テストは `26 passed`、変更対象コードの Ruff は `All checks passed` です。

## 3. データと split

主データは以下です。

```text
data/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/
```

- Subject: 53
- Sampling rate: 125 Hz
- 信号列: `PLETH` (PPG), `II` (ECG)
- 各 subject: 60001 samples
- NaN / Inf: 既存検査では確認されていない

Subject-wise split は [configs/data_split.yaml](../configs/data_split.yaml) に固定保存されています。

- Train: 43 subjects
- Validation: 10 subjects
- Seed: 42
- Train/Validation の subject 重複なし

セグメント条件:

- Window: 10 秒 = 1250 samples
- Overlap: 50%
- Hop: 625 samples
- 1 subject: 95 segments
- 全体: 5035 segments
- Train: 4085 segments
- Validation: 950 segments

segment metadata は以下にあります。

```text
outputs/segmentation/segment_index.csv
```

## 4. 実装ファイル

### Data

- [dataset.py](../src/guard_connection_ai/data/dataset.py)
  - BIDMC CSV loader
  - `BIDMCSTFTDataset`
  - PPG magnitude を `input`、ECG magnitude を `target` として返す
- [segmentation.py](../src/guard_connection_ai/data/segmentation.py)
  - paired signal segmentation
  - segment metadata index
- [preprocessing.py](../src/guard_connection_ai/data/preprocessing.py)
  - `none`
  - `detrend`
  - `bandpass`
  - `segment_zscore`
- [stft.py](../src/guard_connection_ai/data/stft.py)
  - complex STFT
  - magnitude / phase / real / imaginary
  - inverse STFT
  - reconstruction metrics
- [config.py](../src/guard_connection_ai/data/config.py)
  - YAML 読み込み
  - STFTConfig 生成
- [json_metadata.py](../src/guard_connection_ai/data/json_metadata.py)
  - `json_Data` の metadata 専用 loader

### Model / Loss / Metrics

- [resunet_attention.py](../src/guard_connection_ai/models/resunet_attention.py)
  - Residual Attention U-Net
  - `[B, 1, 33, 41] -> [B, 1, 33, 41]`
- [reconstruction.py](../src/guard_connection_ai/losses/reconstruction.py)
  - L1
  - SSIM
  - 2D FFT magnitude frequency-domain loss
- [image_metrics.py](../src/guard_connection_ai/metrics/image_metrics.py)
  - spectrogram MAE / RMSE / PRD / correlation

### Scripts

- [segment_bidmc.py](../scripts/segment_bidmc.py): segment index 生成
- [generate_stft_examples.py](../scripts/generate_stft_examples.py): 固定 subject/segment の STFT 例生成
- [plot_stft_comparison.py](../scripts/plot_stft_comparison.py): Raw / z-score STFT 比較
- [evaluate_stft_reconstruction.py](../scripts/evaluate_stft_reconstruction.py): complex STFT inverse 評価
- [train.py](../scripts/train.py): baseline 学習、MLflow、checkpoint、early stopping
- [evaluate_model.py](../scripts/evaluate_model.py): checkpoint の validation 評価
- [compare_experiments.py](../scripts/compare_experiments.py): 実験比較 CSV 作成
- [inspect_json_data.py](../scripts/inspect_json_data.py): json_Data metadata manifest 生成

## 5. STFT 基準設定

主設定は [configs/data.yaml](../configs/data.yaml) です。

```text
sampling_rate: 125
window: hann
nperseg: 64
noverlap: 32
nfft: 64
boundary: zeros
padded: true
representation: magnitude
```

各 10 秒 segment の STFT magnitude shape は `(33, 41)` です。

complex STFT -> inverse STFT の再構成は、BIDMC の 53 subject でほぼ完全に再構成できています。ただし、これは complex STFT の結果であり、magnitude 単独の inverse 可能性を意味しません。

## 6. 学習の実行方法

環境は Python 3.11 系、依存管理は `uv` です。

```powershell
uv sync
uv run pytest
uv run ruff check src/guard_connection_ai/data src/guard_connection_ai/losses src/guard_connection_ai/metrics src/guard_connection_ai/models src/guard_connection_ai/utils tests scripts/train.py scripts/evaluate_model.py
```

短時間 smoke test:

```powershell
uv run python scripts/train.py --epochs 1 --max-batches 1
```

L1 baseline:

```powershell
uv run python scripts/train.py --epochs 10 --patience 3
uv run python scripts/evaluate_model.py --checkpoint checkpoints/resunet_attention_l1_best.pt --output outputs/evaluation/l1_best_subject_metrics.csv
```

L1 + SSIM:

```powershell
uv run python scripts/train.py --epochs 10 --ssim-weight 0.1 --patience 3
uv run python scripts/evaluate_model.py --checkpoint checkpoints/resunet_attention_l1_ssim_best.pt --output outputs/evaluation/l1_ssim_best_subject_metrics.csv
```

L1 + SSIM + Frequency:

```powershell
uv run python scripts/train.py --epochs 10 --frequency-weight 0.001 --patience 3
uv run python scripts/evaluate_model.py --checkpoint checkpoints/resunet_attention_l1_ssim_frequency_best.pt --output outputs/evaluation/l1_ssim_frequency_best_subject_metrics.csv
```

注意: `train.py` の出力先は loss 条件ごとに分離されています。checkpoint は `last` と `best` の両方が生成されます。

MLflow は SQLite backend を使用します。

```text
sqlite:///mlflow.db
```

## 7. 既存実験結果

同一 seed、全 segments、最大 10 epochs、patience 3 で実行した結果です。3 条件とも epoch 6 で early stopping し、best epoch は 3 でした。

| 実験 | Best epoch | Val loss | Val MAE | Val RMSE | Val PRD | Val correlation |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 3 | 0.017370 | 0.017370 | 0.052236 | 61.485632% | 0.775123 |
| L1 + SSIM | 3 | 0.034339 | 0.018231 | 0.052671 | 61.997831% | 0.766650 |
| L1 + SSIM + Frequency | 3 | 0.018724 | 0.017927 | 0.053166 | 62.580660% | 0.775788 |

Subject 平均の correlation:

- L1: `0.798783`
- L1 + SSIM: `0.791846`
- L1 + SSIM + Frequency: `0.799264`

現時点では、3 条件のうち L1 が validation loss、MAE、RMSE で最も安定した基準です。Frequency は相関だけを見ると L1 と同程度ですが、総合的な改善は確認できていません。

これらは研究用の初期 baseline であり、学会提出用の最終性能値ではありません。

## 8. json_Data の扱い

追加されたデータは以下です。

```text
data/json_Data/
```

確認結果:

- JSON files: 94
- ECG metadata files: 8
- PPG metadata files: 86
- AF annotation metadata files: 8

ファイル名の形式は概ね以下です。

```text
<subject>_<ECG|PPG>_<recording>.json
```

重要な注意点:

- 現在の JSON は信号配列本体ではなく metadata が中心
- 大きな配列には `Large dataset omitted from full JSON export` の注記がある
- ECG、PPG、AF annotation の shape と dtype は確認できる
- 現時点では前段 PPG -> ECG 学習 Dataset に混ぜていない

manifest は以下です。

```text
outputs/json_data/metadata_manifest.csv
```

信号本体が提供されたら、前段学習データとは別の評価系として、AF annotation と同期条件を検証してください。BIDMC の subject split に json_Data の subject を追加してはいけません。

## 9. 既知の注意点

1. `data.yaml` と `resunet.yaml` に STFT 設定が重複しています。変更時は両方を同期するか、設定ローダーを一本化してください。
2. `train.py` の subject 数 `43`、`10` は現在の 53 subject 前提で一部ハードコードされています。将来データ数が変わる場合は修正が必要です。
3. 現在の指標は主に magnitude spectrogram に対するものです。waveform 再構成後の指標とは区別してください。
4. `BIDMCSTFTDataset` は毎回 signal を読み、STFT を計算します。長時間学習では I/O と CPU コストがボトルネックになる可能性があります。
5. `scripts/inspect_bidmc.py`、`inspect_signals.py`、`inspect_stft.py` には既存の重複読み込みや Ruff 指摘が残っています。今回の主パイプラインとは分けて扱ってください。
6. MLflow DB、checkpoint、outputs は研究成果物として重要ですが、大容量データや一時生成物を Git に追加しないでください。

## 10. 次に実装する優先順位

### 優先 1: waveform 再構成評価

現状の model output は ECG magnitude spectrogram です。予測 magnitude だけでは位相がないため、そのまま正確な waveform inverse はできません。

次のどちらかを明確に設計してください。

- ECG target の complex STFT の phase を評価専用に使う oracle-phase reconstruction
- phase を別途推定する拡張

まずは oracle-phase を「上限評価」として実装し、以下を記録するのが現実的です。

- waveform MAE
- waveform RMSE
- waveform PRD
- correlation

magnitude 単独から waveform を復元したかのような結論は出さないでください。

### 優先 2: json_Data の実配列確認

metadata JSON 以外に `.mat`、`.h5`、`.npy`、`.npz` などの本体ファイルが存在するか確認してください。存在しない場合は、JSON を無理に loader 化せず、追加提供を依頼するか metadata 検証に留めます。

### 優先 3: 実験比較の強化

- 乱数 seed を複数試す
- validation subject 単位の中央値・四分位範囲を報告
- best checkpoint の選定基準を明記
- MLflow の run ID と設定を比較表へ追加
- 3 epoch / 5 epoch / 10 epoch の結果を混ぜない

### 優先 4: 学習効率改善

- STFT cache または事前生成 dataset
- DataLoader worker 数の設定化
- mixed precision の GPU 検証
- checkpoint resume

### 優先 5: 本格モデル比較

L1 を基準に、十分な epoch と複数 seed で比較してください。GAN や Diffusion は、非敵対的 baseline の性能と限界を確認した後にのみ検討します。

## 11. 再開時の最初のコマンド

```powershell
Set-Location c:\Users\aruka\guard-connection-ai
uv run pytest
uv run python scripts/inspect_json_data.py
```

その後、変更前に対象ファイルを読み、focused test を追加してから編集してください。

## 12. 引継ぎ時の基本ルール

- Subject 単位 split を維持する
- segment 単位で Train/Validation をランダム分割しない
- PPG/ECG は同じ境界で pair にする
- Raw、前処理、STFT representation を混同しない
- magnitude と complex representation を混同しない
- `json_Data` を本体配列があると仮定しない
- 既存テストを通してから次の実験へ進む
- Notebook を正本にせず、リポジトリの source/config/test を正本とする
- 既存の未関係変更を戻さない
