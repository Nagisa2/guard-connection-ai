# 実装計画書: 学習効率改善（STFT 事前計算キャッシュ・DataLoader Worker 可変・AMP / Resume 対応）の実装

GUARD Connection AI パイプラインにおける CPU / I/O ボトルネックの解消、PyTorch AMP による高速化、および Checkpoint Resume 機能を実装します。

## Proposed Changes

### Data / Dataset

#### [MODIFY] [dataset.py](file:///e:/guard-connection-ai/src/guard_connection_ai/data/dataset.py)
- `BIDMCSTFTDataset` に `cache_stft: bool = False` および `preload: bool = False` を追加。
- 信号読み込み・STFT 計算済みの Tensor / numpy 配列をメモリ内に辞書キャッシュ保持するロジックを実装し、エポックごとのファイル I/O および CPU 計算コストを削減します。

### Scripts

#### [MODIFY] [train.py](file:///e:/guard-connection-ai/scripts/train.py)
- `--num-workers` コマンドライン引数を追加し、DataLoader のマルチプロセス並列化に対応させます。
- `--use-amp` コマンドライン引数を追加し、`torch.cuda.amp.autocast()` および `GradScaler` による 16-bit 混合精度訓練をサポートします（GPU 利用時）。
- `--resume <checkpoint_path>` を追加し、保存済みチェックポイントの Epoch, Model state, Optimizer state をロードして学習を再開するロジックを実装します。

### Tests

#### [NEW] [test_training_efficiency.py](file:///e:/guard-connection-ai/tests/test_training_efficiency.py)
- `cache_stft=True` で初期化した Dataset が非キャッシュ時と全く同一の `input` / `target` Tensor を返すこと。
- `save_checkpoint` および Resume ロジックにより、ロード後の重みと Optimizer 状態が完全復元されること。

---

## Verification Plan

### Automated Tests
- pytest による全テストの実行:
  ```powershell
  $env:PYTHONPATH="e:\guard-connection-ai\src"; C:\Python\Python312\python.exe -m pytest
  ```
- Ruff による静的コード品質チェック:
  ```powershell
  C:\Python\Python312\python.exe -m ruff check src/ tests/ scripts/
  ```

### Manual Verification
- 短時間 smoke test で Resume 機能および AMP オプション、キャッシュ機能付き学習の正常終了を確認:
  ```powershell
  C:\Python\Python312\python.exe scripts/train.py --epochs 1 --max-batches 2 --resume checkpoints/resunet_attention_l1_best.pt
  ```
