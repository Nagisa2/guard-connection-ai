# タスク詳細: 学習効率改善（STFT 事前計算キャッシュ・DataLoader Worker 可変・AMP / Resume 対応）の実装

## 概要
本タスクは、GUARD Connection AI モデルの学習パイプライン (`scripts/train.py` および `src/guard_connection_ai/data/dataset.py`) における **I/O ボトルネックの解消**、**学習実行の高速化**、**GPU メモリ/計算リソースの効率化 (Automatic Mixed Precision: AMP)**、および **学習の中断・再開機能 (Checkpoint Resume)** を実装・整備する作業です。

## 目的・背景
- 現在の `BIDMCSTFTDataset` は `__getitem__` のたびに生波形 CSV をディスクからロードし、即座に STFT を計算するため、DataLoader のイテレーションごとに CPU および ディスク I/O のボトルネックが発生していました。
- 将来的な長時間の学習や大量データ処理において高速な学習ループを実現するため、メモリ上での STFT キャッシュまたは事前計算 Dataset オプションを提供します。
- さらに、DataLoader の `num_workers` 設定の可変性、`PyTorch AMP (Automatic Mixed Precision)` による高速化、および失敗時/追加学習時の `checkpoint resume` 機能を実装します。

## 要件
1. **STFT インメモリキャッシュ機能の追加 (`src/guard_connection_ai/data/dataset.py`)**:
   * `cache_stft: bool = False` オプションを `BIDMCSTFTDataset` に追加し、初回収集時または初期化時に計算結果をキャッシュ可能にする。
2. **DataLoader 並列化と AMP / Resume のサポート (`scripts/train.py`)**:
   * `--num-workers` コマンドライン引数を追加（デフォルトは環境に応じた可変設定）。
   * `--amp` (PyTorch Automatic Mixed Precision `torch.cuda.amp` / `torch.amp`) サポートを追加。
   * `--resume <checkpoint_path>` コマンドライン引数を追加し、エポック途中のチェックポイントからの学習再開機能を実装。
3. **ユニットテストの作成・追加 (`tests/test_training_efficiency.py`)**:
   * キャッシュ化前後の dataset 出力一致テスト。
   * checkpoint resume 動作テスト。
4. **既存テストおよびコード品質チェックのクリア**:
   * pytest 全テストパス
   * ruff check パス

## 成果物
- `docs/training_efficiency_optimization/task.md`
- `docs/training_efficiency_optimization/implementation_plan.md`
- `docs/training_efficiency_optimization/walkthrough.md`
- `src/guard_connection_ai/data/dataset.py` (キャッシュ拡張)
- `scripts/train.py` (num_workers, AMP, resume 拡張)
- `tests/test_training_efficiency.py` (新規テスト)
