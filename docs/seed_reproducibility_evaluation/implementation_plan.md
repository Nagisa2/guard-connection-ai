# 実装計画書: 複数 Seed での追試および Subject 単位の詳細統計評価（中央値・IQR）の強化

GUARD Connection AI のモデル学習・評価パイプラインに対し、複数 Seed（再現性検証）のサポートと、Validation Subject 単位での統計指標（Mean, Std, Median, IQR）の一括比較機能を実装します。

## Proposed Changes

### Scripts

#### [MODIFY] [train.py](file:///e:/guard-connection-ai/scripts/train.py)
- `--seed` オプションを追加し、データ分割およびモデル・Optmizer・Torch 乱数シードの設定に対応させます。
- `--seed` がデフォルト以外の値で渡された場合、チェックポイント名に `_seed<seed>` を自動付与して分離保存できるようにします。

#### [NEW] [compare_seeds.py](file:///e:/guard-connection-ai/scripts/compare_seeds.py)
- 複数のシード（例: `42`, `43`, `44` など）で保存されたモデルチェックポイントを一括読み込み。
- Validation Set の Subject 単位統計量（中央値・IQR・平均）を集計し、シード間の安定性（Mean ± Std）をまとめた比較報告 CSV (`outputs/evaluation/seed_comparison.csv`) を生成します。

### Tests

#### [NEW] [test_seed.py](file:///e:/guard-connection-ai/tests/test_seed.py)
- シード固定時にデータ分割結果およびモデル初期化パラメータが完全に同一再現されること、異なるシードでは適切に分割・初期化が切り替わることをテストします。

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
- 短時間 smoke test で複数 seed の学習・チェックポイント保存を確認:
  ```powershell
  C:\Python\Python312\python.exe scripts/train.py --epochs 1 --max-batches 2 --seed 43
  ```
- 比較スクリプト `scripts/compare_seeds.py` を実行し、`outputs/evaluation/seed_comparison.csv` が期待通りの結果を出力することを検証します。
