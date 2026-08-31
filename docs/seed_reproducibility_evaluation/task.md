# タスク詳細: 複数 Seed での追試および Subject 単位の詳細統計評価（中央値・IQR）の強化

## 概要
本タスクは、GUARD Connection AI モデルの学習および評価における **乱数シード依存性の検証 (Reproducibility & Seed Sensitivity)** と、**Validation Subject 単位での詳細統計指標（Mean ± Std, Median [IQR]）による評価体系の強化** を目的とします。

## 目的・背景
- 現在の基盤モデル学習 (`scripts/train.py`) は固定シード (`seed=42`) で実行されていますが、学術的・臨床的な信頼性を高めるために複数の異なる乱数シード（例: 42, 43, 44 など）で学習および評価を行い、結果の再現性とバラつきを定量評価する必要があります。
- また、全セグメント平均だけでなく、Subject 単位での中央値 (Median) および四分位範囲 (IQR) の分布を報告・集計する比較スクリプトを構築します。

## 要件
1. **学習スクリプトの拡張 (`scripts/train.py`)**:
   * `--seed` コマンドライン引数を追加し、データ分割およびモデル初期化・各種処理のシード値を指定可能にする。
   * シード値ごとにチェックポイント (`resunet_attention_<loss>_seed<seed>_best.pt` 等) および学習履歴 CSV を分離・保存可能にする。
2. **複数 Seed & Subject 統計評価スクリプトの実装 (`scripts/compare_seeds.py`)**:
   * 指定した複数シードのチェックポイントを一括評価。
   * 各シードにおける Validation Subject 単位の指標（MAE, RMSE, PRD, Correlation）の中央値および IQR を集計し、複数シード間での平均・標準偏差を算出して比較 CSV (`outputs/evaluation/seed_comparison.csv`) を出力。
3. **ユニットテストの追加・更新 (`tests/test_seed.py`)**:
   * シード変更時にデータ分割およびモデル初期化が正しく再現/変動することを検証するテストを作成。
4. **既存テストおよびコード品質検査のパス**:
   * pytest 全テストクリア
   * ruff check パス

## 成果物
- `docs/seed_reproducibility_evaluation/task.md`
- `docs/seed_reproducibility_evaluation/implementation_plan.md`
- `docs/seed_reproducibility_evaluation/walkthrough.md`
- `scripts/train.py` (機能拡張)
- `scripts/compare_seeds.py` (新規スクリプト)
- `tests/test_seed.py` (新規テスト)
- `outputs/evaluation/seed_comparison.csv` (比較集計結果)
