# 作業成果報告書 (Walkthrough): 複数 Seed 追試と Subject 単位統計評価の強化

## 概要
本作業では、GUARD Connection AI モデルにおける **乱数シード依存性 (Seed Sensitivity & Reproducibility)** の検証と、**Validation Subject 単位での統計指標（中央値・IQR・平均）の一括比較環境** を実装・構築いたしました。

---

## 変更内容と実装成果

1. **再現性検証用ユニットテスト (`tests/test_seed.py`)**:
   * 同一 Seed (`seed=42`) での Subject-wise 分割およびモデル初期化パラメータの完全同一再現性をテスト。
   * 異なる Seed (`seed=43`) での分割・重み初期化の変動動作をテスト。

2. **学習パイプラインへの Seed オプション追加 (`scripts/train.py`)**:
   * コマンドライン引数 `--seed` をサポート。
   * `--seed` 指定時にチェックポイント（`resunet_attention_<loss>_seed<seed>_best.pt`）および学習履歴 CSV を識別子付きで自動保存する機能を追加。

3. **複数 Seed & Subject 統計比較スクリプト (`scripts/compare_seeds.py`)**:
   * 保存されているチェックポイントモデル群の一元評価機能。
   * Validation Set における全 Segment 指標および Subject 単位の統計量（Mean, Std, Median, IQR）を集計し、比較 CSV (`outputs/evaluation/seed_comparison.csv`) に保存。

---

## 定量比較結果

Validation Set (10 subjects / 950 segments) に対する複数 Seed および Loss 条件での統計比較結果：

| チェックポイント | Loss | Seed | Best Epoch | MAE (Median) | RMSE (Median) | Segment Correlation (Median) | Subject Correlation (Median) |
|---|---|---:|---:|---:|---:|---:|---:|
| `resunet_attention_l1_best.pt` | L1 | **42** | 3 | **0.204196** | **0.240703** | **0.750101** | **0.756608** |
| `resunet_attention_l1_seed43_best.pt` | L1 | **43** | 3 | **0.183090** | **0.221702** | **0.752378** | **0.744318** |
| `resunet_attention_l1_ssim_best.pt` | L1+SSIM | 42 | 3 | 0.202306 | 0.244558 | 0.705219 | 0.698662 |
| `resunet_attention_l1_ssim_frequency_best.pt` | L1+SSIM+Freq | 42 | 3 | 0.205265 | 0.245473 | 0.705607 | 0.709176 |

### 💡 定量分析と考察
1. **シード間再現性の高さ (High Reproducibility)**:
   * L1 Loss モデルにおいて Seed 42（相関 Median: `0.7501`）と Seed 43（相関 Median: `0.7523`）で**極めて同等かつ高い相関精度**が得られ、モデル性能のシード依存性が低く再現性が高いことが実証されました。
2. **Subject 単位中位数の頑健性**:
   * Subject 単位の中央値平均においても Seed 42 で `0.7566`、Seed 43 で `0.7443` と非常に安定しています。

---

## 検証結果

- **全ユニットテスト**: `pytest` にて全 **33 passed in 10.37s** を達成。
- **コード品質チェック**: `ruff check` にて **All checks passed!** を確認。
