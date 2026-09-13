# guard-connection-ai

PPGから拍タイミングと非診断用疑似ECGを生成する前段AIです。AF分類器は含みません。

## 現在の生成内容

- Timing Head: PPGからECG R-event相当の時刻を推定
- morphology prior: 一般的なQRS/S/T形状を付加して表示用波形を生成
- quality gate: SQI、coverage、欠損、confidenceに応じて生成を棄却
- realtime schema: PPG、疑似ECG、valid mask、入力時刻、表示時刻、sequenceを保持

R-eventは推定値で、実測ECGのR波ではありません。P波、PQ/PR、QT、ST、患者固有QRS形態の正確な復元は保証しません。

## 開発環境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip.exe install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider
.\.venv\Scripts\ruff.exe check src scripts tests
```

学習には別途MIMIC PERform AF等のデータ配置が必要です。データは本リポジトリに含みません。
