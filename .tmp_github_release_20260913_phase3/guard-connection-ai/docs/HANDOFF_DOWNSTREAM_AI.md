# 後段AI担当への前段AI出力引渡し

## 渡すもの

引渡しフォルダは`artifacts/team_handoff/downstream_ai_v1_1`である。

- `README.md`: 読込・利用規則
- `downstream_ai.schema.json`: frame JSON Schema
- `af_good.jsonl`: held-out AF記録の実PPG sample
- `non_af_good.jsonl`: held-out non-AF記録の実PPG sample
- `missing.jsonl`: 途中欠損例
- `invalid.jsonl`: 全点無効例
- `manifest.json`: ラベル、出典、checkpoint hash、制限

## 後段AIの主入力

AF判定の主経路には`ppg`、`ppg_valid`、timestamp、sample rateを使用する。元PPGを捨てず、
疑似ECGだけを入力としない。`ppg_valid=false`は補間せずmaskとして扱う。

疑似ECG、Timing event相当のconfidence、pulse intervalは補助特徴として使用できる。ただし疑似ECGは
`ppg_derived_pseudo_ecg`であり実測ECGではない。P波、PQ/PR、QT、STまたは患者固有QRS形態を
復元したものとして扱わない。

`generation_accepted=false`はAF陰性を意味しない。低SQI、欠損、ウォームアップなどにより前段AIが
棄却した状態であり、後段AI側でも「判定不能」と「陰性」を分ける必要がある。

## splitとsampleの制限

`af_good.jsonl`と`non_af_good.jsonl`はfold 0のheld-out test患者から抽出した。患者識別子と元pathは
含めていない。この2例はloaderとschema確認用であり、学習、閾値調整、性能評価には少なすぎる。

正式な学習では患者単位splitを維持し、frameまたはsegment単位のrandom splitを使用しない。
前段AIはAF probabilityや診断結果を出力しない。

## 拍イベントとinterval統計

`beat_events`には、そのframeで新たに出力されたeventだけを格納する。`event_sample_index`はsession内の
絶対sample indexである。`detected_at_input_timestamp_seconds`は因果処理がeventを確定した時刻、
`estimated_r_timestamp_seconds`はTiming Headの表示遅延を補正した推定時刻であり、実測ECG R peakではない。

`pulse_event_count_total`はsession内の単調増加event数、`interval_window_count`はrolling統計に現在使った
interval数である。旧`interval_count`は後方互換用のaliasで、総数として使用しない。`rmssd_ms`と
`interval_cv`は0.25〜2.0秒の範囲内だけを使った表示用参考値であり、後段AF特徴量には使用しない。

`technical_sqi`は現状ではvalid sample coverageに基づく技術品質、`pulse_interval_plausibility`は
0.25〜2.0秒に入った検出intervalの割合である。後者をAF陰性判定へ直接使用しない。

## 山口大学JSON exportの現状

`data/json_Data`は変換時に大容量配列を省略したJSON exportである。評価用の正本には使用しない。
元データは`data/Data/Data`にあり、MATLAB 7.3（HDF5）形式で8 subject、ECG 8件、PPG 86件、合計94件・
約7.15 GBである。全ファイルで必要な実配列を読み取れることを確認した。

- ECG: 500 sample/data record、QRSindex、RR、AF annotationを収録
- PPG: green/ambient各100 sample/data record、加速度50 sample/data record、record timestampを収録
- AF annotation値: 0、0.5、1。値1を含むsubjectは03、05、08

ただし、PPG record timestampには6,029件の重複と、1.5秒を超える9,307件のgapがある。時刻逆転はない。
`03_PPG_14.mat`と`07_PPG_01.mat`はECGの記録時刻範囲と重ならない。また、AF annotation値0.5の意味と、
PPG record timestampが各1秒recordの先頭時刻か末尾時刻かは、データ内の情報だけでは確定できない。

したがって8名全体のzero-shot前段AI出力は作成可能になったが、単純連結や一律補間は行わない。重複recordの
処理、gapのvalid mask化、ECGとの時刻対応、0.5 annotationの扱いを明示したloaderを先に実装し、モデルを
固定した患者単位の外部評価として扱う。元MATの監査は次で再現できる。

```powershell
.\.venv\Scripts\python.exe scripts\audit_yamaguchi_mat.py
```

旧JSON exportの状態は次で確認できる。

```powershell
.\.venv\Scripts\python.exe scripts\audit_yamaguchi_data.py
```

## 読込確認

```powershell
cd <workspace>\guard-connection-ai
.\.venv\Scripts\python.exe scripts\read_front_ai_handoff_sample.py `
  artifacts\team_handoff\downstream_ai_v1_1\af_good.jsonl
```
