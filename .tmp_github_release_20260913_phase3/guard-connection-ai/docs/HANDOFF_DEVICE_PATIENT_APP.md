# 計測デバイス・患者アプリ担当への引渡し

## 渡すもの

引渡しフォルダは`artifacts/team_handoff/device_patient_app_v1`である。

- `README.md`: 実装規則
- `device_ppg_packet.schema.json`: packet JSON Schema
- `session_create.example.json`: session作成例
- `device_ppg_good.jsonl`: 135 Hz、4chの正常連続packet例
- `device_ppg_missing_saturation_disconnect.jsonl`: 欠損、飽和、切断例
- `manifest.json`: データ出典と制限

sampleは実機由来ではなく、通信実装確認用の合成値である。実機値の範囲、極性、clock drift、
BLE欠落率を示すものではない。

## 患者アプリ側へお願いしたい処理

1. `ppg0/ppg1/ppg2/ambient0`を同じ長さで送る。
2. ADC整数値を保ち、平均化、ambient減算、filter、正規化、resamplingを送信前に行わない。
3. device timestampを保持し、packet末尾sampleの時刻を`device_timestamp_end_ns`へ入れる。
4. 同一streamでは`sequence_number`を0から1ずつ増やす。
5. 欠損は`null`、飽和は`saturation`、切断は`sensor_disconnected`で表す。
6. 再接続では旧sessionを終了し、新しいsessionとstream IDを作る。

device SQIとmotion levelは波形を加工せずmetadataとして送る。サーバー側でもSQIを計算するため、
両者の値は区別して保存する。

## 現時点で未確定のもの

- Verity Sense実機ADCの最小・最大、符号、飽和値
- 実測sampling rateの分布
- device clockとphone UTCのanchor更新間隔
- 加速度・ジャイロpacketとPPGの厳密な対応方法

これらは実機sample受領後に確定する。現段階では通信フィールドとエラー処理の実装を進められる。

