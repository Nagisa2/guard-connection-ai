# チーム接続用リアルタイム通信契約

この文書は、患者側アプリ、前段AI、後段AI、医師画面を接続する際の固定境界を示す。疑似ECGはPPG由来の補助表示であり、実測ECG・診断用ECG・確定診断ではない。

## 接続全体

```text
村川さん側アプリ
  -> POST /realtime/sessions
  -> POST /realtime/sessions/{session_id}/device-ppg-chunks
       |-> WebSocket /ws/front-ai/downstream/{session_id} -> 倉本さん側AI
       |-> GET /front-ai/windows/{session_id}/next         -> 倉本さん側AI
       |<- POST /downstream-inferences                  <- 倉本さん側AI
       `-> WebSocket /ws/sensors/{user_id}              -> 医師画面
```

## 村川さん側の受入境界

1. 計測開始時に `POST /realtime/sessions` を1回呼ぶ。
2. 応答の `session_id` を計測中保持する。
3. 約250 msごとに `POST /realtime/sessions/{session_id}/device-ppg-chunks` を呼ぶ。
4. 計測終了時に `DELETE /realtime/sessions/{session_id}` を呼ぶ。

端末パケットは `schema_version=1.0` とし、`ppg0`、`ppg1`、`ppg2`、`ambient0`を未加工のADC整数列として送る。4列の長さは同一にする。送信経路では平均化、ambient減算、フィルタ、正規化、リサンプリングを行わない。

必須項目は、`stream_id`、0始まりの`sequence_number`、`configured_sample_rate_hz`、`device_timestamp_end_ns`、PPG 4チャンネルである。欠損値は`null`、切断は`sensor_disconnected`、飽和が判定できたチャンネルだけ`saturation`へ同じ長さの真偽値列を入れる。

JSONはそのまま送る方法に加え、`Content-Encoding: gzip`で圧縮できる。圧縮済み本文は2 MB、展開後本文は16 MBを上限とする。不正なgzipは構造化された400応答、上限超過は413応答になる。

連番の欠落・重複・順序逆転、timestamp不連続は409応答になる。応答の`error.expected_sequence_number`に従い、欠けたパケットを再送するかセッションを終了して作り直す。

受け渡し前のJSONは次のコマンドで、サーバーを起動せず検証できる。

```powershell
cd <App-monitor-v2の配置先>\backend
.\.venv\Scripts\python.exe validate_device_payload.py <sample.json> <sample.json.gz>
```

## 加速度・ジャイロ

慣性センサーはPPGと同じ配列へ混在させず、次へ送る。

```text
POST /realtime/sessions/{session_id}/device-motion-chunks
```

`sensor_type`は`accelerometer`または`gyroscope`、単位はそれぞれ`mG`、`deg/s`とする。`x/y/z`は同じ長さにする。Verity SenseとH10は独立クロックなので、同じ`clock_domain`を付けない。各`stream_id`で連番、サンプリング周波数、clock domain、timestamp連続性を独立に検証する。

## 倉本さん側への前段AI出力

前段AI出力は次のWebSocketから受け取る。

```text
ws://<server>/ws/front-ai/downstream/{session_id}
```

HTTPヘッダー`x-downstream-api-key`が必須である。サーバー側は同じ値を環境変数`GUARD_DOWNSTREAM_API_KEY`に設定する。APIキーはGit、添付サンプル、ログへ保存しない。

各メッセージは`payload_type=front_ai_device_handoff`で、次を含む。

- `source_device_packet`: 元の4チャンネルPPGと端末メタデータ
- `derived.ppg`: 前段処理後も保持されるPPG
- `derived.ppg_valid`: サンプル単位valid mask
- `derived.beat_events`: 推定拍イベント時刻列
- `derived.technical_sqi`と`signal_coverage`
- `derived.pulse_interval_plausibility`
- `generation_accepted`と`generation_abstention_reason`
- 医師表示用疑似ECG。ただしAF判定の必須入力にはしない

WebSocketを組み込むまでの暫定方法として、村川さん側POST応答の`downstream_ai`にも同一payloadを返す。

受信のみを確認する最小クライアントは次のように実行する。

```powershell
cd <App-monitor-v2の配置先>\backend
$env:GUARD_DOWNSTREAM_API_KEY = "<別経路で共有したキー>"
.\.venv\Scripts\python.exe downstream_ai_client_example.py <session_id>
```

このクライアントは入力検証と要約表示だけを行い、AF推論結果は生成しない。

### 未ラベル30秒窓（接続推奨境界）

前段AI側で30秒窓・5秒ストライドへ集約した入力を、次のHTTP pullでも取得できる。

```text
GET  /front-ai/windows/{session_id}/next
POST /front-ai/windows/{session_id}/ack
schema_version = front_ai_window_v1
```

どちらも`x-downstream-api-key`を必須とする。`GET`は未ACKの最古窓を返し、待機窓がなければ204を返す。後段処理または結果POSTに失敗した場合はACKせず、再接続後に同じ`input_sequence_id`を再取得する。結果を`POST /downstream-inferences`へ正常送信した後にのみACKする。同じ推論連番・同一payloadの結果POSTと、同じ窓ACKの再送は冪等である。

窓の主要フィールドは次のとおり。

- `primary_signal=ppg`と`ppg.values`、`ppg.valid_mask`
- `beats.times_sec`、`beats.amplitudes`、`source=ppg`
- `quality.technical_sqi`、`signal_coverage`、`pulse_interval_plausibility`
- 同一`clock_domain`で時刻が重なる`motion_streams`
- 時計が異なり同期できない`unmapped_motion_streams`
- 補助入力`auxiliary_pseudo_ecg`（常に`diagnostic_ecg=false`）

この契約はライブ推論専用であり、教師ラベルを含まない。後段リポジトリの`WindowSample`は`label_source=rhythm_annotation`を強制する学習・評価用型なので、ライブ窓には使用しない。`ppg`は後段の`SignalChannel`、`beats`は`BeatSeries(source="ppg")`へ変換できる。拍時刻は絶対秒の64 bit浮動小数として保持し、RRは後段で拍時刻差分から算出する。

取得確認用クライアントは次のように実行する。標準ではACKしない。

```powershell
cd <App-monitor-v2の配置先>\backend
$env:GUARD_DOWNSTREAM_API_KEY = "<別経路で共有したキー>"
.\.venv\Scripts\python.exe downstream_window_client_example.py <session_id>
```

`--ack`は、実際の後段処理と結果POSTが成功した場合だけ使用する。WebSocketは250ms単位の観測や低遅延表示に向く一方、このHTTP pull境界は5秒ごとのモデル推論、切断後の再取得、ACK管理に向く。後段AIの本接続には後者を推奨する。

慣性センサーは、窓生成より先に到着し、PPGと同一`clock_domain`である区間のみ窓へ同梱する。別デバイス時計の信号や遅着パケットを無理に同期しない。必要なら生の補助ストリームをWebSocketでも併用する。

## 倉本さん側からの結果入力

後段AIは30秒窓等の処理が完了するたびに`POST /downstream-inferences`へ`schema_version=downstream_v1`を送る。

判定状態は次の3種類に限定する。

- `af_suspected`
- `no_af_suspected`
- `undecidable`

`undecidable`では`af_probability=null`とし、`abstention_reason`を必須にする。`generation_accepted=false`をAF陰性へ変換しない。`inference_mode=model`と`demo_stub`は分離し、実機セッションへ`demo_stub`を関連付けることはできない。

## 医師画面

医師画面は`WebSocket /ws/sensors/{user_id}`を購読する。実測・シミュレーション・PPG由来疑似ECGを`waveform_type`と`diagnostic_ecg`で区別する。疑似ECGではP波、PQ/PR、QRS幅、QT/QTc、STの測定UIを有効にしない。

## 接続確認

操作パネルの「システム稼働状況」で次を確認できる。

- 患者DB件数
- 稼働中セッション数と最終入力時刻
- 医師画面WebSocket接続数
- 後段AIの`model` / `demo_stub`モード
- 後段AI未接続時の`unconnected`、未ACK窓数
- 村川さん側入力、前段AI出力、後段AI結果入力の利用可否

サーバーからは`GET /demo-control/system-status`でも同じ情報を取得できる。波形サンプルやAPIキーは状態応答に含めない。
