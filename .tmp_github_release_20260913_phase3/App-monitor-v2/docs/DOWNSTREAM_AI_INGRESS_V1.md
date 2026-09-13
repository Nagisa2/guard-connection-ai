# 後段AI推論受信API downstream_v1

## 位置付け

後段AF検出AIが30秒窓ごとに出力する結果を、サーバーが順序検証して受け取るための契約である。
このAPIへの入力は確定診断ではない。窓単位の`af_suspected`だけでは通知せず、
`episode.state=confirmed`になった場合だけ既存の本人確認フローへ接続する。

## API

- 登録: `POST /downstream-inferences`
- 最新結果: `GET /downstream-inferences/{session_id}/latest`

登録前に`POST /realtime/sessions`で同じ`session_id`のリアルタイムセッションを作成する。
リアルタイムセッションを終了すると、そのセッションの推論連番も破棄される。
`window.start_seconds`と`window.end_seconds`は、前段AIが渡したPPG入力時刻を基準とする。

## 入力例

```json
{
  "schema_version": "downstream_v1",
  "session_id": "contest-device-demo-test_user_01",
  "inference_sequence_number": 0,
  "model_version": "af-v1",
  "inference_mode": "model",
  "frontend_schema_version": "1.1",
  "window": {
    "input_sequence_id": 120,
    "start_seconds": 100.0,
    "end_seconds": 130.0,
    "window_seconds": 30.0,
    "stride_seconds": 5.0
  },
  "decidable": true,
  "abstention_reason": null,
  "af_probability": 0.91,
  "probability_is_calibrated": false,
  "decision": "af_suspected",
  "episode": {
    "state": "candidate",
    "episode_id": "episode-1",
    "start_seconds": 100.0,
    "duration_seconds": 30.0
  },
  "context": {
    "valid_ratio": 0.94,
    "n_valid_beats": 36,
    "sqi_window": 0.88,
    "gate_value": 0.76,
    "used_morphology": false,
    "frontend_model_version": "front-stage4-v1"
  },
  "inference_timestamp_utc": "2026-09-13T00:00:00Z"
}
```

## 3状態の整合規則

### 判定可能

- `decidable=true`
- `decision`は`af_suspected`または`no_af_suspected`
- `af_probability`は0以上1以下の数値
- `abstention_reason=null`

### 判定不能

- `decidable=false`
- `decision=undecidable`
- `af_probability=null`
- `abstention_reason`は空でない文字列

判定不能時に直前のAF値を保持して再送してはならない。
`probability_is_calibrated=false`の場合、画面ではAF確率ではなくモデルスコアとして扱う。
`inference_mode=demo_stub`は記録再生または合成デモ専用であり、実機入力には使用できない。
デモstubから作成されたcare eventにもこの区別を保持する。

## 監視状態の集計

応答の`monitoring_summary`には次を含む。

- `analysis_state`: `monitoring` / `af_suspected` / `undecidable`
- `valid_ratio`: 受信した解析時間のうち判定可能だった割合
- `consecutive_undecidable_seconds`: 現在連続している判定不能時間
- `undecidable_seconds_by_reason`: 理由ごとの累積判定不能時間
- `af_suspected_ratio_over_valid_decisions`: 有効判定時間中のAF疑い割合

最初の結果では`window_seconds`、以降は重複分を除いた`stride_seconds`を時間寄与として集計する。
連続する窓の終了時刻差が`stride_seconds`と一致しない場合は`window_gap`として棄却する。
`af_suspected_ratio_over_valid_decisions`は窓判定の集計であり、臨床的に検証されたAF burdenではない。

## 連番

`inference_sequence_number`はセッションごとに0から単調増加させる。

- 同一内容の直前結果を再送: 成功、`changed=false`
- 同じ連番で内容が異なる: `duplicate_inference`
- 期待値より大きい: `sequence_gap`
- 直前より古い: `out_of_order`

エラー応答の`expected_sequence_number`を参照して送信側の状態を確認する。

## 現在の制限

- 保持先はインメモリであり、サーバー再起動後には復元されない。
- `episode.state`の語彙は後段AI側の実装確定前のため、空でない文字列として保持する。
- `episode.state=confirmed`だけをcare eventへ冪等に変換する。
- 医師画面への3状態・valid ratio・判定不能理由の表示は次フェーズで実装する。
