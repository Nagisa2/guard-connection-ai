import test from "node:test";
import assert from "node:assert/strict";

import {
  createReceiveState,
  formatWaveformTooltipValue,
  receiveWaveformEnvelope,
  waveformDisplayName,
} from "./realtimeProtocol.js";

function envelope(overrides = {}) {
  const frame = {
    schema_version: "1.1",
    payload_type: "doctor_waveform_visualization",
    session_id: "session-a",
    sequence_number: 0,
    sample_rate_hz: 2,
    input_timestamp_start_seconds: 100,
    display_timestamp_start_seconds: 99.5,
    ppg: [0.1, 0.2],
    ppg_valid: [true, true],
    pseudo_ecg: [0.5, 0.7],
    pseudo_ecg_valid: [true, true],
    waveform_type: "ppg_derived_pseudo_ecg",
    diagnostic_ecg: false,
    generation_accepted: true,
    measurement_ui_enabled: false,
    generation_mode: "population_prior",
    morphology_source: "population_mean_latent",
    timing_confidence: 0.9,
    morphology_confidence: 0.0,
    repolarization_duration_prior_ms: 380,
    p_wave_generated: false,
    qt_measurement_supported: false,
    pq_pr_measurement_supported: false,
    ...overrides,
  };
  return {
    message_type: "waveform_frame",
    schema_version: "1.0",
    source_provenance: { source_mode: "live_device", demo_scenario_id: null },
    frame,
  };
}

test("PPGと疑似ECGは別のtimestampを保持する", () => {
  const result = receiveWaveformEnvelope(createReceiveState(), envelope());
  assert.equal(result.graph.ppgData[0].timestampSeconds, 100);
  assert.equal(result.graph.pseudoEcgData[0].timestampSeconds, 99.5);
  assert.equal(result.graph.ppgData[1].timestampSeconds, 100.5);
});

test("無効区間と棄却区間はnullとなり補間対象にならない", () => {
  const invalid = receiveWaveformEnvelope(
    createReceiveState(),
    envelope({ ppg_valid: [true, false], pseudo_ecg_valid: [true, false] }),
  );
  assert.equal(invalid.graph.ppgData[1].value, null);
  assert.equal(invalid.graph.pseudoEcgData[1].value, null);
  const rejected = receiveWaveformEnvelope(
    createReceiveState(),
    envelope({ generation_accepted: false }),
  );
  assert.deepEqual(rejected.graph.pseudoEcgData.map((point) => point.value), [null, null]);
});

test("重複・欠落・順序逆転を検出する", () => {
  const first = receiveWaveformEnvelope(createReceiveState(), envelope());
  assert.equal(receiveWaveformEnvelope(first.state, envelope()).issue, "duplicate_packet");
  assert.equal(receiveWaveformEnvelope(first.state, envelope({ sequence_number: 3 })).issue, "sequence_gap");
  const advanced = { sessionId: "session-a", expectedSequenceNumber: 3 };
  assert.equal(receiveWaveformEnvelope(advanced, envelope({ sequence_number: 0 })).issue, "out_of_order");
});

test("新しいsession_idでは受信stateをリセットする", () => {
  const old = { sessionId: "old", expectedSequenceNumber: 20 };
  const result = receiveWaveformEnvelope(old, envelope({ session_id: "new", sequence_number: 0 }));
  assert.equal(result.accepted, true);
  assert.equal(result.state.sessionId, "new");
  assert.equal(result.state.expectedSequenceNumber, 1);
});

test("実測・疑似・シミュレーションECGを表示名で区別する", () => {
  assert.equal(waveformDisplayName("measured_ecg", true), "実測ECG");
  assert.match(waveformDisplayName("ppg_derived_pseudo_ecg", false), /診断用ではない/);
  assert.match(waveformDisplayName("simulated_ecg", false), /シミュレーション/);
});

test("Tooltipの波形値は指定した小数桁へ丸める", () => {
  assert.equal(formatWaveformTooltipValue(0.123456789, 3), "0.123");
  assert.equal(formatWaveformTooltipValue(72.3456, 1), "72.3");
});

test("P波またはQT/PQ測定を有効にした疑似ECGを拒否する", () => {
  assert.throws(() => receiveWaveformEnvelope(createReceiveState(), envelope({ p_wave_generated: true })), /P波/);
  assert.throws(() => receiveWaveformEnvelope(createReceiveState(), envelope({ qt_measurement_supported: true })), /QT/);
  assert.throws(() => receiveWaveformEnvelope(createReceiveState(), envelope({ pq_pr_measurement_supported: true })), /PQ\/PR/);
});

test("通信1.0と波形1.1を独立に検証する", () => {
  assert.throws(
    () => receiveWaveformEnvelope(createReceiveState(), { ...envelope(), schema_version: "1.1" }),
    /通信schema_version/,
  );
  assert.throws(
    () => receiveWaveformEnvelope(createReceiveState(), envelope({ schema_version: "1.0" })),
    /frame schema/,
  );
});

test("実機入力と記録再生の来歴を区別する", () => {
  const live = receiveWaveformEnvelope(createReceiveState(), envelope());
  assert.equal(live.graph.sourceProvenance.source_mode, "live_device");

  const replayEnvelope = {
    ...envelope(),
    source_provenance: {
      source_mode: "recorded_demo_replay",
      demo_scenario_id: "af-replay-01",
    },
  };
  const replay = receiveWaveformEnvelope(createReceiveState(), replayEnvelope);
  assert.equal(replay.graph.sourceProvenance.demo_scenario_id, "af-replay-01");

  assert.throws(
    () => receiveWaveformEnvelope(
      createReceiveState(),
      { ...replayEnvelope, source_provenance: { source_mode: "recorded_demo_replay" } },
    ),
    /demo_scenario_id/,
  );
});

test("合成デモ入力を実機や記録再生と区別する", () => {
  const syntheticEnvelope = {
    ...envelope(),
    source_provenance: {
      source_mode: "synthetic_demo",
      demo_scenario_id: "synthetic-normal-v1",
    },
  };
  const result = receiveWaveformEnvelope(createReceiveState(), syntheticEnvelope);
  assert.equal(result.graph.sourceProvenance.source_mode, "synthetic_demo");
});
