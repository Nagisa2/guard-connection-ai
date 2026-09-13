const TRANSPORT_SCHEMA_VERSION = "1.0";
const WAVEFORM_SCHEMA_VERSION = "1.1";

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

export function validateWaveformEnvelope(envelope) {
  requireCondition(envelope?.message_type === "waveform_frame", "message_type が不正です");
  requireCondition(envelope.schema_version === TRANSPORT_SCHEMA_VERSION, "未対応の通信schema_versionです");
  const frame = envelope.frame;
  requireCondition(frame && frame.schema_version === WAVEFORM_SCHEMA_VERSION, "frame schemaが不正です");
  requireCondition(frame.payload_type === "doctor_waveform_visualization", "表示用payloadではありません");
  requireCondition(typeof frame.session_id === "string" && frame.session_id.length > 0, "session_idが必要です");
  requireCondition(Number.isInteger(frame.sequence_number) && frame.sequence_number >= 0, "sequence_numberが不正です");
  requireCondition(Number.isFinite(frame.sample_rate_hz) && frame.sample_rate_hz > 0, "sample_rate_hzが不正です");
  requireCondition(Array.isArray(frame.ppg) && Array.isArray(frame.ppg_valid), "PPG配列が必要です");
  requireCondition(Array.isArray(frame.pseudo_ecg) && Array.isArray(frame.pseudo_ecg_valid), "疑似ECG配列が必要です");
  const size = frame.ppg.length;
  requireCondition(size > 0, "空の波形は受信できません");
  requireCondition(
    frame.ppg_valid.length === size && frame.pseudo_ecg.length === size && frame.pseudo_ecg_valid.length === size,
    "波形とvalid maskの長さが一致しません",
  );
  requireCondition(frame.waveform_type === "ppg_derived_pseudo_ecg", "疑似ECGの型が不正です");
  requireCondition(frame.diagnostic_ecg === false, "疑似ECGを診断用ECGとして受信できません");
  requireCondition(frame.measurement_ui_enabled === false, "疑似ECGの測定UIを有効化できません");
  requireCondition(
    ["interval_template", "population_prior", "learned_morphology", "blank"].includes(frame.generation_mode),
    "generation_mode が不正です",
  );
  requireCondition(typeof frame.morphology_source === "string" && frame.morphology_source.length > 0, "morphology_source が必要です");
  requireCondition(frame.p_wave_generated === false, "疑似ECGではP波を生成できません");
  requireCondition(frame.qt_measurement_supported === false, "疑似ECGではQT測定を有効化できません");
  requireCondition(frame.pq_pr_measurement_supported === false, "疑似ECGではPQ/PR測定を有効化できません");
  requireCondition(typeof frame.generation_accepted === "boolean", "生成可否が必要です");
  if (envelope.source_provenance != null) {
    const provenance = envelope.source_provenance;
    requireCondition(
      ["live_device", "recorded_demo_replay", "synthetic_demo"].includes(provenance.source_mode),
      "入力データのsource_modeが不正です",
    );
    requireCondition(
      provenance.source_mode === "live_device" ||
        (typeof provenance.demo_scenario_id === "string" && provenance.demo_scenario_id.length > 0),
      "記録再生にはdemo_scenario_idが必要です",
    );
  }
  return frame;
}

export function createReceiveState() {
  return { sessionId: null, expectedSequenceNumber: 0 };
}

export function receiveWaveformEnvelope(previousState, envelope) {
  const frame = validateWaveformEnvelope(envelope);
  const state = previousState || createReceiveState();
  const sessionChanged = state.sessionId !== frame.session_id;
  const expected = sessionChanged ? 0 : state.expectedSequenceNumber;
  if (frame.sequence_number < expected) {
    return {
      state,
      accepted: false,
      issue: frame.sequence_number === expected - 1 ? "duplicate_packet" : "out_of_order",
      graph: null,
    };
  }
  if (frame.sequence_number > expected) {
    return { state, accepted: false, issue: "sequence_gap", graph: null };
  }

  const ppgData = frame.ppg.map((value, index) => ({
    seq: index,
    timestampSeconds: frame.input_timestamp_start_seconds + index / frame.sample_rate_hz,
    time: frame.input_timestamp_start_seconds + index / frame.sample_rate_hz,
    value: frame.ppg_valid[index] ? value : null,
  }));
  const pseudoEcgData = frame.pseudo_ecg.map((value, index) => ({
    seq: index,
    timestampSeconds: frame.display_timestamp_start_seconds + index / frame.sample_rate_hz,
    time: frame.display_timestamp_start_seconds + index / frame.sample_rate_hz,
    value: frame.generation_accepted && frame.pseudo_ecg_valid[index] ? value : null,
  }));
  return {
    state: { sessionId: frame.session_id, expectedSequenceNumber: expected + 1 },
    accepted: true,
    issue: null,
    graph: {
      ppgData,
      pseudoEcgData,
      frame,
      sourceProvenance: envelope.source_provenance ?? null,
    },
  };
}

export function waveformDisplayName(waveformType, diagnosticEcg) {
  if (waveformType === "ppg_derived_pseudo_ecg" && diagnosticEcg === false) {
    return "PPG由来疑似ECG・診断用ではない";
  }
  if (waveformType === "measured_ecg" && diagnosticEcg === true) return "実測ECG";
  if (waveformType === "simulated_ecg" && diagnosticEcg === false) {
    return "表示用シミュレーションECG・PPG非対応・診断用ではない";
  }
  return "波形種別不明・診断利用不可";
}

export function formatWaveformTooltipValue(value, decimals = 3) {
  requireCondition(Number.isInteger(decimals) && decimals >= 0 && decimals <= 6, "小数桁数が不正です");
  return Number(value).toFixed(decimals);
}
