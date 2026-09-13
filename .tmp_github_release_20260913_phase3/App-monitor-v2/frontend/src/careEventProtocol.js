const CARE_EVENT_SCHEMA_VERSION = "1.0";

const STATE_TO_STATUS = {
  awaiting_patient_response: "af_suspected",
  monitoring: "monitoring",
  clinician_review: "clinician_review",
  emergency_escalated: "emergency",
  resolved: "resolved",
};

const STATUS_PRIORITY = {
  emergency: 5,
  clinician_review: 4,
  af_suspected: 3,
  ppg_anomaly: 2,
  ecg_anomaly: 2,
  monitoring: 1,
  normal: 0,
  resolved: 0,
  unknown: -1,
};

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

export function receiveCareEventEnvelope(envelope) {
  requireCondition(envelope?.message_type === "care_event", "care_eventではありません");
  requireCondition(envelope.schema_version === CARE_EVENT_SCHEMA_VERSION, "未対応のcare event schemaです");
  const event = envelope.event;
  requireCondition(event?.schema_version === CARE_EVENT_SCHEMA_VERSION, "event schemaが不正です");
  requireCondition(typeof event.event_id === "string" && event.event_id.length > 0, "event_idが必要です");
  requireCondition(event.event_type === "af_suspected", "未対応のevent_typeです");
  requireCondition(event.diagnostic_result === false, "AF疑いを確定診断として表示できません");
  requireCondition(STATE_TO_STATUS[event.state], "未対応のevent stateです");
  requireCondition(event.ai_result?.result_source === "downstream_ai", "後段AIの出力ではありません");
  requireCondition(Number.isFinite(event.ai_result.af_probability), "AF確率が必要です");
  requireCondition(
    event.ai_result.af_probability >= 0 && event.ai_result.af_probability <= 1,
    "AF確率が範囲外です",
  );
  requireCondition(
    typeof event.ai_result.model_version === "string" && event.ai_result.model_version.length > 0,
    "model_versionが必要です",
  );
  requireCondition(
    ["model", "demo_stub"].includes(event.ai_result.inference_mode),
    "inference_modeが不正です",
  );
  if (event.ai_result.inference_mode === "demo_stub") {
    requireCondition(event.demo_mode, "demo stubを実機結果として表示できません");
  }
  if (event.demo_mode) {
    requireCondition(
      ["recorded_demo_replay", "synthetic_demo"].includes(event.source_mode),
      "デモ入力のsource_modeが不正です",
    );
    requireCondition(
      typeof event.demo_scenario_id === "string" && event.demo_scenario_id.length > 0,
      "デモ入力にはscenario IDが必要です",
    );
  }
  return {
    event,
    status: STATE_TO_STATUS[event.state],
    sourceLabel:
      event.source_mode === "recorded_demo_replay"
        ? "デモ用記録済み信号"
        : event.source_mode === "synthetic_demo"
          ? "デモ用合成信号"
          : "実機ライブ信号",
    inferenceLabel:
      event.ai_result.inference_mode === "demo_stub"
        ? "デモ用AF疑い（判定スタブ）"
        : "後段AIによるAF疑い",
    probabilityLabel: `${(event.ai_result.af_probability * 100).toFixed(1)}%`,
  };
}

export function careEventStatusPriority(status) {
  return STATUS_PRIORITY[status] ?? STATUS_PRIORITY.unknown;
}
