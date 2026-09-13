const MODES = new Set(["patient", "family"]);

export function mergePolledCareEvent(currentEvent, nextEvent) {
  if (
    currentEvent
    && nextEvent
    && currentEvent.event_id === nextEvent.event_id
    && currentEvent.revision === nextEvent.revision
  ) {
    return currentEvent;
  }
  return nextEvent ?? null;
}

export function parseCarePortalLocation(search) {
  const params = new URLSearchParams(search);
  const mode = params.get("mode");
  if (!MODES.has(mode)) return null;
  const userId = params.get("user_id") || "test_user_02";
  return { mode, userId };
}

export function buildCareActionRequest(mode, action, idempotencyKey) {
  const allowed = {
    patient: new Set(["patient_ok", "patient_unwell", "patient_help"]),
    family: new Set(["family_acknowledged"]),
  };
  if (!allowed[mode]?.has(action)) {
    throw new Error("This action is not available for the selected demo role.");
  }
  if (!idempotencyKey) throw new Error("idempotencyKey is required.");
  return {
    idempotency_key: idempotencyKey,
    action,
    actor_role: mode,
  };
}

export function carePortalViewModel(mode, event) {
  const stateLabels = {
    awaiting_patient_response: "体調確認への回答待ち",
    monitoring: "経過観察",
    clinician_review: "医師が確認中",
    emergency_escalated: "緊急対応中",
    resolved: "対応完了",
  };
  return {
    roleLabel: mode === "patient" ? "患者アプリ・デモ操作" : "家族アプリ・デモ操作",
    stateLabel: event ? stateLabels[event.state] || event.state : null,
    inferenceLabel:
      event?.ai_result?.inference_mode === "demo_stub"
        ? "デモ用AF疑い（判定スタブ）"
        : "後段AIによるAF疑い",
    actions:
      mode === "patient"
        ? ["patient_ok", "patient_unwell", "patient_help"]
        : ["family_acknowledged"],
  };
}
