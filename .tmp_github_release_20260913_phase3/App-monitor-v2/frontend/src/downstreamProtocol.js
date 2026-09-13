function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

export function receiveDownstreamInferenceEnvelope(envelope) {
  requireCondition(envelope?.message_type === "downstream_inference", "message_typeが不正です");
  requireCondition(envelope.schema_version === "downstream_v1", "未対応の後段AI schemaです");
  const summary = envelope.monitoring_summary;
  const inference = envelope.inference;
  requireCondition(inference && typeof inference === "object", "inferenceが必要です");
  requireCondition(["model", "demo_stub"].includes(inference.inference_mode), "inference_modeが不正です");
  requireCondition(summary && typeof summary === "object", "monitoring_summaryが必要です");
  requireCondition(
    ["monitoring", "af_suspected", "undecidable"].includes(summary.analysis_state),
    "analysis_stateが不正です",
  );
  requireCondition(Number.isFinite(summary.valid_ratio) && summary.valid_ratio >= 0 && summary.valid_ratio <= 1, "valid_ratioが不正です");
  requireCondition(Number.isFinite(summary.observed_seconds) && summary.observed_seconds >= 0, "observed_secondsが不正です");
  requireCondition(Number.isFinite(summary.consecutive_undecidable_seconds) && summary.consecutive_undecidable_seconds >= 0, "判定不能時間が不正です");
  if (summary.analysis_state === "undecidable") {
    requireCondition(summary.decidable === false, "判定不能状態のdecidableが不正です");
    requireCondition(typeof summary.last_abstention_reason === "string" && summary.last_abstention_reason.length > 0, "判定不能理由が必要です");
  }
  return {
    summary,
    inference,
    sourceProvenance: envelope.source_provenance ?? null,
  };
}

export function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : "---";
}
