import test from "node:test";
import assert from "node:assert/strict";

import { formatPercent, receiveDownstreamInferenceEnvelope } from "./downstreamProtocol.js";

function envelope(overrides = {}) {
  return {
    message_type: "downstream_inference",
    schema_version: "downstream_v1",
    source_provenance: { source_mode: "synthetic_demo", demo_scenario_id: "demo-1" },
    inference: {
      decision: "no_af_suspected",
      af_probability: 0.1,
      inference_mode: "model",
    },
    monitoring_summary: {
      analysis_state: "monitoring",
      decidable: true,
      valid_ratio: 0.8,
      observed_seconds: 35,
      consecutive_undecidable_seconds: 0,
      last_abstention_reason: null,
      ...overrides,
    },
  };
}

test("後段AIの3状態を受信する", () => {
  assert.equal(receiveDownstreamInferenceEnvelope(envelope()).summary.analysis_state, "monitoring");
  assert.equal(receiveDownstreamInferenceEnvelope(envelope({ analysis_state: "af_suspected" })).summary.analysis_state, "af_suspected");
});

test("判定不能には理由を必須とする", () => {
  assert.throws(
    () => receiveDownstreamInferenceEnvelope(envelope({ analysis_state: "undecidable", decidable: false })),
    /理由/,
  );
  const received = receiveDownstreamInferenceEnvelope(envelope({
    analysis_state: "undecidable",
    decidable: false,
    last_abstention_reason: "low_sqi",
    consecutive_undecidable_seconds: 5,
  }));
  assert.equal(received.summary.last_abstention_reason, "low_sqi");
});

test("比率は整数パーセント表示にする", () => {
  assert.equal(formatPercent(0.846), "85%");
  assert.equal(formatPercent(null), "---");
});
