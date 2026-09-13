import test from "node:test";
import assert from "node:assert/strict";

import { careEventStatusPriority, receiveCareEventEnvelope } from "./careEventProtocol.js";

function envelope(overrides = {}) {
  return {
    message_type: "care_event",
    schema_version: "1.0",
    changed: true,
    event: {
      schema_version: "1.0",
      event_id: "event-1",
      event_type: "af_suspected",
      state: "awaiting_patient_response",
      severity: "warning",
      source_mode: "recorded_demo_replay",
      demo_mode: true,
      demo_scenario_id: "held-out-af-01",
      diagnostic_result: false,
      ai_result: {
        result_source: "downstream_ai",
        af_probability: 0.91,
        uncertainty: 0.08,
        model_version: "af-v1",
        inference_mode: "model",
      },
      ...overrides,
    },
  };
}

test("AF疑いを診断ではなく後段AI由来イベントとして受信する", () => {
  const result = receiveCareEventEnvelope(envelope());
  assert.equal(result.status, "af_suspected");
  assert.equal(result.sourceLabel, "デモ用記録済み信号");
  assert.equal(result.probabilityLabel, "91.0%");
  assert.equal(result.inferenceLabel, "後段AIによるAF疑い");
});

test("緊急状態を患者一覧の最上位優先度にする", () => {
  const result = receiveCareEventEnvelope(envelope({ state: "emergency_escalated" }));
  assert.equal(result.status, "emergency");
  assert.ok(careEventStatusPriority("emergency") > careEventStatusPriority("af_suspected"));
});

test("診断扱いと出所不明のAF確率を拒否する", () => {
  assert.throws(
    () => receiveCareEventEnvelope(envelope({ diagnostic_result: true })),
    /確定診断/,
  );
  assert.throws(
    () => receiveCareEventEnvelope(envelope({ ai_result: { af_probability: 0.9 } })),
    /後段AI/,
  );
});

test("記録再生ではscenario IDを必須にする", () => {
  assert.throws(
    () => receiveCareEventEnvelope(envelope({ demo_scenario_id: null })),
    /scenario ID/,
  );
});

test("判定スタブを実モデル結果と区別する", () => {
  const result = receiveCareEventEnvelope(envelope({
    ai_result: {
      result_source: "downstream_ai",
      af_probability: 0.91,
      uncertainty: 0.08,
      model_version: "demo-stub-v1",
      inference_mode: "demo_stub",
    },
  }));
  assert.equal(result.inferenceLabel, "デモ用AF疑い（判定スタブ）");
});
