import test from "node:test";
import assert from "node:assert/strict";

import {
  buildCareActionRequest,
  carePortalViewModel,
  mergePolledCareEvent,
  parseCarePortalLocation,
} from "./carePortalProtocol.js";

test("患者・家族デモURLを解析する", () => {
  assert.deepEqual(parseCarePortalLocation("?mode=patient&user_id=test_user_02"), {
    mode: "patient",
    userId: "test_user_02",
  });
  assert.deepEqual(parseCarePortalLocation("?mode=family"), {
    mode: "family",
    userId: "test_user_02",
  });
  assert.equal(parseCarePortalLocation("?mode=doctor"), null);
});

test("同じrevisionのポーリング応答では既存eventオブジェクトを維持する", () => {
  const current = { event_id: "event-1", revision: 3, state: "emergency_escalated" };
  const unchanged = { event_id: "event-1", revision: 3, state: "emergency_escalated" };
  const changed = { event_id: "event-1", revision: 4, state: "resolved" };

  assert.equal(mergePolledCareEvent(current, unchanged), current);
  assert.equal(mergePolledCareEvent(current, changed), changed);
});

test("患者画面の状態とデモ判定表示を構築する", () => {
  const view = carePortalViewModel("patient", {
    state: "awaiting_patient_response",
    ai_result: { inference_mode: "demo_stub" },
  });
  assert.equal(view.roleLabel, "患者アプリ・デモ操作");
  assert.equal(view.stateLabel, "体調確認への回答待ち");
  assert.equal(view.inferenceLabel, "デモ用AF疑い（判定スタブ）");
  assert.deepEqual(view.actions, ["patient_ok", "patient_unwell", "patient_help"]);
});

test("ロール外の操作をクライアント側でも拒否する", () => {
  assert.deepEqual(buildCareActionRequest("patient", "patient_help", "request-1"), {
    idempotency_key: "request-1",
    action: "patient_help",
    actor_role: "patient",
  });
  assert.throws(
    () => buildCareActionRequest("family", "patient_help", "request-2"),
    /not available/,
  );
});
