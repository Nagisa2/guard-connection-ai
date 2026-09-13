import assert from "node:assert/strict";
import test from "node:test";

import {
  buildDemoStartRequest,
  parseDemoControlLocation,
  parseDemoControlStatus,
  parseDemoSystemStatus,
} from "./demoControlProtocol.js";

test("control URLだけをデモ操作画面として判定する", () => {
  assert.deepEqual(parseDemoControlLocation("?mode=control"), { mode: "control" });
  assert.equal(parseDemoControlLocation("?mode=patient"), null);
});

test("操作パネル用システム状態を受信する", () => {
  const status = parseDemoSystemStatus({
    message_type: "demo_system_status",
    schema_version: "1.0",
    ready: true,
    patient_count: 8,
    doctor_view_connection_count: 2,
    stale_session_count: 1,
    inference_modes: ["demo_stub"],
    downstream_operating_mode: "demo_stub",
    realtime: { active_session_count: 2, sessions: [{ session_id: "s1" }] },
    downstream: { active_session_count: 1, sessions: [{ session_id: "s1" }] },
    motion: { active_stream_count: 2, streams: [] },
    downstream_windows: {
      sessions: [{ session_id: "s1", pending_window_count: 2 }],
    },
    interfaces: { device_ppg_ingress: { ready: true } },
  });
  assert.equal(status.patientCount, 8);
  assert.equal(status.activeSessionCount, 2);
  assert.equal(status.staleSessionCount, 1);
  assert.equal(status.motionStreamCount, 2);
  assert.deepEqual(status.inferenceModes, ["demo_stub"]);
  assert.equal(status.downstreamOperatingMode, "demo_stub");
  assert.equal(status.pendingDownstreamWindowCount, 2);
});

test("シナリオ開始要求と状態応答を検証する", () => {
  assert.deepEqual(buildDemoStartRequest("af_emergency"), {
    schema_version: "1.0",
    scenario: "af_emergency",
    loop: true,
  });
  const status = parseDemoControlStatus({
    message_type: "demo_control_status",
    schema_version: "1.0",
    running: true,
    scenario: "af_emergency",
    recent_log: ["started"],
  });
  assert.equal(status.running, true);
  assert.deepEqual(status.recentLog, ["started"]);
});
