import test from "node:test";
import assert from "node:assert/strict";

import {
  createSignalSourceState,
  isSupplementalSimulatedEcg,
  selectSignalSource,
  visibleTimeRange,
} from "./waveformDisplay.js";

test("リアルタイム入力は旧sender表示を引き継がず切り替える", () => {
  const legacy = selectSignalSource(createSignalSourceState(), {
    kind: "legacy", sessionId: "legacy-a", nowMs: 100,
  });
  const realtime = selectSignalSource(legacy.state, {
    kind: "realtime", sessionId: "realtime-a", nowMs: 200,
  });
  assert.equal(realtime.accepted, true);
  assert.equal(realtime.switched, true);
  assert.equal(realtime.state.kind, "realtime");
});

test("補助シミュレーションECGは非診断用の明示を必須とする", () => {
  assert.equal(isSupplementalSimulatedEcg({
    signal_type: "ECG",
    waveform_type: "simulated_ecg",
    diagnostic_ecg: false,
    display_role: "supplemental_simulated_ecg",
  }), true);
  assert.equal(isSupplementalSimulatedEcg({
    signal_type: "ECG",
    waveform_type: "measured_ecg",
    diagnostic_ecg: true,
    display_role: "supplemental_simulated_ecg",
  }), false);
});

test("別sessionの同時入力はtimeoutまでは混在させない", () => {
  const first = selectSignalSource(createSignalSourceState(), {
    kind: "realtime", sessionId: "session-a", nowMs: 100,
  });
  const conflict = selectSignalSource(first.state, {
    kind: "realtime", sessionId: "session-b", nowMs: 200,
  });
  assert.equal(conflict.accepted, false);
  assert.equal(conflict.conflict, true);

  const takeover = selectSignalSource(first.state, {
    kind: "realtime", sessionId: "session-b", nowMs: 2000,
  });
  assert.equal(takeover.accepted, true);
  assert.equal(takeover.switched, true);
});

test("PPGと疑似ECGは共通の最新時刻で固定長表示する", () => {
  const range = visibleTimeRange([
    [{ timestampSeconds: 10, value: 1 }, { timestampSeconds: 15, value: 2 }],
    [{ timestampSeconds: 9.5, value: 1 }, { timestampSeconds: 14.5, value: 2 }],
  ], 5);
  assert.deepEqual(range, { endSeconds: 14.5, durationSeconds: 5 });
});

test("全点無効の疑似ECGはPPG表示範囲を空白にしない", () => {
  const range = visibleTimeRange([
    [{ timestampSeconds: 10, value: 1 }, { timestampSeconds: 10.25, value: 2 }],
    [{ timestampSeconds: 9.5, value: null }, { timestampSeconds: 9.75, value: null }],
  ], 5);
  assert.deepEqual(range, { endSeconds: 10.25, durationSeconds: 0.25 });
});
