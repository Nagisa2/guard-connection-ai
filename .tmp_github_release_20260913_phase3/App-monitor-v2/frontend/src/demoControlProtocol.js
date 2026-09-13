const SCENARIOS = new Set(["af_waiting", "af_unwell", "af_emergency"]);

export function parseDemoControlLocation(search) {
  const params = new URLSearchParams(search);
  return params.get("mode") === "control" ? { mode: "control" } : null;
}

export function buildDemoStartRequest(scenario, loop = true) {
  if (!SCENARIOS.has(scenario)) throw new Error("未対応のデモシナリオです");
  return { schema_version: "1.0", scenario, loop };
}

export function parseDemoControlStatus(payload) {
  if (payload?.message_type !== "demo_control_status" || payload.schema_version !== "1.0") {
    throw new Error("デモ制御応答の形式が不正です");
  }
  return {
    running: payload.running === true,
    scenario: payload.scenario ?? null,
    startedAtSeconds: payload.started_at_seconds ?? null,
    exitCode: payload.exit_code ?? null,
    recentLog: Array.isArray(payload.recent_log) ? payload.recent_log : [],
  };
}

export function parseDemoSystemStatus(payload) {
  if (payload?.message_type !== "demo_system_status" || payload.schema_version !== "1.0") {
    throw new Error("システム状態応答の形式が不正です");
  }
  return {
    ready: payload.ready === true,
    patientCount: Number(payload.patient_count ?? 0),
    doctorConnectionCount: Number(payload.doctor_view_connection_count ?? 0),
    activeSessionCount: Number(payload.realtime?.active_session_count ?? 0),
    staleSessionCount: Number(payload.stale_session_count ?? 0),
    sessions: Array.isArray(payload.realtime?.sessions) ? payload.realtime.sessions : [],
    downstreamSessions: Array.isArray(payload.downstream?.sessions) ? payload.downstream.sessions : [],
    motionStreamCount: Number(payload.motion?.active_stream_count ?? 0),
    inferenceModes: Array.isArray(payload.inference_modes) ? payload.inference_modes : [],
    downstreamOperatingMode: payload.downstream_operating_mode ?? "unconnected",
    pendingDownstreamWindowCount: Array.isArray(payload.downstream_windows?.sessions)
      ? payload.downstream_windows.sessions.reduce(
          (total, item) => total + Number(item.pending_window_count ?? 0),
          0,
        )
      : 0,
    interfaces: payload.interfaces ?? {},
  };
}
