export const SIGNAL_SOURCE_TIMEOUT_MS = 1500;

export function createSignalSourceState() {
  return { kind: null, sessionId: null, lastSeenMs: 0 };
}

export function isSupplementalSimulatedEcg(payload) {
  return (
    payload?.signal_type === "ECG" &&
    payload.waveform_type === "simulated_ecg" &&
    payload.diagnostic_ecg === false &&
    payload.display_role === "supplemental_simulated_ecg"
  );
}

export function selectSignalSource(
  previous,
  { kind, sessionId, nowMs },
  timeoutMs = SIGNAL_SOURCE_TIMEOUT_MS,
) {
  const state = previous || createSignalSourceState();
  const expired = state.kind != null && nowMs - state.lastSeenMs > timeoutMs;
  const sameSource = state.kind === kind && state.sessionId === sessionId;
  const realtimeTakesPriority = kind === "realtime" && state.kind === "legacy";

  if (state.kind == null || expired || sameSource || realtimeTakesPriority) {
    const switched = state.kind != null && !sameSource;
    return {
      accepted: true,
      switched,
      conflict: false,
      state: { kind, sessionId, lastSeenMs: nowMs },
    };
  }

  return {
    accepted: false,
    switched: false,
    conflict: true,
    state,
  };
}

export function latestTimestamp(points) {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    if (Number.isFinite(points[index]?.timestampSeconds)) {
      return points[index].timestampSeconds;
    }
  }
  return null;
}

export function visibleTimeRange(seriesList, requestedSeconds) {
  const populated = seriesList.filter(series => series.some(
    point => Number.isFinite(point.timestampSeconds) && point.value != null,
  ));
  const latestValues = populated.map(latestTimestamp).filter(Number.isFinite);
  if (latestValues.length === 0) return null;

  const endSeconds = Math.min(...latestValues);
  const earliestValues = populated
    .map(series => series.find(point => Number.isFinite(point.timestampSeconds))?.timestampSeconds)
    .filter(Number.isFinite);
  const availableSeconds = Math.max(0, endSeconds - Math.min(...earliestValues));
  return {
    endSeconds,
    durationSeconds: Math.min(requestedSeconds, Math.max(0.25, availableSeconds)),
  };
}
