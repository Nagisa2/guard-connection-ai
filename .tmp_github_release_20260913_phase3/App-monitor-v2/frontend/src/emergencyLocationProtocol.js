export const DEMO_RESPONSE_ORIGIN = Object.freeze({
  latitude: 33.9631522,
  longitude: 132.1807969,
  label: "デモ救急車待機地点（大畠駅付近）",
});

export function mapInitializationKey(location) {
  const latitude = Number(location?.latitude);
  const longitude = Number(location?.longitude);
  return Number.isFinite(latitude) && Number.isFinite(longitude)
    ? `${latitude},${longitude}`
    : null;
}

export function emergencyLocationViewModel(location, state) {
  if (!location) return null;
  const latitude = Number(location.latitude);
  const longitude = Number(location.longitude);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  return {
    latitude,
    longitude,
    accuracyLabel: Number.isFinite(location.accuracy_m)
      ? `精度 約${Math.round(location.accuracy_m)} m`
      : "精度不明",
    capturedAtLabel: Number.isFinite(location.captured_at_seconds)
      ? new Date(location.captured_at_seconds * 1000).toLocaleTimeString("ja-JP")
      : "時刻不明",
    providerLabel: location.provider || "不明",
    urgent: state === "emergency_escalated",
    destination: { latitude, longitude },
    responseOrigin: DEMO_RESPONSE_ORIGIN,
    mapUrl: `https://www.openstreetmap.org/?mlat=${latitude}&mlon=${longitude}#map=16/${latitude}/${longitude}`,
  };
}

export function buildRouteUrl(origin, destination) {
  return "https://router.project-osrm.org/route/v1/driving/"
    + `${origin.longitude},${origin.latitude};${destination.longitude},${destination.latitude}`
    + "?overview=full&geometries=geojson&steps=false";
}

export function parseRouteResponse(payload) {
  const route = payload?.routes?.[0];
  if (!route || route.geometry?.type !== "LineString" || !Array.isArray(route.geometry.coordinates)) {
    throw new Error("経路データがありません");
  }
  return {
    coordinates: route.geometry.coordinates.map(([longitude, latitude]) => [latitude, longitude]),
    distanceKm: route.distance / 1000,
    durationMinutes: route.duration / 60,
  };
}
