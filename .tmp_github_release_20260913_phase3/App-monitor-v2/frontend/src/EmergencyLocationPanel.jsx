import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import {
  buildRouteUrl,
  DEMO_RESPONSE_ORIGIN,
  emergencyLocationViewModel,
  parseRouteResponse,
} from "./emergencyLocationProtocol.js";

function formatCoordinate(value) {
  return Number.isFinite(value) ? value.toFixed(5) : "---";
}

export default function EmergencyLocationPanel({ location, state }) {
  const latitude = Number(location?.latitude);
  const longitude = Number(location?.longitude);
  const accuracy = location?.accuracy_m;
  const capturedAt = location?.captured_at_seconds;
  const provider = location?.provider;
  const view = useMemo(
    () => emergencyLocationViewModel(
      Number.isFinite(latitude) && Number.isFinite(longitude)
        ? {
            latitude,
            longitude,
            accuracy_m: accuracy,
            captured_at_seconds: capturedAt,
            provider,
          }
        : null,
      state,
    ),
    [latitude, longitude, accuracy, capturedAt, provider, state],
  );
  const mapElementRef = useRef(null);
  const mapRef = useRef(null);
  const [routeSummary, setRouteSummary] = useState(null);
  const [routeMode, setRouteMode] = useState("loading");

  useEffect(() => {
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude) || !mapElementRef.current) {
      return undefined;
    }
    const origin = [DEMO_RESPONSE_ORIGIN.latitude, DEMO_RESPONSE_ORIGIN.longitude];
    const destination = [latitude, longitude];
    const map = L.map(mapElementRef.current, { zoomControl: true }).setView(destination, 14);
    mapRef.current = map;
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
      maxZoom: 19,
    }).addTo(map);
    L.circleMarker(origin, { radius: 8, color: "#1d4ed8", fillOpacity: 1 })
      .bindTooltip(DEMO_RESPONSE_ORIGIN.label)
      .addTo(map);
    L.circleMarker(destination, { radius: 10, color: "#b91c1c", fillColor: "#dc2626", fillOpacity: 1 })
      .bindTooltip("患者位置：大島商船高等専門学校")
      .addTo(map);
    const fallback = L.polyline([origin, destination], {
      color: "#64748b",
      weight: 5,
      dashArray: "8 8",
    }).addTo(map);
    map.fitBounds(L.latLngBounds([origin, destination]), { padding: [24, 24] });
    const controller = new AbortController();
    fetch(buildRouteUrl(DEMO_RESPONSE_ORIGIN, { latitude, longitude }), { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        const route = parseRouteResponse(payload);
        fallback.remove();
        const routeLine = L.polyline(route.coordinates, { color: "#2563eb", weight: 6 }).addTo(map);
        map.fitBounds(routeLine.getBounds(), { padding: [24, 24] });
        setRouteSummary(route);
        setRouteMode("road");
      })
      .catch((error) => {
        if (error.name !== "AbortError") setRouteMode("fallback");
      });
    return () => {
      controller.abort();
      map.remove();
      mapRef.current = null;
    };
  }, [latitude, longitude]);

  if (!view) return null;
  return (
    <section style={{ ...styles.panel, borderColor: view.urgent ? "#dc2626" : "#f59e0b" }}>
      <div style={styles.heading}>
        <div>
          <strong>{view.urgent ? "緊急対応位置" : "患者位置（確認用）"}</strong>
          <div style={styles.meta}>
            {formatCoordinate(view.latitude)}, {formatCoordinate(view.longitude)}
            ・{view.accuracyLabel}・取得 {view.capturedAtLabel}
          </div>
        </div>
        <a href={view.mapUrl} target="_blank" rel="noreferrer" style={styles.link}>
          OpenStreetMapで開く
        </a>
      </div>
      <div ref={mapElementRef} role="img" aria-label="患者位置と救急車向け経路のOpenStreetMap" style={styles.map} />
      <div style={styles.routeSummary}>
        <strong>経路：</strong>{view.responseOrigin.label} → 大島商船高等専門学校
        {routeMode === "loading" && "（道路経路を取得中）"}
        {routeMode === "road" && routeSummary && `（約${routeSummary.distanceKm.toFixed(1)} km・約${Math.ceil(routeSummary.durationMinutes)}分）`}
        {routeMode === "fallback" && "（経路APIに接続できないため直線表示）"}
      </div>
      <p style={styles.note}>
        位置と出発地点はデモ用の固定値です。実際の消防署・救急車位置、交通状況、救急通報とは連動しません。
      </p>
    </section>
  );
}

const styles = {
  panel: { marginTop: "16px", padding: "14px", border: "2px solid", borderRadius: "12px", background: "#fff" },
  heading: { display: "flex", justifyContent: "space-between", gap: "12px", alignItems: "center", flexWrap: "wrap" },
  meta: { marginTop: "4px", color: "#475569", fontSize: "12px" },
  link: { color: "#1d4ed8", fontWeight: 700, fontSize: "13px" },
  map: { width: "100%", height: "300px", marginTop: "12px", borderRadius: "8px", border: "1px solid #cbd5e1" },
  routeSummary: { marginTop: "10px", color: "#1e3a8a", fontSize: "13px" },
  note: { margin: "8px 0 0", color: "#9a3412", fontSize: "12px" },
};
