import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRouteUrl,
  emergencyLocationViewModel,
  mapInitializationKey,
  parseRouteResponse,
} from "./emergencyLocationProtocol.js";

test("位置情報を医師・患者・家族画面用の表示モデルに変換する", () => {
  const view = emergencyLocationViewModel({
    latitude: 34.123456,
    longitude: 132.654321,
    accuracy_m: 18.4,
    captured_at_seconds: 1_700_000_000,
    provider: "demo_fixed_coordinate",
  }, "emergency_escalated");

  assert.equal(view.urgent, true);
  assert.equal(view.accuracyLabel, "精度 約18 m");
  assert.match(view.mapUrl, /openstreetmap\.org/);
  assert.equal(view.destination.latitude, 34.123456);
  assert.equal(view.responseOrigin.latitude, 33.9631522);
});

test("同じ座標ならポーリングでlocationオブジェクトが変わっても地図キーは変わらない", () => {
  const first = { latitude: 33.938502, longitude: 132.190863, captured_at_seconds: 1 };
  const polled = { latitude: 33.938502, longitude: 132.190863, captured_at_seconds: 2 };
  assert.equal(mapInitializationKey(first), mapInitializationKey(polled));
  assert.equal(mapInitializationKey(first), "33.938502,132.190863");
});

test("欠損または不正な位置は表示しない", () => {
  assert.equal(emergencyLocationViewModel(null, "monitoring"), null);
  assert.equal(emergencyLocationViewModel({ latitude: "x", longitude: 1 }, "monitoring"), null);
});

test("OSRM用の経路URLと応答を緯度経度順へ変換する", () => {
  const origin = { latitude: 33.96, longitude: 132.18 };
  const destination = { latitude: 33.94, longitude: 132.19 };
  assert.match(buildRouteUrl(origin, destination), /132\.18,33\.96;132\.19,33\.94/);
  const route = parseRouteResponse({
    routes: [{ distance: 2400, duration: 360, geometry: { type: "LineString", coordinates: [[132.18, 33.96], [132.19, 33.94]] } }],
  });
  assert.deepEqual(route.coordinates, [[33.96, 132.18], [33.94, 132.19]]);
  assert.equal(route.distanceKm, 2.4);
  assert.equal(route.durationMinutes, 6);
});
