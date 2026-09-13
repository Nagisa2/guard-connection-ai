import { useCallback, useEffect, useState } from "react";

import { buildDemoStartRequest, parseDemoControlStatus, parseDemoSystemStatus } from "./demoControlProtocol.js";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const SCENARIOS = [
  { id: "af_waiting", label: "AF疑い → 本人回答待ち", description: "通知後に患者操作を手動で見せる" },
  { id: "af_unwell", label: "AF疑い → 体調不良", description: "医師確認が必要な状態まで自動進行" },
  { id: "af_emergency", label: "AF疑い → 救助要請", description: "緊急化・位置・経路表示まで自動進行" },
];

export default function DemoControlPortal() {
  const [scenario, setScenario] = useState("af_emergency");
  const [status, setStatus] = useState({ running: false, recentLog: [] });
  const [system, setSystem] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/demo-control/status`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setStatus(parseDemoControlStatus(await response.json()));
      const systemResponse = await fetch(`${API_BASE_URL}/demo-control/system-status`);
      if (!systemResponse.ok) throw new Error(`system HTTP ${systemResponse.status}`);
      setSystem(parseDemoSystemStatus(await systemResponse.json()));
      setError("");
    } catch (statusError) {
      setError(`バックエンドに接続できません: ${statusError.message}`);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [refresh]);

  const request = async (path, body) => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
      setStatus(parseDemoControlStatus(payload));
    } catch (requestError) {
      setError(`操作に失敗しました: ${requestError.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main style={styles.page}>
      <section style={styles.panel}>
        <div style={styles.badge}>CONTEST DEMO CONTROL</div>
        <h1>GUARD Connection デモ操作</h1>
        <p style={styles.warning}>AF判定はデモ用スタブです。確定診断・実際の救急通報は行いません。</p>
        <div style={styles.status}>
          <span style={{ ...styles.dot, background: status.running ? "#22c55e" : "#94a3b8" }} />
          {status.running ? "デモ実行中" : "停止中"}
          {status.scenario && `・${SCENARIOS.find((item) => item.id === status.scenario)?.label || status.scenario}`}
        </div>
        <section style={styles.healthSection}>
          <h2 style={styles.sectionTitle}>システム稼働状況</h2>
          <div style={styles.healthGrid}>
            <HealthCard label="患者DB" value={system ? `${system.patientCount}名` : "確認中"} ok={system?.patientCount >= 2} />
            <HealthCard label="信号セッション" value={system ? `${system.activeSessionCount}件` : "確認中"} ok={system?.activeSessionCount > 0 && system?.staleSessionCount === 0} />
            <HealthCard label="医師画面接続" value={system ? `${system.doctorConnectionCount}接続` : "確認中"} ok={system?.doctorConnectionCount > 0} />
            <HealthCard label="後段AIモード" value={system?.downstreamOperatingMode || "unconnected"} ok={system?.downstreamOperatingMode === "model" || system?.downstreamOperatingMode === "demo_stub"} />
            <HealthCard label="後段AI待機窓" value={system ? `${system.pendingDownstreamWindowCount}件` : "確認中"} ok={system?.pendingDownstreamWindowCount === 0} />
            <HealthCard label="慣性センサー" value={system ? `${system.motionStreamCount}系統` : "確認中"} ok={system?.motionStreamCount > 0} />
          </div>
          {system?.staleSessionCount > 0 && <p style={styles.healthWarning}>⚠ 2秒以上入力のないセッションが {system.staleSessionCount} 件あります。</p>}
          <div style={styles.interfaceRow}>
            <span>村川さん側入力: {system?.interfaces?.device_ppg_ingress?.ready ? "受入可能" : "未確認"}</span>
            <span>加速度・ジャイロ: {system?.interfaces?.device_motion_ingress?.ready ? "受入可能" : "未確認"}</span>
            <span>倉本さん側出力: {system?.interfaces?.downstream_ai_result_ingress?.ready ? "受入可能" : "未確認"}</span>
            <span>前段AI配信: {system?.interfaces?.front_ai_downstream_output?.ready ? "配信可能" : "APIキー未設定"}</span>
            <span>30秒窓: {system?.interfaces?.front_ai_window_output?.ready ? "取得可能" : "APIキー未設定"}</span>
            <span>医師画面配信: {system?.interfaces?.doctor_waveform_stream?.ready ? "利用可能" : "未確認"}</span>
          </div>
          {system?.sessions?.length > 0 && (
            <details style={styles.sessionDetails}>
              <summary>稼働中セッション</summary>
              {system.sessions.map((item) => (
                <div key={item.session_id} style={styles.sessionLine}>
                  <code>{item.user_id}</code>
                  <span>{item.input_mode || "入力待ち"}</span>
                  <span>{item.last_input_age_seconds == null ? "未受信" : `${item.last_input_age_seconds.toFixed(1)}秒前`}</span>
                  <span>seq {item.expected_sequence_number}</span>
                </div>
              ))}
            </details>
          )}
        </section>
        <div style={styles.scenarios}>
          {SCENARIOS.map((item) => (
            <label key={item.id} style={{ ...styles.scenario, borderColor: scenario === item.id ? "#2563eb" : "#cbd5e1" }}>
              <input type="radio" name="scenario" value={item.id} checked={scenario === item.id} onChange={() => setScenario(item.id)} disabled={status.running} />
              <span><strong>{item.label}</strong><small style={styles.small}>{item.description}</small></span>
            </label>
          ))}
        </div>
        <div style={styles.actions}>
          <button type="button" disabled={busy || status.running} onClick={() => request("/demo-control/start", buildDemoStartRequest(scenario, true))} style={styles.startButton}>デモ開始</button>
          <button type="button" disabled={busy || !status.running} onClick={() => request("/demo-control/stop")} style={styles.stopButton}>デモ停止</button>
        </div>
        {error && <p style={styles.error}>{error}</p>}
        <nav style={styles.links}>
          <a href="/" target="_blank">医師画面</a>
          <a href="/?mode=patient&user_id=test_user_02" target="_blank">患者画面</a>
          <a href="/?mode=family&user_id=test_user_02" target="_blank">家族画面</a>
        </nav>
        <details style={styles.log}>
          <summary>直近の実行ログ</summary>
          <pre>{status.recentLog.length ? status.recentLog.join("\n") : "ログはまだありません。"}</pre>
        </details>
      </section>
    </main>
  );
}

function HealthCard({ label, value, ok }) {
  return (
    <div style={{ ...styles.healthCard, borderColor: ok ? "#86efac" : "#cbd5e1" }}>
      <span style={{ ...styles.healthDot, background: ok ? "#22c55e" : "#94a3b8" }} />
      <div><small>{label}</small><strong>{value}</strong></div>
    </div>
  );
}

const styles = {
  page: { minHeight: "100vh", padding: "32px", background: "#e8eef8", color: "#172033", boxSizing: "border-box" },
  panel: { maxWidth: "760px", margin: "0 auto", padding: "28px", borderRadius: "18px", background: "white", boxShadow: "0 16px 50px rgba(30,41,59,.16)" },
  badge: { display: "inline-block", padding: "5px 10px", borderRadius: "999px", background: "#dbeafe", color: "#1d4ed8", fontWeight: 800, fontSize: "12px" },
  warning: { padding: "12px", border: "1px solid #fb923c", borderRadius: "10px", background: "#fff7ed", color: "#9a3412" },
  status: { display: "flex", alignItems: "center", gap: "8px", padding: "14px", borderRadius: "10px", background: "#f8fafc", fontWeight: 700 },
  healthSection: { marginTop: "18px", padding: "16px", border: "1px solid #dbe3ef", borderRadius: "12px", background: "#f8fafc" },
  sectionTitle: { margin: "0 0 12px", fontSize: "17px" },
  healthGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: "9px" },
  healthCard: { display: "flex", alignItems: "center", gap: "9px", padding: "10px", border: "1px solid", borderRadius: "9px", background: "white" },
  healthDot: { width: "9px", height: "9px", borderRadius: "50%", flexShrink: 0 },
  interfaceRow: { display: "flex", flexWrap: "wrap", gap: "8px 16px", marginTop: "12px", fontSize: "12px", color: "#334155" },
  healthWarning: { margin: "10px 0 0", color: "#b45309", fontWeight: 700, fontSize: "13px" },
  sessionDetails: { marginTop: "12px", color: "#475569", fontSize: "13px" },
  sessionLine: { display: "grid", gridTemplateColumns: "minmax(110px, 1fr) 90px 70px 65px", gap: "8px", padding: "7px 0", borderBottom: "1px solid #e2e8f0" },
  dot: { width: "11px", height: "11px", borderRadius: "50%" },
  scenarios: { display: "grid", gap: "10px", margin: "18px 0" },
  scenario: { display: "flex", gap: "10px", padding: "14px", border: "2px solid", borderRadius: "10px", cursor: "pointer" },
  small: { display: "block", marginTop: "4px", color: "#64748b", fontWeight: 400 },
  actions: { display: "flex", gap: "12px" },
  startButton: { flex: 1, padding: "13px", border: 0, borderRadius: "9px", background: "#2563eb", color: "white", fontWeight: 800 },
  stopButton: { flex: 1, padding: "13px", border: 0, borderRadius: "9px", background: "#dc2626", color: "white", fontWeight: 800 },
  error: { color: "#b91c1c", fontWeight: 700 },
  links: { display: "flex", gap: "18px", marginTop: "20px", flexWrap: "wrap" },
  log: { marginTop: "18px", color: "#475569" },
};
