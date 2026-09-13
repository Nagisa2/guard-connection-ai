import { useCallback, useEffect, useState } from "react";

import {
  buildCareActionRequest,
  carePortalViewModel,
  mergePolledCareEvent,
} from "./carePortalProtocol.js";
import EmergencyLocationPanel from "./EmergencyLocationPanel.jsx";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function requestId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now()}-${Math.random()}`;
}

export default function CareDemoPortal({ mode, userId }) {
  const [event, setEvent] = useState(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/care-events?user_id=${encodeURIComponent(userId)}`,
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      setEvent((current) => mergePolledCareEvent(current, payload.events?.[0]));
      setError("");
    } catch (refreshError) {
      setError(`サーバーに接続できません: ${refreshError.message}`);
    }
  }, [userId]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [refresh]);

  const submitAction = async (action) => {
    if (!event) return;
    setMessage("送信中...");
    setError("");
    try {
      const body = buildCareActionRequest(mode, action, requestId());
      const response = await fetch(`${API_BASE_URL}/care-events/${event.event_id}/actions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok || payload.message_type === "error") {
        throw new Error(payload.error?.message || `HTTP ${response.status}`);
      }
      setEvent(payload.event);
      setMessage("回答を送信しました。");
    } catch (submitError) {
      setMessage("");
      setError(`送信できませんでした: ${submitError.message}`);
    }
  };

  const isPatient = mode === "patient";
  const view = carePortalViewModel(mode, event);
  return (
    <main style={styles.page}>
      <section style={styles.panel}>
        <div style={styles.demoBadge}>LOCAL DEMO</div>
        <h1>{view.roleLabel}</h1>
        <p style={styles.warning}>
          この画面は接続試験用です。AF確定診断や実際の緊急通報は行いません。
        </p>
        <p>対象ユーザー: <strong>{userId}</strong></p>
        {error && <p style={styles.error}>{error}</p>}
        {!event ? (
          <div style={styles.empty}>
            AF疑いイベントを待っています。医師画面のデモを開始してください。
          </div>
        ) : (
          <div style={styles.eventCard}>
            <p>現在の状態</p>
            <h2>{view.stateLabel}</h2>
            <p>
              {view.inferenceLabel}
              ・確定診断ではありません
            </p>
            <EmergencyLocationPanel location={event.location} state={event.state} />
            {isPatient ? (
              <div style={styles.actions}>
                <button type="button" onClick={() => submitAction("patient_ok")}>
                  問題ありません
                </button>
                <button type="button" onClick={() => submitAction("patient_unwell")}>
                  体調が悪いです
                </button>
                <button
                  type="button"
                  style={styles.helpButton}
                  onClick={() => submitAction("patient_help")}
                >
                  助けを呼ぶ
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => submitAction("family_acknowledged")}
              >
                通知を確認しました
              </button>
            )}
            {message && <p style={styles.message}>{message}</p>}
          </div>
        )}
      </section>
    </main>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    display: "grid",
    placeItems: "center",
    padding: "24px",
    background: "#eef2ff",
    color: "#172033",
  },
  panel: {
    width: "min(620px, 100%)",
    padding: "28px",
    borderRadius: "18px",
    background: "white",
    boxShadow: "0 16px 50px rgba(30, 41, 59, 0.15)",
  },
  demoBadge: {
    display: "inline-block",
    padding: "5px 10px",
    borderRadius: "999px",
    background: "#ffedd5",
    color: "#9a3412",
    fontWeight: 700,
  },
  warning: {
    padding: "12px",
    border: "1px solid #fb923c",
    borderRadius: "10px",
    background: "#fff7ed",
  },
  empty: { padding: "24px", borderRadius: "12px", background: "#f8fafc" },
  eventCard: { padding: "20px", borderRadius: "12px", background: "#f8fafc" },
  actions: { display: "grid", gap: "12px", marginTop: "20px" },
  helpButton: { background: "#dc2626", color: "white" },
  error: { color: "#b91c1c" },
  message: { color: "#166534", fontWeight: 700 },
};
