// HistoryTab.jsx - ホルター心電図サマリ表示

import { useEffect, useMemo, useState } from "react";
import axios from "axios";

export default function HistoryTab({ patientId, token }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${import.meta.env.VITE_API_BASE_URL}/holter/${patientId}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => setReport(res.data))
      .catch((e) => {
        console.error("ホルター履歴取得失敗", e);
        setReport(null);
      })
      .finally(() => setLoading(false));
  }, [patientId, token]);

  const summaryCards = useMemo(() => {
    if (!report) return [];
    return [
      { label: "信号品質", value: report.overview.signal_quality, color: "#2563eb" },
      { label: "AF確率", value: `${report.overview.af_likelihood.toFixed(2)}`, color: "#ef4444" },
      { label: "RR変動", value: `${report.overview.rr_variability_ms} ms`, color: "#f59e0b" },
      { label: "イベント回数", value: `${report.overview.episode_count} 回`, color: "#10b981" },
      { label: "データ件数", value: `${report.num_data_records.toLocaleString()} 件`, color: "#1a1a2e" },
    ];
  }, [report]);

  if (loading) return <div style={s.empty}>長時間心電図を解析中...</div>;
  if (!report) return <div style={s.empty}>ホルター心電図データがまだありません。</div>;

  return (
    <div style={s.container}>
      <div style={s.cardRow}>
        {summaryCards.map(card => (
          <div key={card.label} style={s.card}>
            <p style={s.cardLabel}>{card.label}</p>
            <p style={{ ...s.cardValue, color: card.color }}>{card.value}</p>
          </div>
        ))}
      </div>

      <div style={s.section}>
        <h3 style={s.sectionTitle}>自動生成サマリ</h3>
        <div style={s.summaryBox}>{report.summary}</div>
      </div>

      <div style={s.section}>
        <h3 style={s.sectionTitle}>24時間の心拍トレンド</h3>
        <div style={{ overflowX: "auto" }}>
          <table style={s.table}>
            <thead>
              <tr style={{ background: "#f8fafc" }}>
                <th style={s.th}>時間</th>
                <th style={s.th}>心拍数</th>
                <th style={s.th}>RR間隔</th>
                <th style={s.th}>AFリスク</th>
                <th style={s.th}>RR変動</th>
              </tr>
            </thead>
            <tbody>
              {report.trend.map((point, index) => (
                <tr key={`${point.time}-${index}`} style={{ background: index % 2 === 0 ? "#fff" : "#f8fafc" }}>
                  <td style={s.td}>{point.time}</td>
                  <td style={s.td}>{point.hr} bpm</td>
                  <td style={s.td}>{point.rr_ms} ms</td>
                  <td style={s.td}>{point.af_risk.toFixed(2)}</td>
                  <td style={s.td}>{point.rr_variability_ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={s.section}>
        <h3 style={s.sectionTitle}>AF疑いイベント</h3>
        {report.episodes.length === 0 ? (
          <p style={s.emptyText}>イベントは検出されませんでした。</p>
        ) : (
          <div style={s.eventList}>
            {report.episodes.map((episode, index) => (
              <div key={`${episode.start}-${index}`} style={s.eventCard}>
                <div style={s.eventMeta}>開始時刻: {episode.start}</div>
                <div style={s.eventMeta}>継続時間: {episode.duration_min} 分</div>
                <div style={s.eventMeta}>種別: {episode.type}</div>
                <div style={s.eventMeta}>信頼度: {episode.confidence.toFixed(2)}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={s.section}>
        <h3 style={s.sectionTitle}>データソース情報</h3>
        <div style={s.metaBox}>
          <div><strong>セッションID:</strong> {report.session_id}</div>
          <div><strong>記録開始:</strong> {report.recording_startday.join(", ") || "-"}</div>
          <div><strong>利用フィールド:</strong> {report.available_fields.join(", ") || "-"}</div>
          <div><strong>ECGサンプル数:</strong> {report.ecg_points.toLocaleString()}</div>
          <div><strong>RRサンプル数:</strong> {report.rr_points.toLocaleString()}</div>
          <div><strong>QRSサンプル数:</strong> {report.qrs_points.toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
}

const s = {
  container: { display: "flex", flexDirection: "column", gap: "16px" },
  empty: { padding: "40px 0", textAlign: "center", color: "#aaa", fontSize: "14px" },
  cardRow: { display: "flex", gap: "10px", flexWrap: "wrap" },
  card: {
    minWidth: "140px",
    flex: "1 1 140px",
    background: "#fff",
    borderRadius: "10px",
    padding: "14px 16px",
    border: "1px solid #d0d7e2",
    boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  cardLabel: { fontSize: "11px", color: "#888", margin: "0 0 6px" },
  cardValue: { fontSize: "20px", fontWeight: "bold", margin: 0 },
  section: {
    background: "#fff",
    borderRadius: "12px",
    padding: "18px 20px",
    border: "1px solid #d0d7e2",
    boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  sectionTitle: {
    fontSize: "14px",
    color: "#1a1a2e",
    margin: "0 0 14px",
    fontWeight: "bold",
  },
  summaryBox: {
    background: "#f8fafc",
    border: "1px solid #e0e6ef",
    borderRadius: "10px",
    padding: "14px 16px",
    lineHeight: 1.7,
    color: "#1a1a2e",
    fontSize: "13px",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "12px",
    tableLayout: "fixed",
  },
  th: {
    padding: "10px 12px",
    color: "#888",
    borderBottom: "2px solid #e0e6ef",
    fontWeight: "bold",
    textAlign: "left",
    whiteSpace: "nowrap",
  },
  td: {
    padding: "10px 12px",
    borderBottom: "1px solid #e0e6ef",
    color: "#1a1a2e",
    whiteSpace: "nowrap",
  },
  eventList: { display: "flex", flexDirection: "column", gap: "10px" },
  eventCard: {
    background: "#fff7ed",
    border: "1px solid #fcd8b3",
    borderRadius: "10px",
    padding: "12px 14px",
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
    gap: "6px",
  },
  eventMeta: { fontSize: "12px", color: "#6b7280" },
  metaBox: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
    gap: "8px 16px",
    fontSize: "12px",
    lineHeight: 1.7,
    color: "#374151",
  },
  emptyText: { margin: 0, color: "#6b7280", fontSize: "13px" },
};
