// Dashboard.jsx ダッシュボード本体
/*ログイン後のメイン画面
Dashboard
- AlertBanner   : 異常検知バナー（10秒ごと更新）
- Sidebar       : 患者一覧、検索フォーム
- MainContent   : 右側のメインエリア
    - RealtimeChart : 生体情報モニタ（ECG, PPG, 心拍数）
    - NotifyArea    : 通知送信エリア
    - HistoryTab    : 履歴タブ
    - ProfileTab    : プロフィールタブ
- RegisterModal  : 患者登録モーダル
- AlertHistoryModal : アラート履歴モーダル
*/


import { useEffect, useRef, useState, useCallback } from "react";
/*
useEffect : 副作用処理(WebSocket接続、検索、未読カウントなど)
useRef    : WebSocketオブジェクトの保持、paused状態の同期
useState  : 患者データ、グラフデータ、アラート、検索条件、モーダル表示などの状態管理
useCallback : fetchPatients,fetchUnreadCountなど再生成を防ぐ関数をメモ化 
*/
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from "recharts";
import HistoryTab from "./HistoryTab";
import PseudoEcgSafetyPanel from "./PseudoEcgSafetyPanel";
import ProfileTab from "./ProfileTab";
import { createReceiveState, formatWaveformTooltipValue, receiveWaveformEnvelope, waveformDisplayName } from "./realtimeProtocol";
import { careEventStatusPriority, receiveCareEventEnvelope } from "./careEventProtocol";
import EmergencyLocationPanel from "./EmergencyLocationPanel";
import { formatPercent, receiveDownstreamInferenceEnvelope } from "./downstreamProtocol";
import {
  createSignalSourceState,
  isSupplementalSimulatedEcg,
  selectSignalSource,
  visibleTimeRange,
} from "./waveformDisplay";

// 定数定義
const MAX_POINTS = 2000;
const ECG_UPPER = 1.5;
const ECG_LOWER = -1.5;
const PPG_UPPER = 0.85;
const PPG_LOWER = 0.15;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || "ws://127.0.0.1:8000";

const YEARS = Array.from({ length: 100 }, (_, i) => String(new Date().getFullYear() - i)); // 生年選択用（過去100年分）
const MONTHS = Array.from({ length: 12 }, (_, i) => String(i + 1).padStart(2, "0")); // 月選択用（01～12）

// ステータス表示用のスタイル定義
const STATUS_STYLE = {
  normal: { label: "正常", color: "#22c55e", bg: "#f0fdf4" },
  ecg_anomaly: { label: "心電図異常", color: "#ef4444", bg: "#fef2f2" },
  ppg_anomaly: { label: "脈波異常", color: "#f59e0b", bg: "#fffbeb" },
  both_anomaly: { label: "両異常", color: "#991b1b", bg: "#fef2f2" },
  af_suspected: { label: "AF疑い・本人確認中", color: "#f97316", bg: "#fff7ed" },
  clinician_review: { label: "医師確認が必要", color: "#dc2626", bg: "#fef2f2" },
  emergency: { label: "緊急対応", color: "#991b1b", bg: "#fef2f2" },
  monitoring: { label: "経過観察", color: "#2563eb", bg: "#eff6ff" },
  resolved: { label: "対応済み", color: "#64748b", bg: "#f8fafc" },
  unknown: { label: "---", color: "#9ca3af", bg: "#f9fafb" },
};

// ユーティリティ関数

function getDaysInMonth(year, month) {  // 指定された年月の月末日を返す
  if (!year || !month) return 31;
  return new Date(Number(year), Number(month), 0).getDate();
}


function calcAge(birthDateStr) { // 生年月日から年齢を計算する関数
  if (!birthDateStr) return null;
  const birth = new Date(birthDateStr);
  if (isNaN(birth)) return null;
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const m = today.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
  return age;
}


const AnomalyDot = (props) => {     // 異常データに赤い丸を表示するカスタムドット
  const { cx, cy, payload } = props;
  if (!payload.is_anomaly) return null;
  return <circle cx={cx} cy={cy} r={5} fill="#ef4444" stroke="#fff" strokeWidth={2} />;
};

function Select({ value, onChange, options, placeholder, style }) {     // セレクトボックスの共通コンポーネント
  return (
    <select value={value} onChange={e => onChange(e.target.value)}
      style={{ ...s.formSelect, ...style }}>
      <option value="">{placeholder ?? "選択"}</option>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

// 患者登録モーダル
function RegisterModal({ token, doctorId, onClose, onRegistered }) {
  const [name, setName] = useState("");
  const [gender, setGender] = useState("");
  const [ward, setWard] = useState("");
  const [room, setRoom] = useState("");
  const [byYear, setByYear] = useState("");
  const [byMonth, setByMonth] = useState("");
  const [byDay, setByDay] = useState("");
  const [history, setHistory] = useState("");
  const [note, setNote] = useState("");
  const [nameErr, setNameErr] = useState("");
  const [saving, setSaving] = useState(false);
  const [apiError, setApiError] = useState("");

  // 月・年が変わったとき日が範囲外ならリセット
  useEffect(() => {
    const maxDay = getDaysInMonth(byYear, byMonth);
    if (byDay && Number(byDay) > maxDay) setByDay("");
  }, [byYear, byMonth]);

  // 日選択肢を動的に生成（年・月の選択に応じて変わる）
  const availableDays = Array.from(
    { length: getDaysInMonth(byYear, byMonth) },
    (_, i) => String(i + 1).padStart(2, "0")
  );

  // フォーム送信処理
  const handleSubmit = async () => {
    // 生年月日の整合性チェック
    const dateFields = [byYear, byMonth, byDay].filter(Boolean).length;
    if (dateFields > 0 && dateFields < 3) {
      setApiError("生年月日は年・月・日をすべて選択してください");
      return;
    }
    if (!name.trim()) { setNameErr("氏名は必須です"); return; }
    setSaving(true); setApiError("");
    try {
      const birthDate = (byYear && byMonth && byDay) ? `${byYear}-${byMonth}-${byDay}` : "";
      const res = await axios.post(`${API_BASE_URL}/patients`, {
        name, gender, ward, room, birth_date: birthDate, history, note,
        doctor_id: doctorId,
      }, { headers: { Authorization: `Bearer ${token}` } });
      onRegistered(res.data);
      onClose();
    } catch { setApiError("登録に失敗しました。もう一度お試しください。"); }
    finally { setSaving(false); }
  };

  return (
    // オーバーレイ背景。クリックでモーダルを閉じる
    <div style={s.overlay} onClick={e => e.target === e.currentTarget && onClose()}>
      <div style={s.modal}>
        <div style={s.modalHeader}>
          <h2 style={s.modalTitle}>患者登録</h2>
          <button onClick={onClose} style={s.closeBtn}>×</button>
        </div>
        <div style={s.modalBody}>
          <div style={s.formField}>
            <label style={s.formLabel}>氏名 <span style={{ color: "#ef4444" }}>*</span></label>
            <input type="text" value={name}
              onChange={e => { setName(e.target.value); setNameErr(""); }}
              placeholder="例: 田中 太郎"
              style={{ ...s.formInput, borderColor: nameErr ? "#ef4444" : "#d0d7e2" }} />
            {nameErr && <p style={s.fieldErr}>{nameErr}</p>}
          </div>
          <div style={s.formField}>
            <label style={s.formLabel}>生年月日</label>
            <div style={{ display: "flex", gap: "8px" }}>
              <Select value={byYear} onChange={setByYear} options={YEARS} placeholder="年" style={{ width: "90px" }} />
              <Select value={byMonth} onChange={setByMonth} options={MONTHS} placeholder="月" style={{ width: "72px" }} />
              <Select value={byDay} onChange={setByDay} options={availableDays} placeholder="日" style={{ width: "72px" }} />
            </div>
          </div>
          <div style={s.formField}>
            <label style={s.formLabel}>性別</label>
            <Select value={gender} onChange={setGender}
              options={["男性", "女性", "その他"]} placeholder="選択してください" />
          </div>
          <div style={s.formField}>
            <label style={s.formLabel}>病棟</label>
            <input type="text" value={ward} onChange={e => setWard(e.target.value)}
              placeholder="例: A棟" style={s.formInput} />
          </div>
          <div style={s.formField}>
            <label style={s.formLabel}>病室番号</label>
            <input type="text" value={room} onChange={e => setRoom(e.target.value)}
              placeholder="例: 101" style={s.formInput} />
          </div>
          <div style={s.formField}>
            <label style={s.formLabel}>既往歴</label>
            <textarea value={history} onChange={e => setHistory(e.target.value)}
              style={{ ...s.formInput, height: "64px", resize: "vertical" }} />
          </div>
          <div style={s.formField}>
            <label style={s.formLabel}>備考</label>
            <textarea value={note} onChange={e => setNote(e.target.value)}
              style={{ ...s.formInput, height: "64px", resize: "vertical" }} />
          </div>
          {apiError && <p style={s.apiError}>{apiError}</p>}
        </div>
        <div style={s.modalFooter}>
          <button onClick={onClose} style={s.cancelBtn}>キャンセル</button>
          <button onClick={handleSubmit} disabled={saving}
            style={{ ...s.submitBtn, opacity: saving ? 0.6 : 1 }}>
            {saving ? "登録中..." : "登録する"}
          </button>
        </div>
      </div>
    </div>
  );
}


// アラート履歴モーダル

function AlertHistoryModal({ token, onClose }) {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchAlerts = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/alerts`,
        { headers: { Authorization: `Bearer ${token}` } });
      setAlerts(res.data);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [token]);

  useEffect(() => { fetchAlerts(); }, []);

  const handleRead = async (id) => {
    await axios.put(`${API_BASE_URL}/alerts/${id}/read`,
      {}, { headers: { Authorization: `Bearer ${token}` } });
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, is_read: 1 } : a));
  };

  // すべて既読にする処理
  const handleReadAll = async () => {
    await axios.put(`${API_BASE_URL}/alerts/read-all`,
      {}, { headers: { Authorization: `Bearer ${token}` } });
    setAlerts(prev => prev.map(a => ({ ...a, is_read: 1 })));
  };

  const unreadCount = alerts.filter(a => !a.is_read).length;

  return (
    <div style={s.overlay} onClick={e => e.target === e.currentTarget && onClose()}>
      <div style={{ ...s.modal, width: "560px" }}>
        <div style={s.modalHeader}>
          <h2 style={s.modalTitle}>
            🔔 アラート履歴
            {unreadCount > 0 && <span style={s.unreadBadge}>{unreadCount} 件未読</span>}
          </h2>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            {unreadCount > 0 && (
              <button onClick={handleReadAll} style={s.readAllBtn}>すべて既読</button>
            )}
            <button onClick={onClose} style={s.closeBtn}>×</button>
          </div>
        </div>
        <div style={{ ...s.modalBody, gap: "8px" }}>
          {loading && <p style={{ color: "#aaa", textAlign: "center" }}>読み込み中...</p>}
          {!loading && alerts.length === 0 && (
            <p style={{ color: "#aaa", textAlign: "center" }}>アラート履歴がありません</p>
          )}
          {alerts.map(a => (
            <div key={a.id} style={{
              ...s.alertRow,
              background: a.is_read ? "#f8fafc" : "#fff5f5",
              borderLeft: `4px solid ${a.is_read ? "#d0d7e2" : "#ef4444"}`,
            }}>
              <div style={{ flex: 1 }}>
                <p style={s.alertRowPatient}>{a.is_read ? "" : "● "}{a.patient_name}</p>
                <p style={s.alertRowMsg}>{a.message}</p>
                <p style={s.alertRowTime}>{new Date(a.created_at).toLocaleString("ja-JP")}</p>
              </div>
              {!a.is_read && (
                <button onClick={() => handleRead(a.id)} style={s.readBtn}>既読</button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


// Sidebar（患者一覧＋アコーディオン検索付き）

function Sidebar({ patients, selected, onSelect, allData,
  searchName, setSearchName,
  searchBirthDate, setSearchBirthDate,
  searchRoom, setSearchRoom,
  searchWard, setSearchWard,
}) {
  const [searchOpen, setSearchOpen] = useState(false);
  const hasCondition = !!(searchName || searchBirthDate || searchRoom || searchWard);

  return (
    <aside style={s.sidebar}>
      {/* 検索トグルボタン */}
      <div style={{ padding: "8px 12px 4px" }}>
        <button
          onClick={() => setSearchOpen(p => !p)}
          style={{
            ...s.searchToggle,
            background: hasCondition ? "#ebf2ff" : "#f8fafc",
            color: hasCondition ? "#2563eb" : "#555",
            borderColor: hasCondition ? "#2563eb" : "#d0d7e2",
          }}
        >
          <span>検索 / 絞り込み</span>
          <span style={{ fontSize: "11px" }}>
            {hasCondition
              ? `(条件あり) ${searchOpen ? "▲" : "▼"}`
              : searchOpen ? "▲" : "▼"}
          </span>
        </button>
      </div>

      {/* アコーディオン展開エリア */}
      {searchOpen && (
        <div style={s.searchPanel}>
          <div style={s.searchField}>
            <label style={s.searchLabel}>患者名</label>
            <input type="text" value={searchName}
              onChange={e => setSearchName(e.target.value)}
              placeholder="例: 田中" style={s.searchInput} />
          </div>
          <div style={s.searchField}>
            <label style={s.searchLabel}>生年月日</label>
            <input type="text" value={searchBirthDate}
              onChange={e => setSearchBirthDate(e.target.value)}
              placeholder="例: 1956-04-12" style={s.searchInput} />
          </div>
          <div style={s.searchField}>
            <label style={s.searchLabel}>病室</label>
            <input type="text" value={searchRoom}
              onChange={e => setSearchRoom(e.target.value)}
              placeholder="例: 101" style={s.searchInput} />
          </div>
          <div style={s.searchField}>
            <label style={s.searchLabel}>病棟</label>
            <input type="text" value={searchWard}
              onChange={e => setSearchWard(e.target.value)}
              placeholder="例: A棟" style={s.searchInput} />
          </div>
          {hasCondition && (
            <button onClick={() => {
              setSearchName(""); setSearchBirthDate("");
              setSearchRoom(""); setSearchWard("");
            }} style={s.clearBtn}>
              条件をクリア
            </button>
          )}
        </div>
      )}

      <p style={s.sidebarTitle}>患者一覧</p>

      {[...patients].sort((left, right) => (
        careEventStatusPriority(allData[right.id]?.status) -
        careEventStatusPriority(allData[left.id]?.status)
      )).map(p => {
        const d = allData[p.id];
        const st = STATUS_STYLE[d?.status ?? "unknown"];
        const age = calcAge(p.birth_date);
        return (
          <button key={p.id} onClick={() => onSelect(p.id)} style={{
            ...s.sidebarBtn,
            background: selected === p.id ? "#e8eef7" : "transparent",
            borderLeft: selected === p.id ? "3px solid #2563eb" : "3px solid transparent",
          }}>
            <span style={s.sidebarName}>{p.name}</span>
            <div style={s.sidebarMeta}>
              <span style={s.sidebarRoom}>
                {p.ward} {p.room}{age !== null ? ` · ${age}歳` : ""}
              </span>
              <span style={{ fontSize: "11px", color: st.color, fontWeight: "bold" }}>
                {d ? `ECG: ${d.ecg} | PPG: ${d.ppg}` : "---"}
              </span>
            </div>
          </button>
        );
      })}
      {patients.length === 0 && (
        <p style={{ fontSize: "12px", color: "#aaa", padding: "16px", textAlign: "center" }}>
          該当する患者が見つかりません
        </p>
      )}
    </aside>
  );
}


// AlertBanner

function AlertBanner({ alerts, onDismiss }) {
  if (alerts.length === 0) return null;
  return (
    <div style={s.alertContainer}>
      {alerts.map(a => (
        <div key={a.id} style={{ ...s.alertItem, borderLeft: `4px solid ${a.color}` }}>
          <span style={{ fontSize: "14px", color: "#1a1a2e" }}>
            {a.icon} <strong>{a.patientName}</strong>｜{a.message}
            <span style={{ fontSize: "11px", color: "#999", marginLeft: "10px" }}>{a.time}</span>
          </span>
          <button onClick={() => onDismiss(a.id)} style={s.closeBtn}>×</button>
        </div>
      ))}
    </div>
  );
}


// 共通のグラフ描画コンポーネント（白背景・クリーンなスタイル）
function MonitorChart({ title, tooltipName = title, valueDecimals = 3, data, color, domain, height = 150, isBeat = false, lineType = "linear", timeRange = null, windowSeconds = 5, unit = "", syncId }) {
  const targetCount = isBeat ? 100 : data.length;
  const rawData = isBeat ? data.slice(-targetCount) : data;
  const seqBase = rawData.length ? rawData[0].seq : 0;
  const usesTimestamps = rawData.some(point => Number.isFinite(point.timestampSeconds));
  const resolvedRange = usesTimestamps
    ? (timeRange || visibleTimeRange([rawData], windowSeconds))
    : null;
  const rangeStart = resolvedRange
    ? resolvedRange.endSeconds - resolvedRange.durationSeconds
    : null;
  const chartData = rawData
    .filter(point => !resolvedRange || (
      point.timestampSeconds >= rangeStart && point.timestampSeconds <= resolvedRange.endSeconds
    ))
    .map(point => ({
      ...point,
      chartTime: resolvedRange
        ? point.timestampSeconds - resolvedRange.endSeconds
        : (point.seq - seqBase) / 100,
    }));
  const xDomain = resolvedRange
    ? [-resolvedRange.durationSeconds, 0]
    : [0, "dataMax"];
  const latestValue = [...chartData].reverse().find(point => point.value != null)?.value;
  return (
    <div style={{ background: "#fff", padding: "12px 14px 8px", borderRadius: "12px", marginBottom: "12px", border: "1px solid #dbe3ee", borderLeft: `4px solid ${color}`, boxShadow: "0 4px 14px rgba(15,23,42,0.05)" }}>
      <div style={{ margin: "0 4px 5px", display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h4 style={{ margin: 0, fontSize: "13px", color: "#27364a", fontWeight: 700 }}>{title}</h4>
        <span style={{ color, fontSize: "16px", fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
          {latestValue == null ? "---" : formatWaveformTooltipValue(latestValue, valueDecimals)}{latestValue == null ? "" : ` ${unit}`}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} syncId={syncId} margin={{ top: 5, right: 14, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="#e7edf5" strokeWidth={1} vertical />
          <XAxis dataKey="chartTime" type="number" tick={{ fill: "#7b8798", fontSize: 10 }} tickFormatter={(value) => Math.abs(value) < 0.001 ? "現在" : `${value.toFixed(1)}s`} interval="preserveStartEnd" domain={xDomain} allowDataOverflow stroke="#cbd5e1" />
          <YAxis domain={domain} tick={{ fill: "#7b8798", fontSize: 10 }} tickFormatter={(value) => Number(value).toFixed(valueDecimals)} stroke="#cbd5e1" width={58} />
          <Tooltip
            contentStyle={{ background: "rgba(255,255,255,0.98)", border: "1px solid #cbd5e1", borderRadius: "8px", color: "#27364a", boxShadow: "0 6px 18px rgba(15,23,42,0.12)" }}
            itemStyle={{ color: color, fontWeight: "bold" }}
            labelStyle={{ color: "#64748b" }}
            labelFormatter={(label) => Math.abs(Number(label)) < 0.001 ? "現在" : `現在から ${Math.abs(Number(label)).toFixed(2)} 秒前`}
            formatter={(val) => [`${formatWaveformTooltipValue(val, valueDecimals)}${unit ? ` ${unit}` : ""}`, tooltipName]}
            cursor={{ stroke: color, strokeWidth: 1, strokeDasharray: "3 3" }}
            filterNull
            wrapperStyle={{ pointerEvents: "none" }}
          />
          <Line type={lineType} dataKey="value" stroke={color} strokeWidth={2.2} dot={false} activeDot={{ r: 4, strokeWidth: 2, fill: "#fff" }} connectNulls={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function MonitoringSummaryPanel({ summary, inference }) {
  if (!summary) return null;
  const styles = {
    monitoring: { label: "解析中", color: "#2563eb", background: "#eff6ff" },
    af_suspected: { label: "AF疑い", color: "#ea580c", background: "#fff7ed" },
    undecidable: { label: "判定不能", color: "#64748b", background: "#f1f5f9" },
  };
  const state = styles[summary.analysis_state] || styles.undecidable;
  return (
    <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap", padding: "10px 12px", marginBottom: "12px", border: `1px solid ${state.color}44`, borderRadius: "10px", background: state.background }}>
      <strong style={{ color: state.color }}>{state.label}</strong>
      <span style={{ fontSize: "12px", color: "#475569" }}>判定可能時間 {formatPercent(summary.valid_ratio)}</span>
      <span style={{ fontSize: "12px", color: "#475569" }}>解析 {summary.observed_seconds.toFixed(0)}秒</span>
      {summary.decidable && Number.isFinite(inference?.af_probability) && (
        <span style={{ fontSize: "12px", color: "#475569" }}>
          {inference.probability_is_calibrated ? "AF確率" : "AFモデルスコア"} {formatPercent(inference.af_probability)}
        </span>
      )}
      {Number.isFinite(inference?.context?.sqi_window) && (
        <span style={{ fontSize: "12px", color: "#475569" }}>
          SQI {formatPercent(inference.context.sqi_window)}
        </span>
      )}
      {summary.analysis_state === "undecidable" && (
        <span style={{ fontSize: "12px", color: "#475569" }}>
          理由: {summary.last_abstention_reason}（連続 {summary.consecutive_undecidable_seconds.toFixed(0)}秒）
        </span>
      )}
      <span style={{ marginLeft: "auto", fontSize: "11px", color: "#64748b" }}>
        {inference?.inference_mode === "demo_stub" ? "デモ用判定" : "スクリーニング結果"}・確定診断ではありません
      </span>
    </div>
  );
}

// RealtimeChart（リアルタイム生体情報モニタ: ECG, PPG, 心拍数グラフ）

function RealtimeChart({ graphData, paused, setPaused, zoom, setZoom, visibleCharts }) {
  const displayEcg = graphData?.ecgData || [];
  const displayPpg = graphData?.ppgData || [];
  const displayPseudoEcg = graphData?.pseudoEcgData || [];
  const displayHr = graphData?.hrData || [];
  const frame = graphData?.realtimeFrame;
  const synchronizedRange = visibleTimeRange(
    [displayPpg, displayPseudoEcg].filter(series => series.length > 0),
    zoom,
  );
  return (
    <div style={s.chartBox}>
      <div style={s.chartHeader}>
        <p style={s.chartTitle}>リアルタイム モニタリング</p>
        <div style={s.chartControls}>
          <button onClick={() => setPaused(p => !p)} style={s.ctrlBtn}>
            {paused ? "▶ 再生" : "⏸ 一時停止"}
          </button>
          <span style={s.ctrlLabel}>表示範囲:</span>
          {[2, 5, 10].map(v => (
            <button key={v} onClick={() => setZoom(v)} style={{
              ...s.ctrlBtn,
              background: zoom === v ? "#2563eb" : "#f0f4f8",
              color: zoom === v ? "#fff" : "#555",
            }}>{v}秒</button>
          ))}
        </div>
      </div>
      {paused && (
        <div style={s.pausedBanner}>一時停止中 — グラフの更新を停止しています</div>
      )}

      <PseudoEcgSafetyPanel frame={frame} protocolIssue={graphData?.protocolIssue} />
      {graphData?.sourceConflict && (
        <div style={{ ...s.pausedBanner, color: "#b45309" }}>
          複数の送信元を検出しました。先に受信中のセッションだけを表示しています。
        </div>
      )}
      <MonitoringSummaryPanel
        summary={graphData?.downstreamSummary}
        inference={graphData?.downstreamInference}
      />

      <div style={{ background: "#f4f7fb", padding: "12px", borderRadius: "12px", border: "1px solid #dbe3ee" }}>
        {visibleCharts.ecg_full && <MonitorChart title={graphData?.ecgLabel || "ECG未接続"} tooltipName="シミュレーションECG" valueDecimals={3} data={displayEcg} color="#2563eb" domain={[-1.0, 1.5]} windowSeconds={zoom} unit="mV" />}
        {visibleCharts.ecg_beat && <MonitorChart title={`${graphData?.ecgLabel || "ECG未接続"} 1拍`} tooltipName="ECG" valueDecimals={3} data={displayEcg} color="#2563eb" domain={[-1.0, 1.5]} isBeat height={110} unit="mV" />}
        {visibleCharts.ppg_full && <MonitorChart title="PPG入力" tooltipName="PPG" valueDecimals={3} data={displayPpg} color="#059669" domain={['auto', 'auto']} timeRange={synchronizedRange} windowSeconds={zoom} syncId="ppg-pseudo" />}
        {visibleCharts.ppg_beat && <MonitorChart title="PPG 1拍" tooltipName="PPG" valueDecimals={3} data={displayPpg} color="#059669" domain={['auto', 'auto']} isBeat height={110} />}
        {visibleCharts.pseudo_ecg && <MonitorChart title="PPG由来疑似ECG・診断用ではない" tooltipName="疑似ECG" valueDecimals={3} data={displayPseudoEcg} color="#7c3aed" domain={[-0.5, 1.2]} timeRange={synchronizedRange} windowSeconds={zoom} unit="a.u." syncId="ppg-pseudo" />}
        {visibleCharts.hr_full && <MonitorChart title="心拍数" tooltipName="心拍数" valueDecimals={1} data={displayHr} color="#d97706" domain={['auto', 'auto']} height={125} windowSeconds={zoom} unit="bpm" />}
      </div>
    </div>
  );
}


// NotifyArea（通知送信エリア）

function NotifyArea({ patient, token }) {
  const [msg, setMsg] = useState("");
  const [sent, setSent] = useState(false);
  const [sending, setSending] = useState(false);

  const handleSend = async () => {
    if (!msg.trim()) return;
    setSending(true);
    try {
      await axios.post(`${API_BASE_URL}/alerts`,
        { patient_id: patient.id, doctor_id: patient.doctor_id, alert_type: "manual", message: msg },
        { headers: { Authorization: `Bearer ${token}` } });
      setSent(true); setMsg("");
      setTimeout(() => setSent(false), 3000);
    } catch (e) { console.error(e); }
    finally { setSending(false); }
  };

  return (
    <div style={s.notifyBox}>
      <p style={s.notifyTitle}>担当医へ通知を送信</p>
      <p style={{ fontSize: "12px", color: "#888", margin: "0 0 10px" }}>
        ※ 将来的には利用者アプリへのプッシュ通知に連携予定
      </p>
      <textarea value={msg} onChange={e => setMsg(e.target.value)}
        placeholder={`例: ${patient.name} の心拍数に異常が見られます。至急確認をお願いします。`}
        style={{ ...s.formInput, height: "72px", resize: "vertical", width: "100%", boxSizing: "border-box" }}
        rows={3} />
      <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: "12px", marginTop: "10px" }}>
        {sent && <span style={{ fontSize: "13px", color: "#22c55e" }}>通知を送信しました</span>}
        <button onClick={handleSend} disabled={sending || !msg.trim()}
          style={{ ...s.submitBtn, opacity: (sending || !msg.trim()) ? 0.5 : 1 }}>
          {sending ? "送信中..." : "送信する"}
        </button>
      </div>
    </div>
  );
}


// MainContent(右側のメインエリア)

function MainContent({ patient, liveData, graphData, token, paused, setPaused, zoom, setZoom, onPatientUpdated, onPatientDeleted }) {
  const [tab, setTab] = useState("realtime");
  // 表示するグラフの選択状態
  const [visibleCharts, setVisibleCharts] = useState({
    ecg_full: true, ecg_beat: false, ppg_full: true, ppg_beat: false, pseudo_ecg: true, hr_full: true
  });

  const TABS = [
    { key: "realtime", label: "リアルタイム" },
    { key: "history", label: "履歴" },
    { key: "profile", label: "プロフィール" },
  ];

  const st = STATUS_STYLE[liveData?.status ?? "unknown"];
  const age = calcAge(patient.birth_date);

  const toggleChart = (key) => setVisibleCharts(prev => ({ ...prev, [key]: !prev[key] }));

  return (
    <main style={s.main}>
      <div style={s.mainHeader}>
        <div>
          <h2 style={s.patientName}>{patient.name}</h2>
          <span style={s.patientSub}>
            {patient.ward} {patient.room}
            {patient.birth_date && ` / ${patient.birth_date} 生${age !== null ? `（${age}歳）` : ""}`}
            {patient.gender && ` / ${patient.gender}`}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {["recorded_demo_replay", "synthetic_demo"].includes(liveData?.sourceMode) && (
            <span style={{ ...s.stBadge, background: "#fff7ed", color: "#c2410c", border: "1px solid #fb923c" }}>
              {liveData?.sourceMode === "synthetic_demo" ? "デモ用合成信号" : "デモ用記録済み信号"}
            </span>
          )}
          <span style={{ ...s.stBadge, background: st.bg, color: st.color, border: `1px solid ${st.color}` }}>
            {st.label}
          </span>
          <span style={{ ...s.connBadge, background: liveData ? "#22c55e" : "#ef4444" }}>
            {liveData ? "接続中" : "未接続"}
          </span>
        </div>
      </div>

      <div style={s.cardRow}>
        {[
          { label: "現在の心拍数", value: liveData?.hr ? `${liveData.hr} bpm` : "---", color: "#06b6d4" },
          { label: liveData?.ecgLabel || "現在のECG（種別未検証）", value: liveData?.ecg ? `${liveData.ecg} mV` : "---", color: st.color },
          { label: "現在のPPG", value: liveData ? `${liveData.ppg}` : "---", color: st.color },
          { label: "患者ID", value: `# ${patient.id}` },
          { label: "担当病棟", value: `${patient.ward || "---"} ${patient.room || ""}` },
        ].map(c => (
          <div key={c.label} style={s.card}>
            <p style={s.cardLabel}>{c.label}</p>
            <p style={{ ...s.cardValue, color: c.color ?? "#1a1a2e" }}>{c.value}</p>
          </div>
        ))}
      </div>

      <div style={s.tabRow}>
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            ...s.tabBtn,
            borderBottom: tab === t.key ? "2px solid #2563eb" : "2px solid transparent",
            color: tab === t.key ? "#2563eb" : "#888",
          }}>{t.label}</button>
        ))}
      </div>

      <div style={s.tabContent}>
        {tab === "realtime" && (
          <>
            {/* グラフ表示設定UI */}
            <div style={{ background: "#fff", padding: "12px 16px", borderRadius: "12px", border: "1px solid #d0d7e2", marginBottom: "14px", display: "flex", gap: "16px", alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ fontSize: "13px", fontWeight: "bold", color: "#555" }}>表示するグラフ:</span>
              {[
                { key: "ecg_full", label: "ECG (時系列・受信種別)" }, { key: "ecg_beat", label: "ECG (1拍・疑似ECGには不使用)" },
                { key: "ppg_full", label: "PPG (時系列)" }, { key: "ppg_beat", label: "PPG (1拍)" },
                { key: "pseudo_ecg", label: "PPG由来疑似ECG（診断用ではない）" },
                { key: "hr_full", label: "心拍数" }
              ].map(c => (
                <label key={c.key} style={{ fontSize: "13px", display: "flex", alignItems: "center", gap: "4px", cursor: "pointer" }}>
                  <input type="checkbox" checked={visibleCharts[c.key]} onChange={() => toggleChart(c.key)} />
                  {c.label}
                </label>
              ))}
            </div>

            <RealtimeChart graphData={graphData}
              paused={paused} setPaused={setPaused}
              zoom={zoom} setZoom={setZoom}
              visibleCharts={visibleCharts} />
            <EmergencyLocationPanel
              location={liveData?.careEvent?.location}
              state={liveData?.careEvent?.state}
            />
            <NotifyArea patient={patient} token={token} />
          </>
        )}
        {tab === "history" && <HistoryTab patientId={patient.id} token={token} />}
        {tab === "profile" && (
          <ProfileTab
            patient={patient} token={token}
            onUpdated={onPatientUpdated}
            onDeleted={onPatientDeleted}
          />
        )}
      </div>
    </main>
  );
}


// Dashboard（本体）

export default function Dashboard({ auth, onLogout }) {
  const [patients, setPatients] = useState([]);   // バックエンドから取得した患者一覧
  const [selectedId, setSelectedId] = useState(null); // サイドバーで選択中の患者ID
  const [allData, setAllData] = useState({});   // 全患者の最新WebSocketデータをまとめたオブジェクト
  const [graphDataMap, setGraphDataMap] = useState({});  // 全患者のグラフ描画用データ点列
  const [alerts, setAlerts] = useState([]);   // 異常検知アラートの配列
  const [searchName, setSearchName] = useState("");   // 検索条件
  const [searchBirthDate, setSearchBirthDate] = useState("");
  const [searchRoom, setSearchRoom] = useState("");
  const [searchWard, setSearchWard] = useState("");
  const [showRegister, setShowRegister] = useState(false);    // 患者登録モーダルの表示状態
  const [showAlertHist, setShowAlertHist] = useState(false);    // アラート履歴モーダルの表示状態
  const [registerSuccess, setRegisterSuccess] = useState("");   // 登録成功メッセージ
  const [paused, setPaused] = useState(false);    // グラフの一時停止状態
  const [zoom, setZoom] = useState(5);   // グラフの表示範囲（秒）
  const [unreadCount, setUnreadCount] = useState(0);    // 未読アラートの件数

  // Refの定義
  const alertIdRef = useRef(0); // アラートに一意のIDを付与するためのカウンタ
  const socketsRef = useRef({});    // WebSocket接続オブジェクトを保持する辞書
  const pausedRef = useRef(false); // pausedの最新値をコールバック内で参照するためのRef
  const realtimeReceiveStateRef = useRef({}); // 患者・セッションごとの連番状態
  const signalSourceStateRef = useRef({}); // 患者ごとに表示中の送信元を1つに固定
  const desiredPatientIdsRef = useRef(new Set());
  const reconnectTimersRef = useRef({});

  useEffect(() => { pausedRef.current = paused; }, [paused]);

  // 患者一覧の取得
  const fetchPatients = useCallback(async (name = "", birthDate = "", room = "", ward = "") => {
    try {
      const res = await axios.get(`${API_BASE_URL}/patients`, {
        params: { name, birth_date: birthDate, room, ward },
        headers: { Authorization: `Bearer ${auth.token}` },
      });
      setPatients(res.data);
      if (!selectedId && res.data.length > 0) setSelectedId(res.data[0].id);
    } catch (e) {
      if (e.response?.status === 401) {
        onLogout();
        return;
      }
      console.error(e);
    }
  }, [auth.token, selectedId, onLogout]);

  useEffect(() => { fetchPatients(); }, []);
  useEffect(() => {
    fetchPatients(searchName, searchBirthDate, searchRoom, searchWard);
  }, [searchName, searchBirthDate, searchRoom, searchWard]);

  // 未読アラート件数の取得
  const fetchUnreadCount = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/alerts`,
        { params: { unread_only: true }, headers: { Authorization: `Bearer ${auth.token}` } });
      setUnreadCount(res.data.length);
    } catch (error) {
      if (error.response?.status === 401) onLogout();
      return;
    }
  }, [auth.token, onLogout]);

  useEffect(() => {
    fetchUnreadCount();
    const t = setInterval(fetchUnreadCount, 10000);
    return () => clearInterval(t);
  }, []);

  // 患者登録成功時の処理
  const handleRegistered = (newPatient) => {
    fetchPatients(searchName, searchBirthDate, searchRoom, searchWard);
    setSelectedId(newPatient.id);
    setRegisterSuccess(`${newPatient.name} を登録しました`);
    setTimeout(() => setRegisterSuccess(""), 3000);
  };

  // WebSocket接続の管理

  useEffect(() => {
    if (patients.length === 0) return;

    const currentPatientIds = new Set(patients.map(p => p.id));
    desiredPatientIdsRef.current = currentPatientIds;

    // 新しく追加された患者のWebSocketのみ接続する
    patients.forEach(p => {
      if (socketsRef.current[p.id]) return;
      // FastAPI（バックエンド）を経由してデータを受信する
      // ここあとでIPアドレスに変更
      const uid = p.user_id || `test_user_0${p.id}`;
      const ws = new WebSocket(`${WS_BASE_URL}/ws/sensors/${uid}`);
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);

        if (data.message_type === "care_event") {
          let received;
          try {
            received = receiveCareEventEnvelope(data);
          } catch (error) {
            console.error(`care_event schema error: ${error.message}`);
            return;
          }
          const event = received.event;
          const statusStyle = STATUS_STYLE[received.status];
          const stateMessage = {
            awaiting_patient_response: `${received.inferenceLabel} ${received.probabilityLabel}・本人確認中（確定診断ではありません）`,
            monitoring: "本人応答あり・経過観察",
            clinician_review: "本人から体調不良の応答・医師確認が必要",
            emergency_escalated: "転倒・救援要請・無反応のいずれかにより緊急度上昇",
            resolved: "医師が対応済みに変更",
          }[event.state];
          setAllData(prev => ({
            ...prev,
            [p.id]: {
              ...(prev[p.id] || { ecg: "---", ppg: "---", hr: "---" }),
              status: received.status,
              sourceMode: event.source_mode,
              demoScenarioId: event.demo_scenario_id,
              careEvent: event,
            },
          }));
          setAlerts(prev => {
            const newAlert = {
              id: event.event_id,
              patientId: p.id,
              patientName: p.name,
              status: received.status,
              message: `${stateMessage} / ${received.sourceLabel}`,
              icon: event.state === "emergency_escalated" ? "🚨" : "⚠",
              color: statusStyle.color,
              time: new Date(event.updated_at_seconds * 1000).toLocaleTimeString("ja-JP"),
              timestamp: event.updated_at_seconds * 1000,
            };
            return [newAlert, ...prev.filter(item => item.id !== event.event_id)].slice(0, 5);
          });
          return;
        }

        if (data.message_type === "downstream_inference") {
          try {
            const received = receiveDownstreamInferenceEnvelope(data);
            const activeSource = signalSourceStateRef.current[p.id];
            if (activeSource?.kind === "realtime" && activeSource.sessionId !== data.inference?.session_id) {
              return;
            }
            setGraphDataMap(prev => ({
              ...prev,
              [p.id]: {
                ...(prev[p.id] || {}),
                downstreamSummary: received.summary,
                downstreamInference: received.inference,
              },
            }));
          } catch (error) {
            console.error(`downstream_inference schema error: ${error.message}`);
          }
          return;
        }

        if (data.message_type === "waveform_frame") {
          const frameSessionId = data.frame?.session_id;
          const sourceSelection = selectSignalSource(
            signalSourceStateRef.current[p.id] || createSignalSourceState(),
            { kind: "realtime", sessionId: frameSessionId, nowMs: Date.now() },
          );
          if (!sourceSelection.accepted) {
            setGraphDataMap(prev => ({
              ...prev,
              [p.id]: { ...(prev[p.id] || {}), sourceConflict: true },
            }));
            return;
          }
          signalSourceStateRef.current[p.id] = sourceSelection.state;
          let received;
          try {
            const receiveStateKey = `${p.id}:${frameSessionId}`;
            const previous = realtimeReceiveStateRef.current[receiveStateKey] || createReceiveState();
            received = receiveWaveformEnvelope(previous, data);
            if (!received.accepted) {
              setGraphDataMap(prev => ({
                ...prev,
                [p.id]: { ...(prev[p.id] || {}), protocolIssue: received.issue },
              }));
              return;
            }
            realtimeReceiveStateRef.current[receiveStateKey] = received.state;
          } catch (error) {
            setGraphDataMap(prev => ({
              ...prev,
              [p.id]: { ...(prev[p.id] || {}), protocolIssue: `schema_error: ${error.message}` },
            }));
            return;
          }

          const frame = received.graph.frame;
          if (!pausedRef.current) {
            setGraphDataMap(prev => {
              const current = prev[p.id] || { ecgData: [], ppgData: [], pseudoEcgData: [], hrData: [] };
              const base = sourceSelection.switched
                ? { ...current, ecgData: [], ppgData: [], pseudoEcgData: [], hrData: [], downstreamSummary: null }
                : current;
              const ppgSeq = base.ppgData.length ? base.ppgData[base.ppgData.length - 1].seq + 1 : 0;
              const pseudoSeq = base.pseudoEcgData.length ? base.pseudoEcgData[base.pseudoEcgData.length - 1].seq + 1 : 0;
              const hrSeq = base.hrData.length ? base.hrData[base.hrData.length - 1].seq + 1 : 0;
              const ppgData = received.graph.ppgData.map((point, index) => ({ ...point, seq: ppgSeq + index }));
              const pseudoEcgData = received.graph.pseudoEcgData.map((point, index) => ({ ...point, seq: pseudoSeq + index }));
              const hrData = frame.heart_rate_bpm == null
                ? base.hrData
                : [...base.hrData, {
                  seq: hrSeq,
                  time: frame.input_timestamp_start_seconds,
                  timestampSeconds: frame.input_timestamp_start_seconds,
                  value: frame.heart_rate_bpm,
                }].slice(-MAX_POINTS);
              return {
                ...prev,
                [p.id]: {
                  ...base,
                  ppgData: [...base.ppgData, ...ppgData].slice(-MAX_POINTS),
                  pseudoEcgData: [...base.pseudoEcgData, ...pseudoEcgData].slice(-MAX_POINTS),
                  hrData,
                  realtimeFrame: frame,
                  sourceMode: received.graph.sourceProvenance?.source_mode ?? base.sourceMode,
                  demoScenarioId: received.graph.sourceProvenance?.demo_scenario_id ?? base.demoScenarioId,
                  protocolIssue: null,
                  sourceConflict: sourceSelection.switched ? false : base.sourceConflict,
                },
              };
            });
          }
          const latestObservedPpg = [...received.graph.ppgData].reverse().find(point => point.value != null)?.value;
          setAllData(prev => ({
            ...prev,
            [p.id]: {
              ...(prev[p.id] || { ecg: "---", status: "normal" }),
              ppg: latestObservedPpg == null ? "---" : latestObservedPpg.toFixed(3),
              hr: frame.heart_rate_bpm == null ? "---" : frame.heart_rate_bpm.toFixed(1),
              ppgSqi: frame.ppg_sqi,
              coverage: frame.signal_coverage,
              abstentionReason: frame.generation_abstention_reason,
              pseudoEcgLabel: waveformDisplayName(frame.waveform_type, frame.diagnostic_ecg),
              sourceMode: received.graph.sourceProvenance?.source_mode ?? prev[p.id]?.sourceMode,
              demoScenarioId: received.graph.sourceProvenance?.demo_scenario_id ?? prev[p.id]?.demoScenarioId,
            },
          }));
          return;
        }

        // バッチデータ形式のチェック
        if (!data.signal_type || !Array.isArray(data.time_deltas) || !Array.isArray(data.values) || data.values.length === 0) return;

        const type = data.signal_type;
        const legacySessionId = data.session_id || `legacy-${uid}`;
        const sourceSelection = selectSignalSource(
          signalSourceStateRef.current[p.id] || createSignalSourceState(),
          { kind: "legacy", sessionId: legacySessionId, nowMs: Date.now() },
        );
        const supplementalEcg = isSupplementalSimulatedEcg(data);
        const acceptedAsSupplement = !sourceSelection.accepted && supplementalEcg;
        if (!sourceSelection.accepted && !acceptedAsSupplement) {
          setGraphDataMap(prev => ({
            ...prev,
            [p.id]: { ...(prev[p.id] || {}), sourceConflict: true },
          }));
          return;
        }
        if (sourceSelection.accepted) {
          signalSourceStateRef.current[p.id] = sourceSelection.state;
        }

        const baseTime = data.base_timestamp_ms;
        const values = data.values;
        const timeLabels = data.time_deltas.map(delta => {
          const t = new Date(baseTime + delta);
          return t.toLocaleTimeString("ja-JP", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) + "." + String(t.getMilliseconds()).padStart(3, "0");
        });

        const latestVal = values[values.length - 1];

        if (!pausedRef.current) {
          setGraphDataMap(prev => {
            const current = prev[p.id] || { ecgData: [], ppgData: [], pseudoEcgData: [], hrData: [] };
            const curMap = sourceSelection.switched
              ? { ...current, ecgData: [], ppgData: [], pseudoEcgData: [], hrData: [], realtimeFrame: null, downstreamSummary: null }
              : current;
            const seqBase = {
              ECG: curMap.ecgData.length ? curMap.ecgData[curMap.ecgData.length - 1].seq + 1 : 0,
              PPG: curMap.ppgData.length ? curMap.ppgData[curMap.ppgData.length - 1].seq + 1 : 0,
              HR: curMap.hrData.length ? curMap.hrData[curMap.hrData.length - 1].seq + 1 : 0,
            }[type] || 0;

            const newPoints = values.map((value, i) => ({
              seq: seqBase + i,
              time: timeLabels[i],
              timestampSeconds: (baseTime + data.time_deltas[i]) / 1000,
              value,
            }));

            let nextData = [];
            if (type === "ECG") {
              nextData = [...curMap.ecgData, ...newPoints];
              if (nextData.length > MAX_POINTS) nextData = nextData.slice(-MAX_POINTS);
              const waveformType = data.waveform_type || (data.source === "json_output/Data" ? "simulated_ecg" : "unknown");
              return { ...prev, [p.id]: { ...curMap, ecgData: nextData, ecgLabel: waveformDisplayName(waveformType, data.diagnostic_ecg === true), sourceConflict: sourceSelection.switched ? false : curMap.sourceConflict } };
            } else if (type === "PPG") {
              nextData = [...curMap.ppgData, ...newPoints];
              if (nextData.length > MAX_POINTS) nextData = nextData.slice(-MAX_POINTS);
              return { ...prev, [p.id]: { ...curMap, ppgData: nextData, sourceConflict: sourceSelection.switched ? false : curMap.sourceConflict } };
            } else if (type === "HR") {
              nextData = [...curMap.hrData, ...newPoints];
              if (nextData.length > MAX_POINTS) nextData = nextData.slice(-MAX_POINTS);
              return { ...prev, [p.id]: { ...curMap, hrData: nextData, sourceConflict: sourceSelection.switched ? false : curMap.sourceConflict } };
            }
            return prev;
          });
        }

        // 簡易的な異常判定 (アラートバナー用)
        let is_anomaly = false;
        let status = "normal";
        if (type === "ECG" && (latestVal > ECG_UPPER || latestVal < ECG_LOWER)) {
          is_anomaly = true; status = "ecg_anomaly";
        } else if (type === "PPG" && (latestVal > PPG_UPPER || latestVal < PPG_LOWER)) {
          is_anomaly = true; status = "ppg_anomaly";
        }

        setAllData(prev => {
          const cur = prev[p.id] || { ecg: "---", ppg: "---", hr: "---", status: "normal" };
          const updated = { ...cur, status: is_anomaly ? status : cur.status };
          if (type === "ECG") {
            updated.ecg = latestVal.toFixed(3);
            const waveformType = data.waveform_type || (data.source === "json_output/Data" ? "simulated_ecg" : "unknown");
            updated.ecgLabel = waveformDisplayName(waveformType, data.diagnostic_ecg === true);
          }
          if (type === "PPG") updated.ppg = latestVal.toFixed(3);
          if (type === "HR") updated.hr = latestVal.toFixed(1);
          return { ...prev, [p.id]: updated };
        });

        // 異常検知時はアラートバナーを更新する
        if (is_anomaly) {
          const st = STATUS_STYLE[status];
          const now = Date.now();
          const timeStr = new Date().toLocaleTimeString("ja-JP");

          setAlerts(prev => {
            const lastIdx = prev.findIndex(a => a.patientId === p.id);
            const last = prev[lastIdx];

            if (last && last.status === status && (now - last.timestamp) < 10000) {
              return prev;
            }

            const newAlert = {
              id: alertIdRef.current++,
              patientId: p.id,
              patientName: p.name,
              status: status,
              message: `${st.label}を検知 / 値: ${latestVal.toFixed(3)}`,
              icon: "⚠",
              color: st.color,
              time: timeStr,
              timestamp: now,
            };

            if (lastIdx !== -1) {
              const without = prev.filter((_, i) => i !== lastIdx);
              return [newAlert, ...without].slice(0, 5);
            }
            return [newAlert, ...prev].slice(0, 5);
          });
          fetchUnreadCount();
        }
      };
      ws.onclose = () => {
        if (socketsRef.current[p.id] === ws) delete socketsRef.current[p.id];
        Object.keys(realtimeReceiveStateRef.current)
          .filter(key => key.startsWith(`${p.id}:`))
          .forEach(key => delete realtimeReceiveStateRef.current[key]);
        delete signalSourceStateRef.current[p.id];
        if (desiredPatientIdsRef.current.has(p.id)) {
          clearTimeout(reconnectTimersRef.current[p.id]);
          reconnectTimersRef.current[p.id] = setTimeout(() => {
            if (desiredPatientIdsRef.current.has(p.id)) setPatients(current => [...current]);
          }, 1000);
        }
      };
      socketsRef.current[p.id] = ws;
    });

    // 検索等でリストから外れた患者のWebSocketのみ切断する
    Object.keys(socketsRef.current).forEach(id => {
      if (!currentPatientIds.has(Number(id))) {
        socketsRef.current[id].close();
        delete socketsRef.current[id];
        Object.keys(realtimeReceiveStateRef.current)
          .filter(key => key.startsWith(`${id}:`))
          .forEach(key => delete realtimeReceiveStateRef.current[key]);
        delete signalSourceStateRef.current[id];
        clearTimeout(reconnectTimersRef.current[id]);
        delete reconnectTimersRef.current[id];
      }
    });

  }, [patients]); // patientsが変わったときだけ実行。全クリーンアップはしない

  // ダッシュボード自体が閉じられた時（ログアウト時等）に全ての通信を切断する
  useEffect(() => {
    return () => {
      desiredPatientIdsRef.current = new Set();
      Object.values(reconnectTimersRef.current).forEach(timer => clearTimeout(timer));
      reconnectTimersRef.current = {};
      Object.values(socketsRef.current).forEach(ws => ws.close());
      socketsRef.current = {};
      realtimeReceiveStateRef.current = {};
      signalSourceStateRef.current = {};
    };
  }, []);

  const selectedPatient = patients.find(p => p.id === selectedId);

  // UIの定義

  return (
    <div style={s.page}>
      <header style={s.header}>
        <h1 style={s.title}>生体信号モニタリング 管理ダッシュボード</h1>
        <div style={s.headerRight}>
          {registerSuccess && (
            <span style={{ fontSize: "13px", color: "#22c55e", fontWeight: "bold" }}>{registerSuccess}</span>
          )}
          <span style={{ fontSize: "13px", color: "#555" }}>{auth.doctor_name} / {auth.department}</span>
          <button onClick={() => { setShowAlertHist(true); fetchUnreadCount(); }} style={s.alertHistBtn}>
            🔔 アラート履歴
            {unreadCount > 0 && <span style={s.unreadBadge}>{unreadCount}</span>}
          </button>
          <button onClick={() => setShowRegister(true)} style={s.registerBtn}>+ 患者登録</button>
          <button onClick={onLogout} style={s.logoutBtn}>ログアウト</button>
        </div>
      </header>

      <AlertBanner alerts={alerts} onDismiss={id => setAlerts(prev => prev.filter(a => a.id !== id))} />

      <div style={s.body}>
        <Sidebar
          patients={patients} selected={selectedId} onSelect={setSelectedId}
          allData={allData}
          searchName={searchName} setSearchName={setSearchName}
          searchBirthDate={searchBirthDate} setSearchBirthDate={setSearchBirthDate}
          searchRoom={searchRoom} setSearchRoom={setSearchRoom}
          searchWard={searchWard} setSearchWard={setSearchWard}
        />
        {selectedPatient ? (
          <MainContent
            key={selectedId}
            patient={selectedPatient}
            liveData={allData[selectedId]}
            graphData={graphDataMap[selectedId] ?? { ecgData: [], ppgData: [], hrData: [] }}
            token={auth.token}
            paused={paused} setPaused={setPaused}
            zoom={zoom} setZoom={setZoom}
            onPatientUpdated={(updated) => {
              setPatients(prev => prev.map(p => p.id === selectedId ? { ...p, ...updated } : p));
            }}
            onPatientDeleted={(deletedId) => {
              setPatients(prev => {
                const next = prev.filter(p => p.id !== deletedId);
                setSelectedId(next.length > 0 ? next[0].id : null);
                return next;
              });
            }}
          />
        ) : (
          <div style={s.noSelect}>患者を選択してください</div>
        )}
      </div>

      {showRegister && (
        <RegisterModal token={auth.token} doctorId={auth.doctor_id}
          onClose={() => setShowRegister(false)} onRegistered={handleRegistered} />
      )}
      {showAlertHist && (
        <AlertHistoryModal token={auth.token}
          onClose={() => { setShowAlertHist(false); fetchUnreadCount(); }} />
      )}
    </div>
  );
}


// スタイル

const s = {
  page: {
    minHeight: "100vh", background: "#f0f4f8",
    color: "#1a1a2e", fontFamily: "'Segoe UI', sans-serif",
    display: "flex", flexDirection: "column",
  },
  header: {
    padding: "13px 24px", background: "#fff", borderBottom: "1px solid #d0d7e2",
    display: "flex", alignItems: "center", justifyContent: "space-between",
  },
  title: { fontSize: "16px", fontWeight: "bold", color: "#1a1a2e", margin: 0 },
  headerRight: { display: "flex", alignItems: "center", gap: "10px" },
  alertHistBtn: {
    padding: "6px 14px", background: "#fff", border: "1px solid #d0d7e2",
    borderRadius: "8px", color: "#1a1a2e", fontSize: "13px", cursor: "pointer",
    display: "flex", alignItems: "center", gap: "6px",
  },
  unreadBadge: {
    background: "#ef4444", color: "#fff",
    borderRadius: "999px", fontSize: "11px", padding: "1px 6px", fontWeight: "bold",
  },
  registerBtn: {
    padding: "7px 16px", background: "#2563eb", border: "none",
    borderRadius: "8px", color: "#fff", fontSize: "13px", fontWeight: "bold", cursor: "pointer",
  },
  logoutBtn: {
    padding: "6px 14px", background: "none", border: "1px solid #d0d7e2",
    borderRadius: "6px", color: "#888", fontSize: "13px", cursor: "pointer",
  },
  alertContainer: {
    display: "flex", flexDirection: "column", gap: "6px",
    padding: "10px 24px", background: "#fff8f8", borderBottom: "1px solid #d0d7e2",
  },
  alertItem: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    background: "#fff", borderRadius: "8px", padding: "10px 14px",
    boxShadow: "0 1px 4px rgba(0,0,0,0.08)", border: "1px solid #d0d7e2",
  },
  body: { display: "flex", flex: 1 },
  sidebar: {
    width: "210px", background: "#fff", borderRight: "1px solid #d0d7e2",
    padding: "0 0 12px", flexShrink: 0,
  },
  searchToggle: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    width: "100%", padding: "8px 12px", boxSizing: "border-box",
    border: "1px solid #d0d7e2", borderRadius: "8px",
    fontSize: "12px", fontWeight: "bold", cursor: "pointer",
  },
  searchPanel: {
    padding: "8px 12px 8px",
    borderBottom: "1px solid #e0e6ef",
    marginBottom: "4px",
    display: "flex", flexDirection: "column", gap: "8px",
  },
  searchField: { display: "flex", flexDirection: "column", gap: "3px" },
  searchLabel: { fontSize: "11px", color: "#555", fontWeight: "bold" },
  searchInput: {
    width: "100%", padding: "7px 10px", border: "1px solid #d0d7e2", borderRadius: "6px",
    fontSize: "12px", color: "#1a1a2e", background: "#f8fafc", boxSizing: "border-box",
  },
  clearBtn: {
    width: "100%", padding: "6px", background: "none", border: "1px solid #d0d7e2",
    borderRadius: "6px", fontSize: "12px", color: "#2563eb", cursor: "pointer",
  },
  sidebarTitle: { fontSize: "11px", color: "#999", padding: "8px 14px 4px", marginBottom: "0" },
  sidebarBtn: {
    width: "100%", border: "none", cursor: "pointer", padding: "10px 14px",
    display: "flex", flexDirection: "column", alignItems: "flex-start",
    color: "#1a1a2e", transition: "background 0.15s",
  },
  sidebarName: { fontSize: "13px", fontWeight: "bold" },
  sidebarMeta: { display: "flex", justifyContent: "space-between", width: "100%", marginTop: "3px" },
  sidebarRoom: { fontSize: "11px", color: "#999" },
  main: { flex: 1, padding: "18px 20px", overflowY: "auto" },
  mainHeader: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "14px" },
  patientName: { fontSize: "20px", fontWeight: "bold", color: "#1a1a2e", margin: "0 0 3px" },
  patientSub: { fontSize: "12px", color: "#888" },
  stBadge: { padding: "5px 12px", borderRadius: "999px", fontSize: "13px", fontWeight: "bold" },
  connBadge: { padding: "5px 12px", borderRadius: "999px", fontSize: "12px", color: "#fff", fontWeight: "bold" },
  cardRow: { display: "flex", gap: "10px", marginBottom: "14px" },
  card: {
    flex: 1, background: "#fff", borderRadius: "10px", padding: "14px 16px",
    border: "1px solid #d0d7e2", boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  cardLabel: { fontSize: "11px", color: "#888", margin: "0 0 6px" },
  cardValue: { fontSize: "22px", fontWeight: "bold", margin: 0 },
  tabRow: { display: "flex", gap: "4px", borderBottom: "1px solid #d0d7e2" },
  tabBtn: {
    background: "none", border: "none", cursor: "pointer",
    padding: "9px 18px", fontSize: "13px", fontWeight: "bold", transition: "color 0.15s",
  },
  tabContent: { paddingTop: "14px" },
  chartBox: {
    background: "#fff", borderRadius: "12px", padding: "16px 20px", marginBottom: "14px",
    border: "1px solid #d0d7e2", boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  chartHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" },
  chartTitle: { fontSize: "14px", fontWeight: "bold", color: "#1a1a2e", margin: 0 },
  chartControls: { display: "flex", alignItems: "center", gap: "6px" },
  ctrlBtn: {
    padding: "4px 10px", background: "#f0f4f8", border: "1px solid #d0d7e2",
    borderRadius: "6px", fontSize: "12px", cursor: "pointer", color: "#555",
  },
  ctrlLabel: { fontSize: "12px", color: "#888" },
  pausedBanner: {
    fontSize: "12px", color: "#f59e0b", background: "#fffbeb",
    border: "1px solid #fde68a", borderRadius: "6px", padding: "6px 12px", marginBottom: "8px",
  },
  notifyBox: {
    background: "#fff", borderRadius: "12px", padding: "18px 20px",
    border: "1px solid #d0d7e2", boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  notifyTitle: { fontSize: "14px", fontWeight: "bold", color: "#1a1a2e", margin: "0 0 4px" },
  noSelect: {
    flex: 1, display: "flex", alignItems: "center",
    justifyContent: "center", color: "#aaa", fontSize: "16px",
  },
  overlay: {
    position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
  },
  modal: {
    background: "#fff", borderRadius: "16px", width: "480px", maxHeight: "85vh",
    display: "flex", flexDirection: "column",
    boxShadow: "0 8px 40px rgba(0,0,0,0.18)", border: "1px solid #d0d7e2",
  },
  modalHeader: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "18px 24px 14px", borderBottom: "1px solid #e0e6ef",
  },
  modalTitle: { fontSize: "17px", fontWeight: "bold", color: "#1a1a2e", margin: 0, display: "flex", alignItems: "center", gap: "10px" },
  modalBody: { padding: "18px 24px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "14px" },
  modalFooter: { padding: "14px 24px", borderTop: "1px solid #e0e6ef", display: "flex", justifyContent: "flex-end", gap: "12px" },
  closeBtn: { background: "none", border: "none", fontSize: "22px", color: "#999", cursor: "pointer", lineHeight: 1 },
  alertRow: {
    display: "flex", alignItems: "flex-start", gap: "12px",
    padding: "12px 14px", borderRadius: "8px", border: "1px solid #e0e6ef",
  },
  alertRowPatient: { fontSize: "13px", fontWeight: "bold", color: "#1a1a2e", margin: "0 0 3px" },
  alertRowMsg: { fontSize: "13px", color: "#555", margin: "0 0 3px" },
  alertRowTime: { fontSize: "11px", color: "#999", margin: 0 },
  readBtn: {
    padding: "5px 12px", background: "#f0f4f8", border: "1px solid #d0d7e2",
    borderRadius: "6px", fontSize: "12px", cursor: "pointer", whiteSpace: "nowrap",
  },
  readAllBtn: {
    padding: "5px 14px", background: "#f0f4f8", border: "1px solid #d0d7e2",
    borderRadius: "6px", fontSize: "12px", cursor: "pointer",
  },
  formField: { display: "flex", flexDirection: "column", gap: "5px" },
  formLabel: { fontSize: "13px", color: "#555", fontWeight: "bold" },
  formInput: {
    padding: "9px 12px", border: "1px solid #d0d7e2", borderRadius: "8px",
    fontSize: "14px", color: "#1a1a2e", background: "#f8fafc",
    boxSizing: "border-box", outline: "none", width: "100%",
  },
  formSelect: {
    padding: "9px 12px", border: "1px solid #d0d7e2", borderRadius: "8px",
    fontSize: "14px", color: "#1a1a2e", background: "#f8fafc",
    boxSizing: "border-box", outline: "none", width: "100%", cursor: "pointer",
  },
  fieldErr: { fontSize: "12px", color: "#ef4444", margin: 0 },
  apiError: {
    fontSize: "13px", color: "#ef4444", background: "#fef2f2",
    padding: "10px 12px", borderRadius: "8px", margin: 0,
  },
  cancelBtn: {
    padding: "10px 20px", background: "none", border: "1px solid #d0d7e2",
    borderRadius: "8px", color: "#888", fontSize: "14px", cursor: "pointer",
  },
  submitBtn: {
    padding: "10px 28px", background: "#2563eb", border: "none",
    borderRadius: "8px", color: "#fff", fontWeight: "bold", fontSize: "14px", cursor: "pointer",
  },
};
