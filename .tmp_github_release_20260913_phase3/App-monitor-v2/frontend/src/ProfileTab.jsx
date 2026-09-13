// ProfileTab.jsx - 患者プロフィールタブ

import { useState, useEffect } from "react";
import axios from "axios";

const YEARS  = Array.from({ length: 100 }, (_, i) => String(new Date().getFullYear() - i));
const MONTHS = Array.from({ length: 12  }, (_, i) => String(i + 1).padStart(2, "0"));

// 月と年に応じた正しい日数を返す
function getDaysInMonth(year, month) {
  if (!year || !month) return 31;
  return new Date(Number(year), Number(month), 0).getDate();
}

function Select({ value, onChange, options, placeholder, style }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)}
      style={{ ...s.input, cursor: "pointer", ...style }}>
      <option value="">{placeholder ?? "選択"}</option>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function calcAge(birthDateStr) {
  if (!birthDateStr) return null;
  const birth = new Date(birthDateStr);
  if (isNaN(birth)) return null;
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const m = today.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
  return age;
}

function parseBirthDate(str) {
  if (!str) return { y: "", m: "", d: "" };
  const [y, m, d] = str.split("-");
  return { y: y ?? "", m: m ?? "", d: d ?? "" };
}

export default function ProfileTab({ patient, token, onUpdated, onDeleted }) {
  const [editing,     setEditing]     = useState(false);
  const [draft,       setDraft]       = useState({});
  const [byYear,      setByYear]      = useState("");
  const [byMonth,     setByMonth]     = useState("");
  const [byDay,       setByDay]       = useState("");
  const [saving,      setSaving]      = useState(false);
  const [saved,       setSaved]       = useState(false);
  const [showConfirm, setShowConfirm] = useState(false); // 削除確認ダイアログ
  const [deleting,    setDeleting]    = useState(false);
  const [errors,      setErrors]      = useState({});

  // 患者切り替え時にリセット
  useEffect(() => {
    setEditing(false);
    setDraft({ ...patient });
    const { y, m, d } = parseBirthDate(patient?.birth_date);
    setByYear(y); setByMonth(m); setByDay(d);
    setErrors({});
  }, [patient?.id]);

  // 月が変わったとき、日が範囲外なら日をリセット
  useEffect(() => {
    const maxDay = getDaysInMonth(byYear, byMonth);
    if (byDay && Number(byDay) > maxDay) setByDay("");
  }, [byYear, byMonth]);

  // 選択可能な日リスト（年・月に応じて動的に変わる）
  const availableDays = Array.from(
    { length: getDaysInMonth(byYear, byMonth) },
    (_, i) => String(i + 1).padStart(2, "0")
  );

  const validate = () => {
    const errs = {};
    // 生年月日: 年・月・日のいずれかだけ入力されている場合はエラー
    const dateFields = [byYear, byMonth, byDay].filter(Boolean).length;
    if (dateFields > 0 && dateFields < 3) {
      errs.birth_date = "年・月・日をすべて選択してください";
    }
    return errs;
  };

  const handleSave = async () => {
    const errs = validate();
    if (Object.keys(errs).length > 0) { setErrors(errs); return; }

    const birthDate = (byYear && byMonth && byDay)
      ? `${byYear}-${byMonth}-${byDay}` : "";
    const payload = { ...draft, birth_date: birthDate };

    setSaving(true);
    try {
      await axios.put(
        `${import.meta.env.VITE_API_BASE_URL}/patients/${patient.id}`,
        payload,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setSaved(true);
      setEditing(false);
      setErrors({});
      onUpdated && onUpdated(payload);
      setTimeout(() => setSaved(false), 3000);
    } catch (e) { console.error("保存失敗", e); }
    finally { setSaving(false); }
  };

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await axios.delete(
        `${import.meta.env.VITE_API_BASE_URL}/patients/${patient.id}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      onDeleted && onDeleted(patient.id); // 親に削除を通知
    } catch (e) { console.error("削除失敗", e); }
    finally { setDeleting(false); setShowConfirm(false); }
  };

  const set = (key, val) => setDraft(p => ({ ...p, [key]: val }));

  return (
    <div style={s.container}>
      <div style={s.header}>
        <h3 style={s.title}>{patient.name}</h3>
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          {saved && <span style={{ fontSize: "13px", color: "#22c55e" }}>保存しました</span>}
          {!editing && (
            <>
              <button onClick={() => setEditing(true)} style={s.editBtn}>編集する</button>
              <button onClick={() => setShowConfirm(true)} style={s.deleteBtn}>患者を削除</button>
            </>
          )}
        </div>
      </div>

      <table style={s.table}>
        <tbody>
          {/* 氏名 */}
          <tr>
            <td style={s.labelCell}>氏名</td>
            <td style={s.valueCell}>
              {editing
                ? <input type="text" value={draft.name ?? ""}
                    onChange={e => set("name", e.target.value)} style={s.input} />
                : <span>{patient.name || "未入力"}</span>}
            </td>
          </tr>

          {/* 生年月日 */}
          <tr>
            <td style={s.labelCell}>生年月日</td>
            <td style={s.valueCell}>
              {editing ? (
                <div>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <Select value={byYear}  onChange={v => { setByYear(v);  }}
                      options={YEARS}  placeholder="年" style={{ width: "90px" }} />
                    <Select value={byMonth} onChange={v => { setByMonth(v); }}
                      options={MONTHS} placeholder="月" style={{ width: "72px" }} />
                    <Select value={byDay}   onChange={setByDay}
                      options={availableDays} placeholder="日" style={{ width: "72px" }} />
                  </div>
                  {errors.birth_date && (
                    <p style={s.fieldErr}>{errors.birth_date}</p>
                  )}
                </div>
              ) : (
                <span style={{ color: patient.birth_date ? "#1a1a2e" : "#aaa" }}>
                {patient.birth_date
                    ? `${patient.birth_date}（${calcAge(patient.birth_date)}歳）`
                    : "未入力"}
                </span>
              )}
            </td>
          </tr>

          {/* 性別 */}
          <tr>
            <td style={s.labelCell}>性別</td>
            <td style={s.valueCell}>
              {editing ? (
                <Select value={draft.gender ?? ""} onChange={v => set("gender", v)}
                  options={["男性", "女性", "その他"]} placeholder="選択してください"
                  style={{ width: "180px" }} />
              ) : (
                <span style={{ color: patient.gender ? "#1a1a2e" : "#aaa" }}>
                  {patient.gender || "未入力"}
                </span>
              )}
            </td>
          </tr>

          {/* 病棟 */}
          <tr>
            <td style={s.labelCell}>病棟</td>
            <td style={s.valueCell}>
              {editing
                ? <input type="text" value={draft.ward ?? ""}
                    onChange={e => set("ward", e.target.value)}
                    style={s.input} placeholder="例: A棟" />
                : <span style={{ color: patient.ward ? "#1a1a2e" : "#aaa" }}>
                    {patient.ward || "未入力"}
                  </span>}
            </td>
          </tr>

          {/* 病室番号 */}
          <tr>
            <td style={s.labelCell}>病室番号</td>
            <td style={s.valueCell}>
              {editing
                ? <input type="text" value={draft.room ?? ""}
                    onChange={e => set("room", e.target.value)}
                    style={s.input} placeholder="例: 101" />
                : <span style={{ color: patient.room ? "#1a1a2e" : "#aaa" }}>
                    {patient.room || "未入力"}
                  </span>}
            </td>
          </tr>

          {/* 既往歴 */}
          <tr>
            <td style={s.labelCell}>既往歴</td>
            <td style={s.valueCell}>
              {editing
                ? <textarea value={draft.history ?? ""}
                    onChange={e => set("history", e.target.value)}
                    style={{ ...s.input, height: "64px", resize: "vertical" }} />
                : <span style={{ color: patient.history ? "#1a1a2e" : "#aaa" }}>
                    {patient.history || "未入力"}
                  </span>}
            </td>
          </tr>

          {/* 備考 */}
          <tr>
            <td style={s.labelCell}>備考</td>
            <td style={s.valueCell}>
              {editing
                ? <textarea value={draft.note ?? ""}
                    onChange={e => set("note", e.target.value)}
                    style={{ ...s.input, height: "64px", resize: "vertical" }} />
                : <span style={{ color: patient.note ? "#1a1a2e" : "#aaa" }}>
                    {patient.note || "未入力"}
                  </span>}
            </td>
          </tr>
        </tbody>
      </table>

      {/* 編集モード時のボタン */}
      {editing && (
        <div style={{ display: "flex", gap: "12px", marginTop: "20px" }}>
          <button onClick={handleSave} disabled={saving}
            style={{ ...s.saveBtn, opacity: saving ? 0.6 : 1 }}>
            {saving ? "保存中..." : "保存する"}
          </button>
          <button onClick={() => {
            setEditing(false);
            setDraft({ ...patient });
            const { y, m, d } = parseBirthDate(patient?.birth_date);
            setByYear(y); setByMonth(m); setByDay(d);
            setErrors({});
          }} style={s.cancelBtn}>
            キャンセル
          </button>
        </div>
      )}

      {/* 削除確認ダイアログ */}
      {showConfirm && (
        <div style={s.confirmOverlay} onClick={e => e.target === e.currentTarget && setShowConfirm(false)}>
          <div style={s.confirmBox}>
            <h3 style={s.confirmTitle}>患者を削除しますか？</h3>
            <p style={s.confirmMsg}>
              <strong>{patient.name}</strong> の患者情報・計測履歴・アラートを
              すべて削除します。この操作は取り消せません。
            </p>
            <div style={{ display: "flex", gap: "12px", justifyContent: "flex-end" }}>
              <button onClick={() => setShowConfirm(false)} style={s.cancelBtn}>
                キャンセル
              </button>
              <button onClick={handleDelete} disabled={deleting}
                style={{ ...s.deleteConfirmBtn, opacity: deleting ? 0.6 : 1 }}>
                {deleting ? "削除中..." : "削除する"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const s = {
  container: {
    background: "#fff", borderRadius: "12px", padding: "24px",
    border: "1px solid #d0d7e2", boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  header:    { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" },
  title:     { fontSize: "18px", fontWeight: "bold", color: "#1a1a2e", margin: 0 },
  table:     { width: "100%", borderCollapse: "collapse", fontSize: "14px" },
  labelCell: {
    width: "110px", padding: "11px 14px", color: "#888",
    borderBottom: "1px solid #e0e6ef", verticalAlign: "top", whiteSpace: "nowrap",
  },
  valueCell: { padding: "9px 14px", borderBottom: "1px solid #e0e6ef", color: "#1a1a2e" },
  input: {
    width: "100%", padding: "8px 10px", border: "1px solid #d0d7e2", borderRadius: "6px",
    fontSize: "14px", color: "#1a1a2e", background: "#f8fafc",
    boxSizing: "border-box", outline: "none",
  },
  fieldErr:  { fontSize: "12px", color: "#ef4444", margin: "4px 0 0" },
  editBtn: {
    padding: "7px 14px", background: "#f0f4f8", border: "1px solid #d0d7e2",
    borderRadius: "8px", color: "#1a1a2e", fontSize: "13px", cursor: "pointer",
  },
  deleteBtn: {
    padding: "7px 14px", background: "#fff",
    border: "1px solid #ef4444", borderRadius: "8px",
    color: "#ef4444", fontSize: "13px", cursor: "pointer",
  },
  saveBtn: {
    padding: "10px 24px", background: "#2563eb", border: "none",
    borderRadius: "8px", color: "#fff", fontWeight: "bold", fontSize: "14px", cursor: "pointer",
  },
  cancelBtn: {
    padding: "10px 20px", background: "none", border: "1px solid #d0d7e2",
    borderRadius: "8px", color: "#888", fontSize: "14px", cursor: "pointer",
  },
  // 削除確認ダイアログ
  confirmOverlay: {
    position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1100,
  },
  confirmBox: {
    background: "#fff", borderRadius: "12px", padding: "28px 32px",
    width: "400px", boxShadow: "0 8px 32px rgba(0,0,0,0.16)",
    border: "1px solid #d0d7e2",
  },
  confirmTitle: { fontSize: "17px", fontWeight: "bold", color: "#1a1a2e", margin: "0 0 12px" },
  confirmMsg:   { fontSize: "14px", color: "#555", margin: "0 0 24px", lineHeight: "1.6" },
  deleteConfirmBtn: {
    padding: "10px 24px", background: "#ef4444", border: "none",
    borderRadius: "8px", color: "#fff", fontWeight: "bold", fontSize: "14px", cursor: "pointer",
  },
};