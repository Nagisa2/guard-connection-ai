// LoginPage.jsx - ログイン画面

import { useState } from "react";
import axios from "axios";

export default function LoginPage({ onLogin }) {
  const [loginId,  setLoginId]  = useState("");
  const [password, setPassword] = useState("");
  const [error,    setError]    = useState("");
  const [loading,  setLoading]  = useState(false);

  const handleSubmit = async () => {
    if (!loginId || !password) { setError("IDとパスワードを入力してください"); return; }
    setLoading(true); setError("");
    try {
      const res = await axios.post(`${import.meta.env.VITE_API_BASE_URL}/auth/login`, {
        login_id: loginId, password,
      });
      onLogin(res.data);
    } catch {
      setError("IDまたはパスワードが正しくありません");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={s.page}>
      <div style={s.card}>
        <div style={s.titleRow}>
          <span style={{ fontSize: "40px" }}></span>
          <div>
            <h1 style={s.title}>心電図モニタリング</h1>
            <p style={s.sub}>管理ダッシュボード</p>
          </div>
        </div>

        <div style={s.form}>
          <div style={s.field}>
            <label style={s.label}>ログインID</label>
            <input type="text" value={loginId}
              onChange={e => setLoginId(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSubmit()}
              placeholder="login_id" style={s.input} />
          </div>
          <div style={s.field}>
            <label style={s.label}>パスワード</label>
            <input type="password" value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSubmit()}
              placeholder="password" style={s.input} />
          </div>
          {error && <p style={s.error}>{error}</p>}
          <button onClick={handleSubmit} disabled={loading}
            style={{ ...s.btn, opacity: loading ? 0.7 : 1 }}>
            {loading ? "認証中..." : "ログイン"}
          </button>
        </div>

        <div style={s.hint}>
          <p style={{ fontSize: "11px", color: "#888", margin: "0 0 6px", fontWeight: "bold" }}>開発用アカウント</p>
          <p style={{ fontSize: "12px", color: "#555", margin: "2px 0", fontFamily: "monospace" }}>ID: yamada / PW: password123</p>
          <p style={{ fontSize: "12px", color: "#555", margin: "2px 0", fontFamily: "monospace" }}>ID: suzuki / PW: password123</p>
        </div>
      </div>
    </div>
  );
}

const s = {
  page: {
    minHeight: "100vh", background: "#f0f4f8",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontFamily: "'Segoe UI', sans-serif",
  },
  card: {
    background: "#fff", borderRadius: "16px", padding: "48px 40px", width: "380px",
    boxShadow: "0 4px 24px rgba(0,0,0,0.10)", border: "1px solid #d0d7e2",
  },
  titleRow: { display: "flex", alignItems: "center", gap: "16px", marginBottom: "32px" },
  title:    { fontSize: "20px", fontWeight: "bold", color: "#1a1a2e", margin: 0 },
  sub:      { fontSize: "13px", color: "#888", margin: "4px 0 0" },
  form:     { display: "flex", flexDirection: "column", gap: "16px" },
  field:    { display: "flex", flexDirection: "column", gap: "6px" },
  label:    { fontSize: "13px", color: "#555", fontWeight: "bold" },
  input: {
    padding: "10px 12px", borderRadius: "8px", border: "1px solid #d0d7e2",
    fontSize: "14px", color: "#1a1a2e", background: "#f8fafc", outline: "none",
  },
  error: { fontSize: "13px", color: "#ef4444", margin: 0 },
  btn: {
    padding: "12px", borderRadius: "8px", background: "#2563eb",
    border: "none", color: "#fff", fontSize: "15px",
    fontWeight: "bold", cursor: "pointer", marginTop: "8px",
  },
  hint: {
    marginTop: "24px", padding: "12px 16px",
    background: "#f0f4f8", borderRadius: "8px", border: "1px solid #d0d7e2",
  },
};
