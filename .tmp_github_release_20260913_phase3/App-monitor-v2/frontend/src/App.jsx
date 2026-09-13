// App.jsx - ルートコンポーネント

import { useState } from "react";
import LoginPage from "./LoginPage";
import Dashboard from "./Dashboard";
import CareDemoPortal from "./CareDemoPortal";
import DemoControlPortal from "./DemoControlPortal";
import { parseCarePortalLocation } from "./carePortalProtocol.js";
import { parseDemoControlLocation } from "./demoControlProtocol.js";
import { clearAuth, loadAuth, saveAuth } from "./authStorage.js";

export default function App() {
  const carePortal = parseCarePortalLocation(window.location.search);
  const demoControl = parseDemoControlLocation(window.location.search);
  const [auth, setAuth] = useState(() => loadAuth());

  const handleLogin = (authData) => {
    setAuth(authData);
    saveAuth(authData);
  };

  const handleLogout = () => {
    setAuth(null);
    clearAuth();
  };

  if (demoControl) {
    return <DemoControlPortal />;
  }
  if (carePortal) {
    return <CareDemoPortal mode={carePortal.mode} userId={carePortal.userId} />;
  }

  return auth
    ? <Dashboard auth={auth} onLogout={handleLogout} />
    : <LoginPage onLogin={handleLogin} />;
}
