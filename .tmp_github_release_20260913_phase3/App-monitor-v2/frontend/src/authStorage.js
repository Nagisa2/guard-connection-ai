const AUTH_STORAGE_KEY = "hm_auth";

export function loadAuth(localStore = localStorage, sessionStore = sessionStorage) {
  try {
    const saved = localStore.getItem(AUTH_STORAGE_KEY) ?? sessionStore.getItem(AUTH_STORAGE_KEY);
    if (!saved) return null;
    const auth = JSON.parse(saved);
    localStore.setItem(AUTH_STORAGE_KEY, saved);
    sessionStore.removeItem(AUTH_STORAGE_KEY);
    return auth;
  } catch {
    clearAuth(localStore, sessionStore);
    return null;
  }
}

export function saveAuth(auth, localStore = localStorage, sessionStore = sessionStorage) {
  localStore.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
  sessionStore.removeItem(AUTH_STORAGE_KEY);
}

export function clearAuth(localStore = localStorage, sessionStore = sessionStorage) {
  localStore.removeItem(AUTH_STORAGE_KEY);
  sessionStore.removeItem(AUTH_STORAGE_KEY);
}
