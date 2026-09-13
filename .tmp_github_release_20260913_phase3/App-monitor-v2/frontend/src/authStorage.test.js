import assert from "node:assert/strict";
import test from "node:test";

import { clearAuth, loadAuth, saveAuth } from "./authStorage.js";

function storage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

test("認証情報をタブ間で共有できるlocalStorageへ保存する", () => {
  const local = storage();
  const session = storage();
  saveAuth({ token: "token-1", doctor_id: 1 }, local, session);
  assert.deepEqual(loadAuth(local, session), { token: "token-1", doctor_id: 1 });
});

test("旧sessionStorageの認証情報をlocalStorageへ移行する", () => {
  const local = storage();
  const session = storage({ hm_auth: JSON.stringify({ token: "legacy" }) });
  assert.deepEqual(loadAuth(local, session), { token: "legacy" });
  assert.equal(session.getItem("hm_auth"), null);
  assert.notEqual(local.getItem("hm_auth"), null);
});

test("ログアウト時は両方の保存領域を消去する", () => {
  const local = storage({ hm_auth: "local" });
  const session = storage({ hm_auth: "session" });
  clearAuth(local, session);
  assert.equal(local.getItem("hm_auth"), null);
  assert.equal(session.getItem("hm_auth"), null);
});
