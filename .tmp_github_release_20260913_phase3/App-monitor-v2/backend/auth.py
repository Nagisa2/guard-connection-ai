import os
import secrets

from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

SECRET_KEY = os.getenv("GUARD_JWT_SECRET") or secrets.token_urlsafe(32)
ALGORITHM    = "HS256"  # 署名アルゴリズム
EXPIRE_HOURS = 12 # トークンの有効期限

security = HTTPBearer()

"""
    JWTトークンを生成して返す。
    ログイン成功時（main.py の login 関数）から呼ばれる。

    引数:
        doctor_id: 医師のID（トークンのペイロードに格納）
        login_id : 医師のログインID（トークンのペイロードに格納）
    戻り値:
        JWT トークン文字列（例: "eyJhbGciOiJIUzI1NiJ9...."）
"""
def create_token(doctor_id: int, login_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=EXPIRE_HOURS)
    return jwt.encode(
        {"sub": login_id, "doctor_id": doctor_id, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )

"""
    JWTトークンをデコードしてペイロード（辞書）を返す。
    トークンが無効（改ざん・期限切れ）な場合は None を返す。

    引数:
        token: JWT トークン文字列
    戻り値:
        ペイロード辞書（例: {"sub": "yamada", "doctor_id": 1, "exp": ...}）
        無効な場合は None
"""
def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

"""
    APIリクエストのBearerトークンを検証して doctor_id を返す依存関数。
    main.py の各エンドポイントで Depends(auth.get_current_doctor_id) として使う。

    【動作の流れ】
    1. HTTPBearer が "Authorization: Bearer xxxxx" ヘッダーからトークンを取り出す
    2. decode_token() でトークンを検証・デコードする
    3. 無効なら 401 Unauthorized エラーを返す
    4. 有効なら doctor_id を返す（エンドポイント関数の引数として渡される）

    引数:
        credentials: HTTPBearer が自動で取り出した認証情報
    戻り値:
        doctor_id（int）
    エラー:
        401: トークンが無効・期限切れ・ヘッダーがない場合
"""
def get_current_doctor_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証トークンが無効です",
        )
    return payload.get("doctor_id")
