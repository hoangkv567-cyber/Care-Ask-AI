# -*- coding: utf-8 -*-
"""
src/auth.py — Xác thực & phân quyền (bác sĩ / người dùng) cho HealthWatch AI Clinic.

THIẾT KẾ
--------
- Lưu tài khoản trong Neo4j (node :User) — dùng luôn DB sẵn có, KHÔNG thêm hạ tầng, và chạy
  được trên Vercel serverless (filesystem chỉ đọc nên SQLite/file JSON là không dùng được).
  Nhãn :User tách hẳn khỏi dữ liệu y khoa (:BenhLy/:HoiChung/...), không đụng KG chẩn đoán.
- Mật khẩu băm bằng bcrypt (có salt riêng từng tài khoản). TUYỆT ĐỐI không lưu mật khẩu thô.
- Phiên đăng nhập bằng JWT ký HS256 — stateless, hợp với serverless (không giữ session ở RAM).

BẢO MẬT — các điểm đã cân nhắc
------------------------------
- JWT_SECRET đọc từ biến môi trường. KHÔNG có sẵn -> sinh ngẫu nhiên mỗi lần khởi động, nghĩa là
  restart sẽ vô hiệu mọi token cũ (an toàn khi quên cấu hình, nhưng phải đặt ở production).
- Vai trò được kiểm tra ở BACKEND (require_doctor), không chỉ ẩn/hiện ở giao diện — chặn ở UI
  thôi thì ai cũng gọi thẳng API được.
- Tự đăng ký CHỈ tạo được vai 'user'. Tài khoản 'doctor' phải do người quản trị tạo
  (scripts/seed_users.py) — tránh ai cũng tự nhận là bác sĩ để mở khoá tính năng chuyên môn.
- So sánh mật khẩu luôn chạy bcrypt kể cả khi username không tồn tại (chống dò tài khoản qua
  chênh lệch thời gian phản hồi).
"""
import os
import re
import logging
import secrets
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

ROLE_DOCTOR = "doctor"
ROLE_USER = "user"
VALID_ROLES = (ROLE_DOCTOR, ROLE_USER)

_ALGORITHM = "HS256"
_TOKEN_TTL_HOURS = int(os.getenv("JWT_TTL_HOURS", "12"))

# Băm giả dùng khi username không tồn tại -> vẫn tốn đúng một lần bcrypt như ca hợp lệ.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-timing", bcrypt.gensalt())

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{3,32}$")
MIN_PASSWORD_LEN = 8

_bearer = HTTPBearer(auto_error=False)


def _secret() -> str:
    s = os.getenv("JWT_SECRET", "").strip()
    if not s:
        # Chỉ cảnh báo MỘT lần; token sẽ mất hiệu lực sau mỗi lần khởi động lại.
        global _EPHEMERAL_SECRET
        try:
            return _EPHEMERAL_SECRET
        except NameError:
            _EPHEMERAL_SECRET = secrets.token_urlsafe(48)
            logger.warning(
                "JWT_SECRET chưa được đặt — dùng khoá tạm sinh ngẫu nhiên. Mọi phiên đăng nhập sẽ "
                "mất hiệu lực khi khởi động lại. HÃY đặt JWT_SECRET trong .env trước khi chạy thật."
            )
            return _EPHEMERAL_SECRET
    return s


# ----------------------------------------------------------------------------- mật khẩu
def hash_password(raw: str) -> str:
    return bcrypt.hashpw(raw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(raw: str, hashed: Optional[str]) -> bool:
    """So khớp mật khẩu. hashed=None (không có tài khoản) vẫn chạy bcrypt để giữ thời gian
    phản hồi đồng đều — không lộ việc username có tồn tại hay không."""
    target = (hashed or "").encode("utf-8") or _DUMMY_HASH
    try:
        ok = bcrypt.checkpw(raw.encode("utf-8"), target)
    except (ValueError, TypeError):
        # hash hỏng/định dạng lạ -> vẫn tiêu tốn một lần bcrypt rồi trả False
        bcrypt.checkpw(raw.encode("utf-8"), _DUMMY_HASH)
        return False
    return bool(ok) and hashed is not None


def validate_credentials(username: str, password: str) -> Optional[str]:
    """Trả về thông báo lỗi (tiếng Việt) nếu không hợp lệ, None nếu OK."""
    if not _USERNAME_RE.match(username or ""):
        return ("Tên đăng nhập phải dài 3–32 ký tự, chỉ gồm chữ, số và các ký tự . _ -")
    if len(password or "") < MIN_PASSWORD_LEN:
        return f"Mật khẩu phải dài ít nhất {MIN_PASSWORD_LEN} ký tự."
    return None


# ----------------------------------------------------------------------------- JWT
def create_token(username: str, role: str, full_name: str = "") -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "name": full_name or username,
        "iat": now,
        "exp": now + timedelta(hours=_TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, _secret(), algorithm=_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, _secret(), algorithms=[_ALGORITHM])


# ----------------------------------------------------------------------------- kho tài khoản (Neo4j)
class UserStore:
    """Kho tài khoản trên Neo4j. Nhận sẵn driver để tái dùng kết nối của pipeline."""

    def __init__(self, driver):
        self.driver = driver

    def ensure_schema(self):
        """Ràng buộc DUY NHẤT trên username — chặn tạo trùng ở mức DB (không chỉ ở tầng ứng dụng,
        vì hai request song song có thể cùng vượt qua bước kiểm tra 'đã tồn tại chưa')."""
        try:
            with self.driver.session() as s:
                s.run("CREATE CONSTRAINT user_username_unique IF NOT EXISTS "
                      "FOR (u:User) REQUIRE u.username IS UNIQUE")
        except Exception as e:
            logger.warning(f"Không tạo được ràng buộc UNIQUE cho :User — {e}")

    def get(self, username: str) -> Optional[dict]:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (u:User {username: $u}) "
                "RETURN u.username AS username, u.password_hash AS password_hash, "
                "       u.role AS role, u.full_name AS full_name, u.active AS active",
                u=(username or "").strip().lower(),
            ).single()
        return dict(rec) if rec else None

    def create(self, username: str, password: str, role: str = ROLE_USER,
               full_name: str = "") -> dict:
        if role not in VALID_ROLES:
            raise ValueError(f"Vai trò không hợp lệ: {role}")
        uname = (username or "").strip().lower()
        with self.driver.session() as s:
            rec = s.run(
                """
                MERGE (u:User {username: $u})
                ON CREATE SET u.password_hash = $ph, u.role = $role, u.full_name = $fn,
                              u.active = true, u.created_at = datetime(), u._new = true
                ON MATCH  SET u._new = false
                WITH u, u._new AS is_new
                REMOVE u._new
                RETURN is_new AS is_new, u.username AS username, u.role AS role,
                       u.full_name AS full_name
                """,
                u=uname, ph=hash_password(password), role=role, fn=full_name or uname,
            ).single()
        if not rec or not rec["is_new"]:
            raise ValueError("Tên đăng nhập đã tồn tại.")
        return {"username": rec["username"], "role": rec["role"], "full_name": rec["full_name"]}

    def set_password(self, username: str, new_password: str) -> bool:
        with self.driver.session() as s:
            rec = s.run("MATCH (u:User {username: $u}) SET u.password_hash = $ph RETURN u.username AS n",
                        u=(username or "").strip().lower(), ph=hash_password(new_password)).single()
        return rec is not None

    def list_users(self) -> list:
        with self.driver.session() as s:
            return [dict(r) for r in s.run(
                "MATCH (u:User) RETURN u.username AS username, u.role AS role, "
                "u.full_name AS full_name, u.active AS active ORDER BY u.role, u.username")]

    def create_firebase_user(self, email: str, role: str, full_name: str = "") -> dict:
        if role not in VALID_ROLES:
            raise ValueError(f"Vai trò không hợp lệ: {role}")
        uname = (email or "").strip().lower()
        with self.driver.session() as s:
            rec = s.run(
                """
                MERGE (u:User {username: $u})
                ON CREATE SET u.password_hash = "firebase", u.role = $role, u.full_name = $fn,
                              u.active = true, u.created_at = datetime(), u._new = true
                ON MATCH  SET u.full_name = $fn, u.role = $role, u._new = false
                WITH u, u._new AS is_new
                REMOVE u._new
                RETURN is_new AS is_new, u.username AS username, u.role AS role,
                       u.full_name AS full_name
                """,
                u=uname, role=role, fn=full_name or uname,
            ).single()
        return {"username": rec["username"], "role": rec["role"], "full_name": rec["full_name"]}


# ----------------------------------------------------------------------------- dependency FastAPI
_store: Optional[UserStore] = None


def init_auth(driver):
    """Gắn driver Neo4j cho tầng auth (gọi một lần lúc khởi động API)."""
    global _store
    _store = UserStore(driver)
    _store.ensure_schema()
    return _store


def get_store() -> UserStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Hệ thống xác thực chưa sẵn sàng.")
    return _store


def _unauthorized(msg="Bạn cần đăng nhập để sử dụng chức năng này."):
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg,
                         headers={"WWW-Authenticate": "Bearer"})


async def get_current_user(
    cred: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Giải mã JWT -> thông tin người dùng. Ném 401 nếu thiếu/hỏng/hết hạn."""
    if cred is None or not cred.credentials:
        raise _unauthorized()
    try:
        payload = decode_token(cred.credentials)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.")
    except jwt.PyJWTError:
        raise _unauthorized("Phiên đăng nhập không hợp lệ.")
    username, role = payload.get("sub"), payload.get("role")
    if not username or role not in VALID_ROLES:
        raise _unauthorized("Phiên đăng nhập không hợp lệ.")
    # Đối chiếu lại với DB: tài khoản có thể đã bị khoá/xoá SAU khi token được cấp.
    rec = get_store().get(username)
    if not rec or rec.get("active") is False:
        raise _unauthorized("Tài khoản không còn hiệu lực.")
    return {"username": username, "role": rec.get("role") or role,
            "full_name": rec.get("full_name") or username}


async def require_doctor(user: dict = Depends(get_current_user)) -> dict:
    """Chỉ cho vai BÁC SĨ. Dùng cho tính năng chuyên môn (hỏi-đáp KG, chi tiết kỹ thuật)."""
    if user.get("role") != ROLE_DOCTOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chức năng này dành riêng cho tài khoản bác sĩ.",
        )
    return user


_FIREBASE_CERTS_URL = ("https://www.googleapis.com/robot/v1/metadata/x509/"
                       "securetoken@system.gserviceaccount.com")


def verify_firebase_token(id_token: str) -> dict:
    """Giải mã và kiểm chữ ký ID Token của Firebase. Ném ValueError nếu KHÔNG kiểm được.

    LỖI ĐÃ CÓ Ở ĐÂY — ghi lại để không ai vô tình dựng lại:
    Bản trước bọc bước kiểm chữ ký trong try/except, và nhánh except lại `return jwt.decode(
    ..., verify_signature=False)`. Nghĩa là chữ ký SAI thì token vẫn được nhận — kiểm chữ ký chỉ
    còn tính trang trí. Ghép với việc vai trò suy thẳng từ trường email trong token (api.py), ai
    cũng tự ký được một token mang email của bác sĩ để lấy quyền bác sĩ, và quyền đó mở ra cả
    /api/ask lẫn TOÀN BỘ lịch sử chẩn đoán của mọi người dùng.

    Đo được lúc vá, tệ hơn cả suy đoán ban đầu: gói `cryptography` KHÔNG được cài, nên PyJWT chỉ
    có ['HS256','HS384','HS512','none'] — RS256 CHƯA TỪNG khả dụng. Mọi lượt gọi đều ném
    InvalidAlgorithmError rồi rơi vào except. Xác minh chữ ký chưa chạy đúng một lần nào, kể cả
    khi FIREBASE_PROJECT_ID đã đặt. Vì vậy bản vá này BẮT BUỘC đi kèm `cryptography` trong
    requirements.txt — thiếu nó thì mọi đăng nhập Firebase sẽ bị từ chối (thà hỏng còn hơn nhận bừa).

    Cạm bẫy thứ hai đã đo: PyJWT 2.13 KHÔNG nhận chuỗi cert X.509 làm khóa (InvalidKeyError) —
    phải bóc public key ra khỏi cert trước. Truyền thẳng cert_pem như bản cũ là luôn ném lỗi.
    """
    try:
        header = jwt.get_unverified_header(id_token)
        kid = header.get("kid")
        if not kid:
            raise ValueError("Token Firebase không hợp lệ (thiếu kid).")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Không thể đọc header của Token: {e}")

    proj_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    if not proj_id:
        # Đường dev bỏ qua chữ ký phải được BẬT TƯỜNG MINH. Trước đây chỉ cần biến môi trường
        # TRỐNG là tự động bỏ kiểm — tức quên cấu hình khi triển khai = mở toang xác thực.
        if os.getenv("FIREBASE_ALLOW_INSECURE_DEV", "").strip().lower() in ("1", "true", "yes"):
            logger.warning("FIREBASE_ALLOW_INSECURE_DEV đang BẬT — token KHÔNG được kiểm chữ ký. "
                           "Chỉ dùng cho máy dev, TUYỆT ĐỐI không bật khi triển khai.")
            return jwt.decode(id_token, options={"verify_signature": False})
        raise ValueError("Chưa cấu hình FIREBASE_PROJECT_ID nên không thể kiểm chữ ký token.")

    try:
        res = httpx.get(_FIREBASE_CERTS_URL, timeout=10)
        res.raise_for_status()
        certs = res.json()
    except Exception as e:
        # KHÔNG nhận bừa khi không lấy được chứng chỉ: mạng hỏng là lý do để TỪ CHỐI, không phải
        # để bỏ qua xác thực.
        raise ValueError(f"Không lấy được chứng chỉ Firebase để kiểm chữ ký: {e}")

    if kid not in certs:
        raise ValueError("Không tìm thấy chứng chỉ tương ứng với kid từ Google.")

    try:
        # PyJWT không nhận cert X.509 trực tiếp -> bóc public key (đã đo: truyền thẳng cert ném
        # InvalidKeyError, và lỗi đó chính là thứ rơi vào except ở bản cũ).
        from cryptography.x509 import load_pem_x509_certificate
        pub_key = load_pem_x509_certificate(certs[kid].encode()).public_key()
        return jwt.decode(
            id_token,
            pub_key,
            algorithms=["RS256"],
            audience=proj_id,
            issuer=f"https://securetoken.google.com/{proj_id}",
        )
    except ImportError as e:
        raise ValueError(f"Thiếu gói 'cryptography' nên không kiểm được chữ ký RS256: {e}")
    except Exception as e:
        # TỪ CHỐI. Đây chính là chỗ bản cũ nhận bừa token.
        logger.warning(f"Từ chối token Firebase — kiểm chữ ký thất bại: {e}")
        raise ValueError(f"Token Firebase không hợp lệ: {e}")
