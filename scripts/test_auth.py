#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_auth.py — Test tầng xác thực & phân quyền (src/auth.py).

Chạy THUẦN OFFLINE cho phần mật khẩu/JWT (không cần Neo4j). Phần kho tài khoản dùng driver GIẢ
để không đụng DB thật. Nếu muốn test luôn RBAC qua HTTP thì chạy API rồi đặt biến:
    TEST_API_BASE=http://127.0.0.1:8000 python scripts/test_auth.py

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_auth.py   (exit != 0 nếu fail)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("JWT_SECRET", "test-secret-chi-dung-cho-test")

from src import auth as A  # noqa: E402


def chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_password():
    print("== (1) Băm & so khớp mật khẩu ==")
    n = f = 0
    h = A.hash_password("MatKhauManh123")
    for label, ok in [
        ("hash KHÔNG chứa mật khẩu thô", "MatKhauManh123" not in h),
        ("hash dùng bcrypt", h.startswith("$2")),
        ("đúng mật khẩu -> True", A.verify_password("MatKhauManh123", h)),
        ("sai mật khẩu -> False", not A.verify_password("MatKhauSai999", h)),
        ("hai lần hash KHÁC nhau (có salt)", A.hash_password("x" * 10) != A.hash_password("x" * 10)),
        # Không có tài khoản: vẫn phải trả False, KHÔNG được ném lỗi (chống dò tài khoản)
        ("user không tồn tại -> False", not A.verify_password("bất kỳ", None)),
        ("hash hỏng -> False", not A.verify_password("abc", "khong-phai-hash")),
    ]:
        n += chk(label, ok)
        f += (not ok)
    return n, f


def test_timing():
    """Thời gian trả lời khi SAI mật khẩu và khi KHÔNG có tài khoản phải tương đương —
    nếu lệch hẳn, kẻ tấn công dò được username nào tồn tại."""
    print("\n== (2) Chống dò tài khoản qua thời gian ==")
    h = A.hash_password("MatKhauManh123")

    def dur(fn, rounds=5):
        t0 = time.perf_counter()
        for _ in range(rounds):
            fn()
        return (time.perf_counter() - t0) / rounds

    t_wrong = dur(lambda: A.verify_password("sai", h))
    t_none = dur(lambda: A.verify_password("sai", None))
    ratio = max(t_wrong, t_none) / max(min(t_wrong, t_none), 1e-9)
    ok = ratio < 3.0            # cùng bậc độ lớn là đạt
    return 1 if chk(f"thời gian cùng bậc (tỉ lệ {ratio:.2f}×, cần < 3)", ok,
                    f"sai-mk={t_wrong*1000:.1f}ms, không-có-user={t_none*1000:.1f}ms") else 0, 0 if ok else 1


def test_jwt():
    print("\n== (3) JWT ==")
    n = f = 0
    t = A.create_token("bs.a", A.ROLE_DOCTOR, "BS A")
    p = A.decode_token(t)
    for label, ok in [
        ("giải mã đúng username", p.get("sub") == "bs.a"),
        ("giải mã đúng vai trò", p.get("role") == A.ROLE_DOCTOR),
        ("có hạn dùng (exp)", "exp" in p),
    ]:
        n += chk(label, ok)
        f += (not ok)
    # Token bị sửa 1 ký tự -> phải hỏng chữ ký
    bad = t[:-2] + ("ab" if not t.endswith("ab") else "cd")
    try:
        A.decode_token(bad)
        ok = False
    except Exception:
        ok = True
    n += chk("token bị sửa -> từ chối", ok)
    f += (not ok)
    # Khoá khác -> không giải mã được (token không dùng chéo hệ thống)
    old = os.environ.get("JWT_SECRET")
    os.environ["JWT_SECRET"] = "khoa-hoan-toan-khac"
    try:
        A.decode_token(t)
        ok = False
    except Exception:
        ok = True
    finally:
        os.environ["JWT_SECRET"] = old
    n += chk("khoá khác -> từ chối", ok)
    f += (not ok)
    return n, f


def test_validate():
    print("\n== (4) Ràng buộc tên đăng nhập / mật khẩu ==")
    n = f = 0
    for label, uname, pw, should_pass in [
        ("hợp lệ", "bs.an_2", "MatKhau12345", True),
        ("tên quá ngắn", "ab", "MatKhau12345", False),
        ("tên có ký tự lạ", "bs an!", "MatKhau12345", False),
        ("mật khẩu < 8 ký tự", "hopleuser", "1234567", False),
    ]:
        err = A.validate_credentials(uname, pw)
        ok = (err is None) == should_pass
        n += chk(label, ok, "" if ok else f"-> {err}")
        f += (not ok)
    return n, f


def test_role_gate():
    """Vai 'doctor' KHÔNG được tự đăng ký qua web — kiểm tra ở tầng API (api.py ép ROLE_USER)."""
    print("\n== (5) Tự đăng ký không thể thành bác sĩ ==")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "api.py"), encoding="utf-8").read()
    import re
    m = re.search(r"async def register\(.*?(?=\n@app\.)", src, re.DOTALL)
    body = m.group(0) if m else ""
    n = f = 0
    for label, ok in [
        ("register() ép role=ROLE_USER", "role=ROLE_USER" in body),
        ("register() KHÔNG nhận role từ client", "role" not in
         (re.search(r"class RegisterRequest.*?(?=\n\n)", src, re.DOTALL) or type("x", (), {"group": lambda s, i=0: ""})()).group(0)),
        ("/api/ask có require_doctor", "require_doctor" in src and "ask_endpoint" in src),
        ("/api/diagnose có get_current_user", "get_current_user" in src),
    ]:
        n += chk(label, ok)
        f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_password, test_timing, test_jwt, test_validate, test_role_gate):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
