#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/seed_users.py — Quản trị tài khoản (tạo bác sĩ / đổi mật khẩu / liệt kê).

Vai 'doctor' KHÔNG thể tự đăng ký qua web (nếu cho thì ai cũng tự nhận là bác sĩ để mở khoá
tính năng chuyên môn) — phải tạo bằng script này.

Dùng:
    python scripts/seed_users.py --list
    python scripts/seed_users.py --create-doctor bs.an --password 'MatKhauManh123' --name 'BS. Nguyễn An'
    python scripts/seed_users.py --create-user  benhnhan1 --password 'MatKhau12345'
    python scripts/seed_users.py --set-password bs.an --password 'MatKhauMoi456'

Mật khẩu có thể truyền qua biến môi trường SEED_PASSWORD thay cho --password (tránh lộ trong
lịch sử shell).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _load_env():
    """Nạp .env thủ công (dự án không dùng python-dotenv)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="liệt kê tài khoản")
    ap.add_argument("--create-doctor", metavar="USERNAME")
    ap.add_argument("--create-user", metavar="USERNAME")
    ap.add_argument("--set-password", metavar="USERNAME")
    ap.add_argument("--password", default="", help="mật khẩu (hoặc đặt biến SEED_PASSWORD)")
    ap.add_argument("--name", default="", help="họ tên hiển thị")
    args = ap.parse_args()

    _load_env()
    from neo4j import GraphDatabase
    from src.auth import UserStore, ROLE_DOCTOR, ROLE_USER, validate_credentials

    uri = os.getenv("NEO4J_URI")
    if not uri:
        print("LỖI: thiếu NEO4J_URI trong .env")
        return 1
    driver = GraphDatabase.driver(
        uri, auth=(os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"),
                   os.getenv("NEO4J_PASSWORD")))
    store = UserStore(driver)
    store.ensure_schema()

    pw = args.password or os.getenv("SEED_PASSWORD", "")

    if args.list:
        rows = store.list_users()
        if not rows:
            print("(chưa có tài khoản nào)")
        for r in rows:
            trang_thai = "khoá" if r.get("active") is False else "hoạt động"
            print(f"  {r['role']:<7} {r['username']:<20} {r.get('full_name') or '':<28} [{trang_thai}]")
        return 0

    target = args.create_doctor or args.create_user or args.set_password
    if not target:
        ap.print_help()
        return 1

    err = validate_credentials(target, pw)
    if err:
        print("LỖI:", err)
        return 1

    if args.set_password:
        ok = store.set_password(target, pw)
        print("Đã đổi mật khẩu cho", target if ok else f"(không thấy tài khoản {target})")
        return 0 if ok else 1

    role = ROLE_DOCTOR if args.create_doctor else ROLE_USER
    try:
        u = store.create(target, pw, role=role, full_name=args.name)
    except ValueError as e:
        print("LỖI:", e)
        return 1
    print(f"Đã tạo tài khoản [{u['role']}] {u['username']} — {u['full_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
