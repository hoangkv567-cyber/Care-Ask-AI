#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/split_csv_columns.py — Tách cột CSV: thêm 'chủ_chứng', 'lưỡi', 'mạch' (KHÔNG phá 'triệu_chứng').

KHÔNG PHÁ HỦY: cột 'triệu_chứng' gốc được GIỮ NGUYÊN (tầng khớp bệnh danh vẫn chạy y hệt, không hồi
quy). Ba cột mới CHỈ tách/bổ sung để bác sĩ xem & chỉnh, và cho tầng vọng chẩn khớp riêng dấu lưỡi:
  - lưỡi     : các field chứa 'lưỡi'/'rêu' (vọng chẩn lưỡi) — tách từ triệu_chứng.
  - mạch     : các field mạch tượng (thiết chẩn) — tách từ triệu_chứng (dùng _is_pulse_field).
  - chủ_chứng: gieo từ data/disease_gates.json theo TÊN bệnh (từ khóa chỉ điểm bệnh danh). Chỉ là
               gợi ý MÁY — bác sĩ nên rà lại. Rỗng nếu bệnh không có cổng.

Sao lưu bản gốc: data/Medicine_clean.bak.<timestamp>.csv. Idempotent (chạy lại tái tạo được).
Chạy:  python scripts/split_csv_columns.py [--dry-run]
"""
import os
import sys
import csv
import json
import shutil
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F

CSV_PATH = "data/Medicine_clean.csv"
GATES_PATH = "data/disease_gates.json"
OUT_FIELDS = ["tên_bệnh", "hội_chứng", "chủ_chứng", "triệu_chứng", "lưỡi", "mạch", "bài_thuốc", "vị_thuốc"]
CHIEF_CAP = 8   # số từ khóa chủ chứng gieo tối đa/bệnh (tránh cột quá dài)


def _is_tongue_field(f: str) -> bool:
    fl = f.lower()
    return ("lưỡi" in fl) or ("rêu" in fl)


def _load_chief_by_disease():
    """{tên_bệnh_lower_substring_match -> [từ khóa chủ chứng]} suy từ cổng (requires của cổng khớp tên)."""
    try:
        gates = json.load(open(GATES_PATH, encoding="utf-8")).get("gates", [])
    except Exception:
        return []
    # Giữ nguyên danh sách cổng; khớp theo tên như _validate_disease_safety.
    return gates


def _chief_for(disease_lower: str, gates) -> str:
    picked = []
    for g in gates:
        named = any(n in disease_lower for n in g.get("names", ()))
        if not named:
            import re
            for tok in g.get("names_regex", ()):
                if re.search(r'\b' + re.escape(tok) + r'\b', disease_lower):
                    named = True
                    break
        if named and g.get("requires"):
            for kw in g["requires"]:
                if kw not in picked:
                    picked.append(kw)
    return ", ".join(picked[:CHIEF_CAP])


def run(dry_run=False):
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    gates = _load_chief_by_disease()

    out_rows = []
    n_tongue = n_pulse = n_chief = 0
    for r in rows:
        trieu = r.get("triệu_chứng", "")
        fields = [s.strip() for s in trieu.split(",") if s.strip()]
        tongue = [f for f in fields if _is_tongue_field(f)]
        pulse = [f for f in fields if F._is_pulse_field(f.lower())]
        chief = _chief_for(r.get("tên_bệnh", "").lower(), gates)
        if tongue:
            n_tongue += 1
        if pulse:
            n_pulse += 1
        if chief:
            n_chief += 1
        out_rows.append({
            "tên_bệnh": r.get("tên_bệnh", ""),
            "hội_chứng": r.get("hội_chứng", ""),
            "chủ_chứng": chief,
            "triệu_chứng": trieu,                 # GIỮ NGUYÊN
            "lưỡi": ", ".join(tongue),
            "mạch": ", ".join(pulse),
            "bài_thuốc": r.get("bài_thuốc", ""),
            "vị_thuốc": r.get("vị_thuốc", ""),
        })

    print(f"Tổng {len(rows)} dòng | có dấu lưỡi: {n_tongue} | có mạch: {n_pulse} | có chủ chứng gieo: {n_chief}")
    if dry_run:
        print("[dry-run] không ghi file. Ví dụ 2 dòng đầu:")
        for r in out_rows[:2]:
            print("  ", {k: r[k] for k in ("tên_bệnh", "chủ_chứng", "lưỡi", "mạch")})
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"data/Medicine_clean.bak.{ts}.csv"
    shutil.copy2(CSV_PATH, backup)
    print(f"Đã sao lưu bản gốc -> {backup}")

    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Đã ghi CSV có cột mới ({', '.join(OUT_FIELDS)}) -> {CSV_PATH}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(run(args.dry_run))
