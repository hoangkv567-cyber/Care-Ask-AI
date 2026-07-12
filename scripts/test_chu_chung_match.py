#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_chu_chung_match.py — Test KHỚP CHỦ_CHỨNG cardinal đặc thù giới.

Trước đây _find_matching_diseases chỉ chấm triệu_chứng -> bệnh nam/phụ khoa khai đúng CHỦ CHỨNG
('liệt dương', 'thống kinh') vẫn TRỐNG (bị chôn). Nay nạp bệnh có chủ chứng cardinal đặc thù giới.
CHỈ ĐỌC CSV, không Neo4j.  python scripts/test_chu_chung_match.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def top(o, syms, raw, k=3):
    o._patient_sex = None
    return [m["benh_ly"].strip() for m in F._find_matching_diseases(o, syms, raw_user_text=raw)[:k]]


def main():
    o = F.__new__(F)
    F._load_csv_data(o)
    if not getattr(o, "csv_rows", None):
        print("KHÔNG nạp CSV"); return 1
    cases = []

    # 1. CHỈ chủ chứng cardinal -> ra đúng họ bệnh (trước là TRỐNG)
    t = top(o, ["liệt dương"], "liệt dương")
    cases.append(("Chỉ 'liệt dương' -> ra nam khoa (Dương nuy/Liệt dương/Di tinh)",
                  any(x in t for x in ("Dương nuy", "Liệt dương", "Di tinh"))))

    # 2. Ca thật: liệt dương + nhiễu (ngứa/quầng) -> nam khoa VƯỢT bệnh da (Bạch biến)
    t2 = top(o, ["liệt dương", "ngứa", "quầng đen dưới mắt"], "liệt dương, ngứa, quầng đen dưới mắt")
    cases.append(("Ca thật liệt dương+ngứa -> #1 là nam khoa, KHÔNG phải Bạch biến",
                  t2 and t2[0] in ("Dương nuy", "Liệt dương", "Di tinh") and "Bạch biến" not in t2[:1]))

    # 3. Chủ chứng CHUNG CHUNG (ợ chua/biếng ăn) KHÔNG được cc nạp (tránh soán ngôi)
    #    -> vẫn cho ra bệnh tiêu hoá hợp lý, không phải bệnh chỉ khớp 'ăn kém'
    t3 = top(o, ["nôn ra thức ăn", "ợ chua", "đau bụng"], "nôn ra thức ăn, ợ chua, đau bụng", k=5)
    cases.append(("Chủ chứng chung 'ợ chua' KHÔNG nạp Yếm thực lên #1",
                  "Yếm thực (biếng ăn)" not in t3[:1]))

    # 4. helper: chủ chứng đặc thù giới nằm trong _cc_admit_set
    o._cc_match_ratio(o.csv_rows[0], [], "")   # init _cc_admit_set
    cases.append(("'liệt dương' in _cc_admit_set", "liệt dương" in o._cc_admit_set))
    cases.append(("'ợ chua' NOT in _cc_admit_set", "ợ chua" not in o._cc_admit_set))
    cases.append(("_cc_ratio('liệt dương')>0", o._cc_ratio("liệt dương") > 0))

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += bool(c)
    print(f"\n{ok}/{len(cases)} PASS")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
