#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_synonym_gate.py — Test tầng [FALLBACK THỂ ĐỒNG NGHĨA] Mục 5.

(1) Unit test _syndromes_are_synonyms (bản đồ curated data/syndrome_synonyms.json):
    cặp đồng nghĩa -> True; cặp lệch Âm/Dương (nguy hiểm) -> False.
(2) Mô phỏng fallback trên csv_rows THẬT: core 'Can vị bất hòa' + bệnh Nôn mửa/Ẩu thổ
    phải bắc cầu sang 'Can khí phạm vị' -> đúng bài (KHÔNG cần Neo4j/LLM).

Chạy:  python scripts/test_synonym_gate.py   (exit !=0 nếu fail)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as F

SYN_CASES = [
    ("Can vị bất hòa", "Can khí phạm vị", True),
    ("Can vị bất hòa (can khí uất trệ)", "Can khí phạm vị", True),   # bỏ ngoặc vẫn khớp
    ("Vị âm hư", "Vị âm bất túc", True),
    ("Tỳ vị hư nhược", "Tỳ vị hư yếu", True),
    ("Can vị bất hòa", "Can vị bất hòa", False),                     # đồng nhất -> tầng exact lo
    ("Can Thận âm hư", "Tỳ Thận dương hư", False),                   # ⚠ lệch Âm/Dương — PHẢI False
    ("Tỳ vị hư nhược", "Tỳ vị hư hàn", False),                       # ⚠ lệch Hàn — PHẢI False
    ("Phong nhiệt", "Phong hàn", False),                            # trái cực
    ("Can vị bất hòa", "Vị âm hư", False),                          # khác cụm
]


def test_helper():
    npass = nfail = 0
    print("== (1) _syndromes_are_synonyms ==")
    for a, b, expect in SYN_CASES:
        got = F._syndromes_are_synonyms(a, b)
        ok = (got == expect)
        print(f"[{'PASS' if ok else 'FAIL'}] {a!r} ~ {b!r} -> {got} (kỳ vọng {expect})")
        npass += ok
        nfail += (not ok)
    return npass, nfail


def test_fallback_simulation():
    """Mô phỏng đúng logic khối [FALLBACK THỂ ĐỒNG NGHĨA] trên csv_rows thật."""
    print("\n== (2) Mô phỏng fallback Mục 5 (core 'Can vị bất hòa') ==")
    o = F.__new__(F)
    F._load_csv_data(o)
    npass = nfail = 0
    for disease, expect_bai in [("Nôn mửa", "Bình can bổ thổ"),
                                ("Ẩu thổ", "Tiêu dao tán gia vị hợp Tả kim hoàn")]:
        core = "Can vị bất hòa"
        hits = []
        for r in o.csv_rows:
            b = r.get("benh_ly", "").strip()
            hc = r.get("hoi_chung", "").strip()
            bt = r.get("bai_thuoc", "").strip()
            if b.lower() != disease.lower() or not hc or not bt:
                continue
            if not F._syndromes_are_synonyms(core, hc):
                continue
            if F._syndrome_is_hu(hc) != F._syndrome_is_hu(core):
                continue
            if F._syndromes_thermal_conflict(core, hc):
                continue
            hits.append((hc, bt))
        got = hits[0][1] if hits else None
        ok = (got == expect_bai) and len(hits) == 1
        print(f"[{'PASS' if ok else 'FAIL'}] {disease} × {core} -> {hits} (kỳ vọng đúng 1: {expect_bai!r})")
        npass += ok
        nfail += (not ok)
    return npass, nfail


def main():
    p1, f1 = test_helper()
    p2, f2 = test_fallback_simulation()
    total_p, total_f = p1 + p2, f1 + f2
    print(f"\n{total_p} PASS, {total_f} FAIL")
    sys.exit(1 if total_f else 0)


if __name__ == "__main__":
    main()
