#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_thermal_gate.py — Test CỔNG THERMAL bài/thể (đổi bài HÀN -> thể ẤM cho ca hàn/0-nhiệt).
Khoá các bất biến do thẩm định đối kháng chỉ ra: (1) thang điểm tính vị exact-first/longest-key
(phụ tử ấm, Sinh địa≠Thục địa); (2) chọn thể WARM-DEF (dương hư/hư hàn/khí hư) đúng cơ chế, LOẠI
nhiệt/âm-hư/huyết-ứ/đàm; (3) dấu HÀN mạnh KHÔNG gồm mơ hồ, dấu KHÓA gồm âm-hư. Cần CSV (không Neo4j).
    python scripts/test_thermal_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def main():
    o = F.__new__(F)
    o._load_csv_data()
    F._herb_thermal_map = None  # nạp lại từ file
    cases = []

    def chk(d, c):
        cases.append((d, bool(c)))

    # (1) thang điểm tính vị
    chk("phụ tử/nhục quế/can khương -> ẤM (>0.5)", F._formula_thermal_mean("Phụ tử, Nhục quế, Can khương") > 0.5)
    chk("thạch cao/hoàng liên/sinh địa -> HÀN (<-1)", F._formula_thermal_mean("Thạch cao, Hoàng liên, Sinh địa") < -1)
    chk("Sinh địa (hàn) ≠ Thục địa (ấm)", F._formula_thermal_mean("Sinh địa") < 0 < F._formula_thermal_mean("Thục địa"))
    chk("exact-first: 'phụ tử' không bị 'địa phụ tử' lật (>0)", F._formula_thermal_mean("Phụ tử") > 0)

    # (2) chọn thể WARM-DEF cho Tiêu khát (ca hàn, cur -1.0)
    alt = o._warmest_alt_warm_formula("Tiêu khát", -1.0)
    chk("Tiêu khát -> có thể WARM-DEF ấm hơn", alt is not None and alt[4] >= -0.3)
    chk("thể chọn là DƯƠNG HƯ (đúng cơ chế hàn)", alt is not None and "dương hư" in alt[1].lower())
    # KHÔNG chọn thể nhiệt/âm-hư/huyết-ứ (dù có thể ấm hơn về số)
    if alt:
        chk("thể chọn KHÔNG phải nhiệt/âm-hư/huyết-ứ/đàm",
            not any(k in alt[1].lower() for k in ("nhiệt", "hỏa", "âm hư", "huyết ứ", "đàm")))

    # (3) regex thể + dấu
    chk("WARM_DEF khớp 'Thận dương hư'", bool(F._WARM_DEF_THE_RE.search("thận dương hư")))
    chk("WARM_DEF khớp 'Tỳ vị hư hàn'", bool(F._WARM_DEF_THE_RE.search("tỳ vị hư hàn")))
    chk("WARM_DEF KHÔNG khớp 'Thận âm hư'", not F._WARM_DEF_THE_RE.search("thận âm hư"))
    chk("NON_DEF khớp 'Khí trệ huyết ứ'", bool(F._NON_DEF_PATHO_RE.search("khí trệ huyết ứ")))
    chk("dấu HÀN mạnh CÓ 'nước tiểu trong'", "nước tiểu trong" in F._THERMAL_COLD_STRONG)
    chk("dấu HÀN mạnh KHÔNG gồm 'tiểu đêm' (mơ hồ)", "tiểu đêm" not in F._THERMAL_COLD_STRONG)
    chk("dấu KHÓA gồm 'mồ hôi trộm' (âm hư)", "mồ hôi trộm" in F._THERMAL_NO_SWAP_SIGNS)
    chk("dấu KHÓA gồm 'lưỡi đỏ'", "lưỡi đỏ" in F._THERMAL_NO_SWAP_SIGNS)

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += c
    print(f"\n{ok}/{len(cases)} PASS")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
