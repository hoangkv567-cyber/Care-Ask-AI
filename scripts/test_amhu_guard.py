#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_amhu_guard.py — Test CỔNG ÂM-HƯ KHÔNG NHIỆT (GUARD RULE 4 deterministic).

Core âm-hư mà lời khai không dấu nhiệt + có dấu hư-hàn/thấp -> hạ bậc sang non-âm-hư.
CHỈ test hàm _demote_amhu_without_heat (không cần Neo4j).  python scripts/test_amhu_guard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def main():
    o = F.__new__(F)
    cases = []

    # 1. Ca thật: Thận âm hư + không nhiệt + có hư-hàn/thấp -> HẠ BẬC sang Đờm Trọc
    syn = ["Thận âm hư", "Đờm Trọc Ngăn Trở", "Khí hư"]
    txt = "không muốn hoạt động, người nặng nề, mồ hôi trộm, nặng đầu, mệt mỏi, quầng đen dưới mắt, rêu trắng mỏng"
    new, reason = o._demote_amhu_without_heat(syn, txt)
    cases.append(("Thận âm hư + không nhiệt + hư-hàn -> hạ bậc", new[0] == "Đờm Trọc Ngăn Trở" and reason))

    # 2. Âm hư CÓ dấu nhiệt (gò má đỏ, ngũ tâm phiền nhiệt) -> GIỮ NGUYÊN (âm hư thật)
    new2, r2 = o._demote_amhu_without_heat(
        ["Thận âm hư", "Đờm thấp"], "gò má đỏ, ngũ tâm phiền nhiệt, mồ hôi trộm, mệt mỏi")
    cases.append(("Âm hư + gò má đỏ/ngũ tâm phiền nhiệt -> GIỮ", new2[0] == "Thận âm hư" and r2 is None))

    # 3. Âm hư + dấu KHÔ (họng khô/khát) -> GIỮ (khô ủng hộ âm hư)
    new3, r3 = o._demote_amhu_without_heat(["Phế âm hư", "Khí hư"], "họng khô, ho khan, khát nước, mệt mỏi")
    cases.append(("Âm hư + họng khô/khát -> GIỮ", new3[0] == "Phế âm hư" and r3 is None))

    # 4. Core KHÔNG phải âm hư (Phế khí hư) -> GIỮ NGUYÊN
    new4, r4 = o._demote_amhu_without_heat(["Phế khí hư", "Phế âm hư"], "mệt mỏi, sợ lạnh, rêu trắng")
    cases.append(("Core khí hư (không âm hư) -> GIỮ", new4[0] == "Phế khí hư" and r4 is None))

    # 5. Âm hư + không nhiệt NHƯNG cũng không dấu hư-hàn rõ -> GIỮ (không đủ cơ sở)
    new5, r5 = o._demote_amhu_without_heat(["Can thận âm hư", "Huyết ứ"], "đau lưng, ù tai")
    cases.append(("Âm hư + không nhiệt + không hư-hàn rõ -> GIỮ", new5[0] == "Can thận âm hư" and r5 is None))

    # 6. Âm hư core nhưng KHÔNG có ứng viên non-âm-hư -> GIỮ (không có gì để thay)
    new6, r6 = o._demote_amhu_without_heat(["Thận âm hư", "Can thận âm hư"], "mệt mỏi, sợ lạnh, rêu trắng")
    cases.append(("Âm hư + chỉ toàn âm hư -> GIỮ", new6[0] == "Thận âm hư" and r6 is None))

    # 7. helper _is_amhu_syndrome
    cases.append(("_is_amhu('Thận âm hư')=True", F._is_amhu_syndrome("Thận âm hư") is True))
    cases.append(("_is_amhu('Thận dương hư')=False", F._is_amhu_syndrome("Thận dương hư") is False))
    cases.append(("_is_amhu('Đờm thấp')=False", F._is_amhu_syndrome("Đờm thấp") is False))

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += bool(c)
    print(f"\n{ok}/{len(cases)} PASS")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
