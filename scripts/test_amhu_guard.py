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

    # 1. Ca thật: Thận âm hư + không nhiệt + có hư-hàn/thấp -> LOẠI HẲN âm-hư, core sang Đờm Trọc
    syn = ["Thận âm hư", "Đờm Trọc Ngăn Trở", "Khí hư"]
    txt = "không muốn hoạt động, người nặng nề, mồ hôi trộm, nặng đầu, mệt mỏi, quầng đen dưới mắt, rêu trắng mỏng"
    new, reason = o._demote_amhu_without_heat(syn, txt)
    cases.append(("Thận âm hư + không nhiệt + hư-hàn -> LOẠI HẲN âm-hư",
                  new[0] == "Đờm Trọc Ngăn Trở" and "Thận âm hư" not in new and reason))

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

    # 6b. [REGRESSION] Ca thật Nhĩ minh (đàm-thấp bị ép Can thận âm hư): KHÔNG có token 'mệt' nào,
    # vẫn phải LOẠI âm-hư qua dấu đàm-thấp/lưỡi (rêu trắng/lưỡi hồng nhạt/lưỡi bệu) — chốt fix B
    # (bỏ 'mệt') không làm ca này hồi quy.
    nhimin = "ù tai, rêu lưỡi trắng nhớt, lưỡi bệu, lưỡi hồng nhạt, rêu trắng mỏng, rìa lưỡi có hằn răng, mặt nhợt nhạt"
    new6b, r6b = o._demote_amhu_without_heat(["Can thận âm hư", "Khí huyết lưỡng hư", "Đờm thấp"], nhimin)
    cases.append(("Nhĩ minh (đàm-thấp, KHÔNG có 'mệt') -> vẫn LOẠI âm-hư",
                  "Can thận âm hư" not in new6b and bool(r6b)))

    # 6c. [REGRESSION] Ca hưởng lợi fix B: 'tiểu nhiều, lượng ít, mệt mỏi, ít ngủ' — dấu hư-hàn DUY
    # NHẤT là 'mệt' (đã bỏ) -> KHÔNG được loại Thận âm hư (trước đây loại oan -> core Huyết hư ngoại
    # lai + Mục 5 trắng). Giữ Thận âm hư làm cốt lõi.
    new6c, r6c = o._demote_amhu_without_heat(["Thận âm hư", "Huyết hư", "Khí hư"],
                                             "tiểu nhiều, lượng ít, mệt mỏi, ít ngủ")
    cases.append(("Thận âm hư + chỉ 'mệt' (không dấu hư-hàn khác) -> GIỮ (không loại oan)",
                  new6c[0] == "Thận âm hư" and r6c is None))

    # 7. helper _is_amhu_syndrome
    cases.append(("_is_amhu('Thận âm hư')=True", F._is_amhu_syndrome("Thận âm hư") is True))
    cases.append(("_is_amhu('Thận dương hư')=False", F._is_amhu_syndrome("Thận dương hư") is False))
    cases.append(("_is_amhu('Đờm thấp')=False", F._is_amhu_syndrome("Đờm thấp") is False))

    # ===== CỔNG THERMAL-POLARITY =====
    # T1. Ca thật: Phong nhiệt phạm phế + rêu trắng + tay chân lạnh + 0 dấu nhiệt -> loại nhiệt
    txt_cold = "ho tiếng thô nặng hoặc ho khan, tay chân lạnh, ra mồ hôi, khó thở, quầng đen dưới mắt, rêu trắng mỏng"
    n1, nr1 = o._demote_nhiet_without_heat(["Phong nhiệt phạm phế", "Phong hàn", "Đàm thấp"], txt_cold)
    cases.append(("Phong nhiệt + rêu trắng + tay chân lạnh + 0 nhiệt -> LOẠI nhiệt",
                  n1[0] == "Phong hàn" and "Phong nhiệt phạm phế" not in n1 and nr1))
    # T2. Phong nhiệt CÓ dấu nhiệt (họng đỏ sưng, sốt) -> GIỮ
    n2, nr2 = o._demote_nhiet_without_heat(
        ["Phong nhiệt phạm phế", "Phong hàn"], "ho, họng đỏ sưng đau, sốt, khát nước, rêu vàng")
    cases.append(("Phong nhiệt + họng đỏ/sốt/rêu vàng -> GIỮ", n2[0] == "Phong nhiệt phạm phế" and nr2 is None))
    # T3. Nhiệt + KHÔNG dấu hàn -> GIỮ (không đủ cơ sở)
    n3, nr3 = o._demote_nhiet_without_heat(["Thấp nhiệt", "Khí trệ"], "người mệt, đầy bụng")
    cases.append(("Nhiệt + không dấu hàn -> GIỮ", n3[0] == "Thấp nhiệt" and nr3 is None))
    # T4. Core không phải nhiệt -> GIỮ
    n4, nr4 = o._demote_nhiet_without_heat(["Phong hàn", "Đàm thấp"], "tay chân lạnh, rêu trắng")
    cases.append(("Core Phong hàn (không nhiệt) -> GIỮ", n4[0] == "Phong hàn" and nr4 is None))
    # T5. helper _is_nhiet_syndrome
    cases.append(("_is_nhiet('Phong nhiệt phạm phế')=True", F._is_nhiet_syndrome("Phong nhiệt phạm phế") is True))
    cases.append(("_is_nhiet('Phong hàn')=False", F._is_nhiet_syndrome("Phong hàn") is False))
    cases.append(("_is_nhiet('Thượng nhiệt hạ hàn')=False (tạp)", F._is_nhiet_syndrome("Thượng nhiệt hạ hàn") is False))
    cases.append(("_is_nhiet('Thận âm hư')=False (để cổng âm-hư lo)", F._is_nhiet_syndrome("Thận âm hư") is False))

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += bool(c)
    print(f"\n{ok}/{len(cases)} PASS")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
