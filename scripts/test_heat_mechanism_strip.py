#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_heat_mechanism_strip.py — Test 2 cổng nhất-quán prose:
  _strip_unfounded_heat_mechanism  (gỡ 'âm hư nội nhiệt' bịa cho ca hư-hàn không căn cứ nhiệt)
  _strip_no_coating_yin_claims     (lưỡi-không-rêu: vị-khí-hư cho core hàn, âm-dịch cho core âm)
CHỈ test hàm thuần (không Neo4j).  python scripts/test_heat_mechanism_strip.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def main():
    o = F.__new__(F)
    cases = []

    # === _strip_unfounded_heat_mechanism ===
    # 1. Ca THẬT: Khí hư + Bát Cương Hàn -> phải gỡ 'âm hư nội nhiệt / nhiệt bức tân dịch'
    prose1 = ("Mồ hôi trộm và ra mồ hôi là do âm hư sinh nội nhiệt, nhiệt bức tân dịch tiết ra ngoài, "
              "đặc biệt là vào ban đêm.")
    r1 = o._strip_unfounded_heat_mechanism(prose1, "Lý - Hàn - Hư", "Khí hư", "Không có", "mồ hôi trộm, sợ lạnh, mệt mỏi")
    cases.append(("Khí hư+Hàn: gỡ 'âm hư sinh nội nhiệt'", "âm hư sinh nội nhiệt" not in r1 and "nội nhiệt" not in r1))
    cases.append(("Khí hư+Hàn: gỡ 'nhiệt bức tân dịch'", "nhiệt bức" not in r1))
    cases.append(("Khí hư+Hàn: có cơ chế vệ khí bất cố", "vệ biểu bất cố" in r1 or "cố nhiếp" in r1))

    # 2. Ca ÂM-HƯ THẬT (Bát Cương Nhiệt, core Thận âm hư) -> GIỮ NGUYÊN
    r2 = o._strip_unfounded_heat_mechanism(prose1, "Lý - Nhiệt - Hư", "Thận âm hư", "Không có", "mồ hôi trộm, gò má đỏ")
    cases.append(("Âm hư + Bát Cương Nhiệt -> GIỮ 'âm hư sinh nội nhiệt'", r2 == prose1))

    # 3. Bát Cương Hàn nhưng lời khai CÓ dấu nhiệt (sốt/khát) -> GIỮ (có căn cứ nhiệt)
    r3 = o._strip_unfounded_heat_mechanism(prose1, "Lý - Hàn - Hư", "Khí hư", "Không có", "sốt, khát nước, mồ hôi")
    cases.append(("Có dấu nhiệt trong lời khai -> GIỮ", r3 == prose1))

    # 4. Core âm hư (dù Bát Cương không rõ) -> GIỮ
    r4 = o._strip_unfounded_heat_mechanism(prose1, "Lý - Hư", "Can thận âm hư", "Không có", "mồ hôi trộm")
    cases.append(("Core âm hư -> GIỮ", r4 == prose1))

    # === _strip_no_coating_yin_claims (thermal-aware) ===
    txt = "Lưỡi không có rêu là biểu hiện của huyết hư, cho thấy huyết không đủ."
    # 5. Core Khí hư + Bát Cương Hàn -> sanctioned VỊ KHÍ HƯ (không âm dịch)
    n5 = o._strip_no_coating_yin_claims(txt, "Khí hư", "Lý - Hàn - Hư")
    cases.append(("Lưỡi-không-rêu + Khí hư+Hàn -> 'VỊ KHÍ hư tổn'", "vị khí" in n5.lower() and "âm dịch" not in n5.lower()))
    # 6. Core không âm, không hàn info -> sanctioned âm dịch (mặc định cũ)
    n6 = o._strip_no_coating_yin_claims(txt, "Huyết ứ", "")
    cases.append(("Lưỡi-không-rêu + core khác (no hàn) -> 'âm dịch' (mặc định)", "âm dịch" in n6.lower()))
    # 7. Core âm hư -> GIỮ NGUYÊN (không gỡ)
    n7 = o._strip_no_coating_yin_claims(txt, "Thận âm hư", "Lý - Nhiệt - Hư")
    cases.append(("Lưỡi-không-rêu + core âm hư -> GIỮ nguyên", n7 == txt))
    # 8. Core Phong nhiệt + Bát Cương Biểu-Nhiệt-Thực -> 'nhiệt hao tân dịch' (KHÔNG 'âm hư')
    n8 = o._strip_no_coating_yin_claims(txt, "Phong nhiệt", "Biểu - Nhiệt - Thực")
    cases.append(("Lưỡi-không-rêu + Phong nhiệt Thực -> 'nhiệt hao tân dịch', KHÔNG 'âm hư'",
                  "nhiệt làm hao tân dịch" in n8.lower() and "dấu âm hư" not in n8.lower()))

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += bool(c)
    print(f"\n{ok}/{len(cases)} PASS")
    if ok < len(cases):
        print("\n--- prose1 sau strip (ca thật) ---\n", r1)
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
