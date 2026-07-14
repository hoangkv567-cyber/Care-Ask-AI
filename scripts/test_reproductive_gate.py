#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_reproductive_gate.py — Test CỔNG TRẠNG THÁI SINH SẢN (_reproductive_state_conflict):
loại bệnh THAI SẢN/HẬU SẢN khi lời khai đang HÀNH KINH + không dấu mang thai/hậu sản (đang có kinh
thì không thể mang thai). Không cần Neo4j.  python scripts/test_reproductive_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def main():
    o = F.__new__(F)
    cases = [
        # (bệnh, lời khai, kỳ vọng conflict)
        ("Động thai", "kinh nguyệt không đều, đau bụng, đau mình, sợ lạnh", True),   # hành kinh -> loại
        ("Sản hậu phúc thống", "kinh nguyệt không đều, đau bụng", True),             # hành kinh -> loại
        ("Lưu sản", "hành kinh, đau bụng dưới", True),                               # hành kinh -> loại
        ("Nhâm thần ố trở", "thống kinh, buồn nôn", True),                           # thống kinh (đang kinh) -> loại
        ("Động thai", "có thai 3 tháng, đau bụng, ra ít huyết", False),              # mang thai -> GIỮ
        ("Sản hậu phúc thống", "mới sinh 10 ngày, đau bụng, sợ lạnh", False),        # hậu sản -> GIỮ
        ("Động thai", "đau bụng, sợ lạnh", False),                                   # không dấu kinh -> bay mù, GIỮ
        ("Thống kinh", "kinh nguyệt không đều, đau bụng", False),                    # KHÔNG phải bệnh thai sản -> GIỮ
        ("Canh niên kỳ hội chứng", "kinh nguyệt không đều, bốc hỏa", False),         # bệnh kinh nguyệt -> GIỮ
        ("Tử giản (sản giật)", "kinh nguyệt không đều, phù", True),                  # sản giật + hành kinh -> loại
    ]
    ok = 0
    for dz, txt, want in cases:
        got = o._reproductive_state_conflict(dz, txt, [])
        s = "PASS" if got == want else "FAIL"
        ok += got == want
        print(f"  [{s}] {dz!r} / {txt[:38]!r} -> conflict={got} (want {want})")
    print(f"\n{ok}/{len(cases)} PASS")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
