#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_no_coating_guard.py — Unit test guard NHẤT QUÁN LƯỠI-KHÔNG-RÊU
(_strip_no_coating_yin_claims): chặn LLM Mục 4 quy 'lưỡi không/ít rêu' cho huyết hư/thấp
khi cốt lõi KHÔNG liên quan âm/tân. KHÔNG cần Neo4j/LLM.

Chạy:  python scripts/test_no_coating_guard.py   (exit !=0 nếu fail)
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
o = F.__new__(F)

# Câu Mục 4 THẬT từ ca Đầu thống × Huyết hư
CASE_TEXT = ("Huyết hư khiến chóng mặt và hoa mắt. "
             "Lưỡi không có rêu cũng là một biểu hiện của huyết hư, khi lượng máu không đủ để "
             "vinh nhuận đầy đủ lên thân lưỡi, lưỡi sẽ trở nên nhạt và không có rêu.")

CASES = [
    # (tên, text, final_primary, kỳ_vọng_strip?, chuỗi phải/không được còn)
    ("Core Huyết hư: gỡ câu 'lưỡi không rêu = huyết hư'",
     CASE_TEXT, "Huyết hư", True, "âm dịch"),
    ("Core Can thận ÂM hư: GIỮ nguyên (quy đúng)",
     "Lưỡi ít rêu do âm dịch hư tổn, phản ánh Can thận âm hư.", "Can thận âm hư", False, None),
    ("Không nhắc lưỡi-không-rêu: no-op",
     "Huyết hư gây chóng mặt, hoa mắt, mệt mỏi.", "Huyết hư", False, None),
    ("Lưỡi không rêu nhưng KHÔNG quy sai: no-op (không over-strip)",
     "Lưỡi không rêu, cần theo dõi thêm về sau.", "Huyết hư", False, None),
    ("Core Khí hư + 'ít rêu do khí huyết': gỡ",
     "Người mệt do khí hư. Lưỡi ít rêu là do khí huyết suy yếu không nuôi lưỡi.", "Khí hư", True, "âm dịch"),
]


def main():
    npass = nfail = 0
    for name, text, core, expect_strip, must_contain in CASES:
        out = F._strip_no_coating_yin_claims(o, text, core)
        stripped = (out != text)
        ok = (stripped == expect_strip)
        if ok and must_contain:
            ok = must_contain in out.lower()
        # với ca no-op, đảm bảo KHÔNG mất câu gốc
        if not expect_strip:
            ok = ok and (out == text)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        strip={stripped} (kỳ vọng {expect_strip})")
        if stripped:
            print(f"        -> {out[:150]}")
        npass += ok
        nfail += (not ok)
    print(f"\n{npass} PASS, {nfail} FAIL / {len(CASES)}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
