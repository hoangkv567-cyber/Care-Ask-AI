#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_herb_dedupe.py — Khóa hành vi khử trùng vị thuốc (_dedupe_herbs).

KG gộp nhiều dòng CSV gần trùng của cùng bài -> danh sách vị thuốc lặp biến thể bào chế/chính tả
('Sinh mẫu lệ'+'Mẫu lệ', 'Màn kinh'+'Mạn kinh tử'). Mục 5 phải hiển thị mỗi vị MỘT lần, ưu tiên
dạng chuẩn.

Chạy:  python scripts/test_herb_dedupe.py    (Exit 0 nếu PASS)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


CASES = [
    # (input, kỳ vọng)
    ("Sinh mẫu lệ, Mẫu lệ", "Mẫu lệ"),                       # bỏ tiền tố bào chế, giữ dạng gốc
    ("Mẫu lệ, Sinh mẫu lệ", "Mẫu lệ"),
    ("Màn kinh, Mạn kinh tử", "Mạn kinh tử"),                # alias -> dạng chuẩn
    ("Mạn kinh tử, Màn kinh", "Mạn kinh tử"),
    ("Cam thảo, Chích cam thảo", "Cam thảo"),
    ("Đương qui, Đương quy", "Đương qui, Đương quy"),         # KHÔNG gộp y/i (thận trọng)
    ("Đẳng sâm, Bạch truật, Cam thảo", "Đẳng sâm, Bạch truật, Cam thảo"),  # không đổi khi không trùng
    ("", ""),
]


def run():
    ok = True
    for inp, exp in CASES:
        got = F._dedupe_herbs(inp)
        status = "PASS" if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"  [{status}] {inp!r} -> {got!r} (kỳ vọng {exp!r})")
    # Ca thật của người dùng: 20 vị có 2 cặp trùng -> còn 18, không còn cặp biến thể
    real = ("Đẳng sâm, Bạch truật, Cam thảo, Xuyên khung, Bạch thược, Tang kí sinh, Kỉ tử, Cúc hoa, "
            "Thục địa, Bạch linh, Đương qui, Ngũ vị, Ngưu tất, Sinh mẫu lệ, Mẫu lệ, Long nhãn, "
            "Hà thủ ô, Địa long, Màn kinh, Mạn kinh tử")
    out = F._dedupe_herbs(real)
    n = len(out.split(","))
    cond = n == 18 and "Sinh mẫu lệ" not in out and "Màn kinh" not in out and "Mẫu lệ" in out and "Mạn kinh tử" in out
    print(f"  [{'PASS' if cond else 'FAIL'}] ca thật: 20 -> {n} vị, dùng dạng chuẩn")
    ok &= cond
    print("\n" + ("✅ TẤT CẢ PASS" if ok else "❌ CÓ CA FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
