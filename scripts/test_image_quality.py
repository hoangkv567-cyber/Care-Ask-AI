#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_image_quality.py — Kiểm GÁC CHẤT LƯỢNG ẢNH vọng chẩn (src/image_quality.py).

Sinh ảnh tổng hợp bằng Pillow rồi kiểm assess_image_quality phân loại đúng: ảnh quá nhỏ / quá tối /
quá sáng / quá mờ (đồng màu) bị CHẶN, ảnh đủ sáng-rõ nét được QUA. Hoàn toàn offline.

Chạy:  python scripts/test_image_quality.py    (Exit 0 nếu PASS)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_quality import assess_image_quality

from PIL import Image


def _save(img, name, tmp):
    p = os.path.join(tmp, name)
    img.save(p)
    return p


def run():
    tmp = tempfile.mkdtemp(prefix="tcm_imgq_")
    cases = []  # (path, ok kỳ vọng, nhãn)

    # 1. Ảnh quá nhỏ (100x100) dù nội dung nhiễu rõ
    small = Image.frombytes("L", (100, 100), os.urandom(100 * 100))
    cases.append((_save(small, "small.png", tmp), False, "quá nhỏ"))

    # 2. Ảnh quá tối (gần đen)
    dark = Image.new("RGB", (400, 400), (8, 8, 8))
    cases.append((_save(dark, "dark.png", tmp), False, "quá tối"))

    # 3. Ảnh quá sáng / lóa (gần trắng)
    bright = Image.new("RGB", (400, 400), (250, 250, 250))
    cases.append((_save(bright, "bright.png", tmp), False, "quá sáng"))

    # 4. Ảnh mờ nặng: đồng màu xám vừa (không chi tiết)
    flat = Image.new("L", (400, 400), 128)
    cases.append((_save(flat, "flat.png", tmp), False, "quá mờ/đồng màu"))

    # 5. Ảnh TỐT: đủ sáng + nhiều chi tiết (nhiễu ngẫu nhiên, mean ~128)
    good = Image.frombytes("L", (400, 400), os.urandom(400 * 400))
    cases.append((_save(good, "good.png", tmp), True, "đủ sáng, rõ nét"))

    # 6. File hỏng (không phải ảnh)
    bad_path = os.path.join(tmp, "broken.png")
    with open(bad_path, "wb") as f:
        f.write(b"not an image")
    cases.append((bad_path, False, "file hỏng"))

    all_ok = True
    for path, expect_ok, label in cases:
        ok, reason = assess_image_quality(path)
        status = "PASS" if ok == expect_ok else "FAIL"
        if ok != expect_ok:
            all_ok = False
        print(f"  [{status}] {label}: ok={ok} reason={reason!r} (kỳ vọng ok={expect_ok})")

    print("\n" + ("✅ TẤT CẢ PASS" if all_ok else "❌ CÓ CA FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
