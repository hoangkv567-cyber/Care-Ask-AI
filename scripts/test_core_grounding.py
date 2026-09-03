#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_core_grounding.py — Unit test CỔNG GROUNDING CORE
(TCMFusionPipeline._reground_core / _core_grounded_in_window). Test THẲNG classmethod
thuần với cửa sổ tổng hợp -> KHÔNG cần Neo4j/LLM.

Chạy:  python scripts/test_core_grounding.py   (exit !=0 nếu fail)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as P

# Cửa sổ bệnh nôn (Nôn mửa/Ẩu thổ): 6 hội chứng thật
NON = [{"benh_ly": "Nôn mửa", "ratio": 0.5,
        "hoi_chung_all": ["Vị nhiệt", "Can khí phạm vị", "Vị hư hàn", "Âm hư", "Thương thực", "Đàm ẩm"]}]
# Cửa sổ bệnh có thể chung 'Khí hư' (để test biến thể tạng)
KHIHU = [{"benh_ly": "Viêm phế quản", "ratio": 0.6, "hoi_chung_all": ["Khí hư", "Phong hàn", "Phong nhiệt"]}]

CASES = [
    # (tên, final_primary, all_syndromes, window, kỳ vọng core sau)
    ("FOREIGN: Khí trệ huyết ứ (promiscuity) -> re-rank Thương thực",
     "Khí trệ huyết ứ", ["Khí trệ huyết ứ", "Thực tích", "Thương thực", "Can vị bất hòa"], NON, "Thương thực"),
    ("GROUNDED exact: Can khí phạm vị -> no-op",
     "Can khí phạm vị", ["Can khí phạm vị", "Thương thực"], NON, "Can khí phạm vị"),
    ("GROUNDED synonym: Can vị bất hòa ≡ Can khí phạm vị -> no-op",
     "Can vị bất hòa", ["Can vị bất hòa", "Thương thực"], NON, "Can vị bất hòa"),
    ("GROUNDED biến thể tạng: Phế khí hư ⊃ Khí hư -> no-op",
     "Phế khí hư", ["Phế khí hư", "Phong hàn"], KHIHU, "Phế khí hư"),
    ("FOREIGN nhưng KHÔNG ứng viên grounded -> giữ core cũ",
     "Khí trệ huyết ứ", ["Khí trệ huyết ứ", "Huyết ứ", "Độc ứ"], NON, "Khí trệ huyết ứ"),
    ("Cửa sổ rỗng -> giữ core cũ",
     "Khí trệ huyết ứ", ["Khí trệ huyết ứ", "Thương thực"], [], "Khí trệ huyết ứ"),
    ("FOREIGN: re-rank chọn ứng viên grounded CAO NHẤT (Can khí phạm vị trước Thương thực)",
     "Độc ứ", ["Độc ứ", "Can khí phạm vị", "Thương thực"], NON, "Can khí phạm vị"),
]


def main():
    npass = nfail = 0
    for name, fp, allsyn, win, expect in CASES:
        got, reason = P._reground_core(fp, allsyn, win)
        ok = (got == expect)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        {fp!r} -> {got!r} (kỳ vọng {expect!r})" + (f"  | {reason}" if reason else "  | no-op"))
        npass += ok
        nfail += (not ok)
    print(f"\n{npass} PASS, {nfail} FAIL / {len(CASES)}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
