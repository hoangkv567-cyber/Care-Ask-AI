#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_concurrent_gate.py — Unit test CỔNG CÙNG-BỆNH cho hội chứng kèm theo
(TCMFusionPipeline._gate_concurrent_by_disease). Test THẲNG classmethod thuần với
matched_diseases tổng hợp -> KHÔNG cần Neo4j/LLM.

Chạy:  python scripts/test_concurrent_gate.py   (exit !=0 nếu có case fail)
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


def D(benh, ratio, syns, hoi_chung=None):
    return {"benh_ly": benh, "ratio": ratio, "hoi_chung": hoi_chung or (syns[0] if syns else ""),
            "hoi_chung_all": syns}


CASES = [
    # (tên, final_primary, final_concurrent, matched_diseases, disease_names, overridden, kỳ vọng)
    (
        "BUG: Doanh vệ bất hòa (Tiểu nhi hãn chứng) NGOÀI cửa sổ -> loại",
        "Phong nhiệt phạm phế", "Doanh vệ bất hòa",
        [D("Viêm yết hầu", 0.80, ["phong nhiệt", "âm hư", "khí trệ huyết ứ"]),
         D("Tiểu nhi hãn chứng", 0.40, ["doanh vệ bất hòa", "phế vệ bất cố"])],
        ["Viêm yết hầu"], False, "Không có",
    ),
    (
        "BUG-biến thể: bệnh-nhà TRONG cửa sổ nhưng vẫn khác cực (giữ — residual risk đã biết)",
        # tài liệu hoá residual risk #1: nếu bệnh-nhà lọt cửa sổ, cổng cho qua.
        "Phong nhiệt phạm phế", "Doanh vệ bất hòa",
        [D("Viêm yết hầu", 0.80, ["phong nhiệt"]),
         D("Tiểu nhi hãn chứng", 0.72, ["doanh vệ bất hòa"])],
        ["Viêm yết hầu", "Tiểu nhi hãn chứng"], False, "Doanh vệ bất hòa",
    ),
    (
        "LEGIT chéo bệnh (biểu-lý đồng bệnh): Cảm mạo Phong hàn + Thực tích (Thương thực) -> GIỮ",
        "Phong hàn", "Thực tích",
        [D("Cảm mạo", 0.70, ["phong hàn", "phong nhiệt"]),
         D("Thương thực", 0.66, ["thực tích", "tỳ vị hư nhược"])],
        ["Cảm mạo", "Thương thực"], False, "Thực tích",
    ),
    (
        "THERMAL phụ trợ: member=True nhưng cốt lõi Phong hàn vs kèm Phong nhiệt -> loại",
        "Phong hàn", "Phong nhiệt",
        [D("Cảm mạo", 0.75, ["phong hàn", "phong nhiệt"])],
        ["Cảm mạo"], False, "Không có",
    ),
    (
        "OVERRIDDEN (hardcode) -> GIỮ nguyên dù không grounded",
        "Khí huyết đều hư", "Tâm huyết hư",
        [D("Suy nhược cơ thể", 0.55, ["khí huyết đều hư"])],
        ["Suy nhược cơ thể"], True, "Tâm huyết hư",
    ),
    (
        "Concurrent = 'Không có' -> nguyên trạng",
        "Phong nhiệt phạm phế", "Không có",
        [D("Viêm yết hầu", 0.80, ["phong nhiệt"])],
        ["Viêm yết hầu"], False, "Không có",
    ),
    (
        "disease_names rỗng (bay mù) -> GIỮ nguyên",
        "Phong nhiệt phạm phế", "Doanh vệ bất hòa",
        [D("Viêm yết hầu", 0.80, ["phong nhiệt"])],
        [], False, "Doanh vệ bất hòa",
    ),
    (
        "matched_diseases rỗng -> GIỮ nguyên",
        "Phong nhiệt phạm phế", "Doanh vệ bất hòa",
        [], ["Viêm yết hầu"], False, "Doanh vệ bất hòa",
    ),
    (
        "SUBSTRING hợp lệ: kèm 'Tỳ khí hư' ⊂ 'Tỳ khí hư (kèm thấp)' trong cửa sổ -> GIỮ",
        "Can khí uất kết", "Tỳ khí hư",
        [D("Bĩ mãn", 0.70, ["can khí uất kết", "tỳ khí hư (kèm thấp)"])],
        ["Bĩ mãn"], False, "Tỳ khí hư",
    ),
]


def main():
    npass = nfail = 0
    for name, fp, fc, md, dn, ov, expect in CASES:
        got, reason = P._gate_concurrent_by_disease(fp, fc, md, dn, ov)
        ok = (got == expect)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        in={fc!r} -> out={got!r} (kỳ vọng {expect!r})" + (f"  | {reason}" if reason else ""))
        if ok:
            npass += 1
        else:
            nfail += 1
    print(f"\n{npass} PASS, {nfail} FAIL / {len(CASES)}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
