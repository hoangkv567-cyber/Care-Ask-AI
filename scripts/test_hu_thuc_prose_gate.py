#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_hu_thuc_prose_gate.py — Cổng NHẤT QUÁN HƯ/THỰC ở prose Mục 3:
  _strip_thuc_cold_stagnation_in_pure_hu — gỡ mệnh đề THỰC 'hàn ngưng trệ' khi Bát Cương THUẦN HƯ.
Test hàm thuần (không Neo4j).  PYTHONIOENCODING=utf-8 python scripts/test_hu_thuc_prose_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def main():
    o = F.__new__(F)
    cases = []

    # ===== CA THẬT (A) — output app thật, nữ 22t, Bát Cương 'Lý - Hư', Mục 4 'Hư chứng thuần túy'
    A = ("- Đau đầu xuất hiện do khí huyết hư không đủ nuôi dưỡng kinh mạch vùng đầu cổ, kèm theo âm "
         "hàn ngưng trệ, khí huyết lưu thông kém, gây ra cảm giác đau nặng.")
    SYM_A = ("rêu trắng mỏng, mồ hôi trộm, ăn uống kém, người nặng, chóng mặt, hoa mắt, mất ngủ, "
             "sợ lạnh, đau đầu, ăn kém")
    BC_A = "Lý - Hư (tổng cương: thiên Âm)"

    # 0. Chứng minh hàm CŨ trượt (lời khai có 'sợ lạnh' -> return sớm)
    old = o._strip_unfounded_cold_mechanism(A, BC_A, "Khí huyết hư", "Không có", SYM_A)
    cases.append(("[nền] _strip_unfounded_cold_mechanism VẪN trượt câu (A)", old == A))

    # 1. Hàm mới GỠ đúng câu (A)
    r = o._strip_thuc_cold_stagnation_in_pure_hu(A, BC_A, "Khí huyết hư", "Không có")
    cases.append(("(A) gỡ được 'âm hàn ngưng trệ'", "hàn ngưng" not in r.lower()))
    cases.append(("(A) thay bằng cơ chế hư ('huyết hành vô lực')", "huyết hành vô lực" in r))
    cases.append(("(A) giữ nguyên phần còn lại của câu",
                  "khí huyết lưu thông kém, gây ra cảm giác đau nặng" in r
                  and "không đủ nuôi dưỡng kinh mạch" in r))
    cases.append(("(A) giữ 'kèm theo' -> câu không vỡ ngữ pháp", "kèm theo chính khí hư nhược" in r))

    # ===== NO-FIRE =====
    # 2. NGOẠI CẢM PHONG HÀN THẬT — 'hàn ngưng' HỢP LỆ, tuyệt đối không đụng
    P = ("- Phong hàn thúc biểu, hàn tà ngưng trệ kinh lạc vùng gáy vai, kinh khí bất thông nên đau "
         "đầu đau gáy, sợ lạnh, không mồ hôi.")
    n2 = o._strip_thuc_cold_stagnation_in_pure_hu(P, "Biểu - Hàn - Thực", "Phong hàn", "Không có")
    cases.append(("Ngoại cảm phong hàn (Biểu-Hàn-Thực): GIỮ 'hàn tà ngưng trệ'", n2 == P))

    # 2b. Ngoại cảm phong hàn nhưng Bát Cương lỡ ghi thiếu trục (chỉ 'Biểu - Hư' / rỗng)
    n2b = o._strip_thuc_cold_stagnation_in_pure_hu(P, "Biểu - Hư", "Phong hàn", "Không có")
    cases.append(("Phong hàn + Bát Cương thiếu tag Hàn: GIỮ (cổng hội chứng ngoại cảm)", n2b == P))

    # 3. DƯƠNG HƯ THẬT (Lý - Hàn - Hư)
    D = ("- Tỳ thận dương hư, hàn ngưng trung tiêu, dương khí không ôn vận nên đau bụng lạnh, ỉa chảy "
         "lúc sáng sớm.")
    n3 = o._strip_thuc_cold_stagnation_in_pure_hu(D, "Lý - Hàn - Hư", "Tỳ thận dương hư", "Không có")
    cases.append(("Dương hư (Lý-Hàn-Hư): GIỮ 'hàn ngưng trung tiêu'", n3 == D))

    # 3b. Dương hư mà Bát Cương BỎ trục Hàn (đúng lỗ hổng của ca A) -> cổng 2 phải chặn
    n3b = o._strip_thuc_cold_stagnation_in_pure_hu(D, "Lý - Hư", "Tỳ thận dương hư", "Không có")
    cases.append(("Core dương hư + Bát Cương thiếu Hàn: GIỮ (cổng hội chứng)", n3b == D))

    # 4. BẢN HƯ TIÊU THỰC có hàn ngưng huyết ứ (Thực hợp lệ)
    B = "- Trên nền khí huyết hư, hàn ngưng huyết ứ làm bào cung mất ôn dưỡng nên thống kinh."
    n4 = o._strip_thuc_cold_stagnation_in_pure_hu(B, "Lý - Bản Hư Tiêu Thực", "Khí huyết hư",
                                                  "Hàn ngưng huyết ứ")
    cases.append(("Bản Hư Tiêu Thực: GIỮ 'hàn ngưng huyết ứ'", n4 == B))

    # 5. HÀN NHIỆT THÁC TẠP
    n5 = o._strip_thuc_cold_stagnation_in_pure_hu(B, "Lý - Hàn Nhiệt Thác Tạp - Hư", "Khí hư", "Không có")
    cases.append(("Hàn Nhiệt Thác Tạp: GIỮ nguyên", n5 == B))

    # 6. Ca thuần Hư nhưng prose chỉ nói 'ngưng trệ' KHÔNG do hàn (nhân hư trí ứ — HỢP LỆ)
    H = ("- Khí hư nên huyết hành vô lực, khí huyết ngưng trệ nhẹ ở lạc mạch, gây đau âm ỉ; kinh mạch "
         "lưu thông kém.")
    n6 = o._strip_thuc_cold_stagnation_in_pure_hu(H, "Lý - Hư", "Khí huyết hư", "Không có")
    cases.append(("Thuần Hư + 'khí huyết ngưng trệ' (không hàn): GIỮ nguyên", n6 == H))

    # 7. Bát Cương Thực thuần (không 'hư') -> ngoài phạm vi, không đụng
    n7 = o._strip_thuc_cold_stagnation_in_pure_hu(P, "Lý - Thực", "Hàn thấp", "Không có")
    cases.append(("Bát Cương Thực thuần: GIỮ nguyên", n7 == P))

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += bool(c)
    print(f"\n{ok}/{len(cases)} PASS")
    print("\n--- Câu (A) SAU khi vá ---\n" + r)
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
