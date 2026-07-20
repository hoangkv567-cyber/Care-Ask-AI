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

    # ---- HUYẾT Ứ trong ca thuần Hư (cùng lớp lỗi, khác chất tà) ----
    # Ca thật 'Đầu thống × Huyết hư': Mục 2 chốt 'Lý - Hư', Mục 4 ghi 'Không có Tiêu Thực, Hư chứng
    # thuần túy', nhưng Mục 3 viết "huyết ứ tại kinh mạch vùng đầu cổ" — huyết ứ là TÀ THỰC, tự chọi
    # với hai mục hiển thị ngay cạnh; tin theo thì pháp trị phải HOẠT HUYẾT chứ không bổ huyết đơn
    # thuần. Đo 6 lần chạy app: 2/6 lần LLM viết mệnh đề này (dao động -> phải chặn ở hậu xử lý).
    BC_HU = "Lý - Hư (tổng cương: thiên Âm)"
    U = "Khí huyết không lưu thông, huyết ứ tại kinh mạch vùng đầu cổ, gây ra đau đầu."
    u1 = o._strip_thuc_cold_stagnation_in_pure_hu(U, BC_HU, "Huyết hư", "Không có")
    cases.append(("Thuần Hư + 'huyết ứ tại kinh mạch': GỠ", "huyết ứ" not in u1.lower()))
    cases.append(("thay bằng cơ chế hư đúng ('huyết hành vô lực')", "huyết hành vô lực" in u1.lower()))
    cases.append(("GIỮ vế kết quả của câu (không nuốt 'gây ra đau đầu')", "gây ra đau đầu" in u1))

    # PHỦ ĐỊNH: thay chữ trong câu phủ định sẽ ĐẢO NGƯỢC nghĩa -> phải giữ nguyên
    NEGU = "Bệnh còn nhẹ, không có huyết ứ, cũng không kèm ứ trệ."
    cases.append(("Câu PHỦ ĐỊNH huyết ứ: GIỮ nguyên",
                  o._strip_thuc_cold_stagnation_in_pure_hu(NEGU, BC_HU, "Huyết hư", "Không có") == NEGU))
    # Cổng 2: cốt lõi VỐN là thể huyết ứ -> huyết ứ là ĐÚNG, cấm gỡ
    UO = "huyết ứ tại kinh mạch gây đau nhức."
    cases.append(("Cốt lõi 'Khí trệ huyết ứ': GIỮ nguyên",
                  o._strip_thuc_cold_stagnation_in_pure_hu(UO, BC_HU, "Khí trệ huyết ứ", "") == UO))
    # Cổng 1: Bát Cương có Thực / có Hàn -> ngoài phạm vi
    for bc in ("Lý - Bản Hư Tiêu Thực", "Lý - Hàn - Hư", "Biểu - Hàn - Thực"):
        cases.append((f"Bát Cương '{bc}': GIỮ nguyên huyết ứ",
                      o._strip_thuc_cold_stagnation_in_pure_hu(UO, bc, "Huyết hư", "") == UO))

    # ---------------------------------------------------------------------------------------
    # [CHỦ NGỮ + ĐỊNH VỊ] Guard chỉ được viết lại khi LLM khẳng định Ổ Ứ HUYẾT CÓ ĐỊA CHỈ.
    #
    # Lỗi thật đã xảy ra: regex bắt cả 'ứ trệ'/'ứ đọng' TRẦN (vị ngữ đình tụ TRUNG TÍNH, nhận mọi
    # chủ ngữ) và nhóm định vị đóng bằng ')?' tức TÙY CHỌN — trái với chính chú thích của nó.
    # Hệ quả đo trên dữ liệu thật: 55/55 span cụm-ứ trong prose 40 ca bị nổ, trong khi chỉ 2 span
    # có định vị và CẢ HAI chủ ngữ đều PHI-huyết. Ca thật in ra "Thủy thấp huyết hành vô lực" ba
    # lần trong một đoạn — vô nghĩa, vì 水湿停聚 là HỆ QUẢ của Tỳ hư chứ không phải tà thực.
    # Nặng hơn: 5/5 chuỗi do CHÍNH MÃ sinh (_THUC_TEMPLATES) cũng bị bóp méo — hệ tự phá văn nó viết.
    # Và prompt Mục 3 RA LỆNH viết 'Thủy thấp ứ đọng' cho lưỡi bệu: prompt đúng, guard sai.
    #
    # ⚠ Bộ test này TRƯỚC ĐÓ cho 21/21 PASS với CẢ mã hỏng LẪN mã đúng — nó mù hoàn toàn với lớp
    # lỗi này. Đó là lý do phải thêm nhóm dưới đây.
    KEEP = [
        ("thủy thấp đình tụ (hệ quả Tỳ hư)", "Thủy thấp ứ đọng, huyết hành vô lực làm trệ khí."),
        ("lưỡi bệu do thủy thấp", "Lưỡi bệu là do Thủy thấp ứ đọng, không được vận hóa."),
        ("đàm thấp — chủ ngữ phi huyết", "Đàm thấp ứ trệ tại kinh lạc gây tê bì."),
        ("thức ăn đình trệ", "Thức ăn ứ đọng tại vị quản."),
        ("khí cơ đình trệ", "Khí cơ ứ trệ, ngực bụng đầy tức."),
        ("dấu VỌNG CHẨN, không phải khẳng định cơ chế", "Rìa lưỡi có ban ứ huyết, mạch sáp."),
        ("因虚致瘀 — mắt xích HỢP LỆ", "Khí hư nên huyết hành vô lực, lâu ngày sinh huyết ứ."),
        # Bất biến: MÃ KHÔNG ĐƯỢC TỰ BÓP MÉO VĂN CỦA CHÍNH NÓ (chuỗi lấy từ _THUC_TEMPLATES).
        ("chuỗi do chính mã sinh",
         "Rêu nhớt phản ánh đàm trọc / thủy thấp ứ đọng ở trung tiêu (yếu tố Tiêu Thực)."),
    ]
    for lab, s in KEEP:
        cases.append((f"GIỮ NGUYÊN [{lab}]",
                      o._strip_thuc_cold_stagnation_in_pure_hu(s, BC_HU, "Tỳ khí hư", "Không có") == s))
    # Vẫn phải GỠ: ổ ứ huyết CÓ ĐỊA CHỈ trong ca thuần Hư (lưới chặn ai đó "vá" bằng cách tắt guard)
    STRIP = [
        "Khí huyết không lưu thông, huyết ứ tại kinh mạch vùng đầu cổ, gây ra đau đầu.",
        "huyết ứ tại kinh mạch gây đau nhức.",
        "Ứ huyết ở kinh lạc gây đau cố định.",
    ]
    for s in STRIP:
        cases.append((f"VẪN GỠ [{s[:38]}...]",
                      o._strip_thuc_cold_stagnation_in_pure_hu(s, BC_HU, "Tỳ khí hư", "Không có") != s))
    # Không được để lại LẶP CỤM hay đuôi mồ côi
    _r1 = o._strip_thuc_cold_stagnation_in_pure_hu(KEEP[0][1], BC_HU, "Tỳ khí hư", "Không có")
    cases.append(("KHÔNG lặp cụm 'huyết hành vô lực'", _r1.lower().count("huyết hành vô lực") <= 1))

    # Khóa TRỰC TIẾP trên regex: hai điều kiện phải cùng BẮT BUỘC. Kiểm qua hàm là chưa đủ — cổng
    # vào của hàm có thể che mất tác dụng của từng điều kiện, khiến ai đó nới regex mà test vẫn xanh.
    _RX = F._BLOOD_STASIS_IN_HU_RE
    cases.append(("ĐK1 chủ ngữ HUYẾT: 'thủy thấp ứ đọng tại kinh lạc' KHÔNG khớp",
                  not _RX.search("thủy thấp ứ đọng tại kinh lạc")))
    cases.append(("ĐK2 định vị BẮT BUỘC: 'huyết ứ' trần KHÔNG khớp",
                  not _RX.search("lâu ngày sinh huyết ứ.")))
    cases.append(("vẫn khớp khi ĐỦ CẢ HAI: 'huyết ứ tại kinh mạch'",
                  bool(_RX.search("huyết ứ tại kinh mạch vùng đầu cổ"))))

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += bool(c)
    print(f"\n{ok}/{len(cases)} PASS")
    print("\n--- Câu (A) SAU khi vá ---\n" + r)
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
