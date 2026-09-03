#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_pulse_exclusion.py — Khóa hành vi LOẠI FIELD MẠCH khỏi mẫu số tỷ lệ khớp.

Bối cảnh: hệ chỉ có Vấn chẩn + Vọng chẩn (lưỡi/mặt), KHÔNG bắt mạch. Field mạch tượng trong
CSV không bao giờ quan sát được, nên phải bị loại khỏi MẪU SỐ khi tính match_ratio (nếu không,
thể bệnh mô tả mạch càng kỹ càng bị phạt ratio oan).

Kiểm 2 điều:
  1. _is_pulse_field phân loại đúng (bắt 'mạch trầm/tế/sác...', KHÔNG bắt 'tĩnh mạch', 'mạch máu',
     'phù', 'vô lực', hay field lưỡi mở đầu bằng rêu rồi mới nhắc mạch).
  2. Với một dòng CSV có field mạch, mẫu số tỷ lệ khớp = số field QUAN SÁT ĐƯỢC (đã trừ mạch),
     nên khớp hết phần quan sát được cho ratio = 1.0 (trước fix sẽ < 1.0).

Chạy:  python scripts/test_pulse_exclusion.py    (Exit 0 nếu PASS, 1 nếu FAIL — không cần Neo4j)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


# (field, kỳ vọng là field mạch thuần?)
CLASSIFY_CASES = [
    ("mạch trầm tế", True),
    ("mạch phù khẩn", True),
    ("mạch trì hoãn", True),
    ("Mạch Huyền Sác", True),          # hoa/thường không ảnh hưởng
    ("mạch tế sác vô lực", True),
    # KHÔNG phải mạch chẩn:
    ("phù", False),                     # phù nề (triệu chứng), không phải 'mạch phù'
    ("vô lực", False),                  # mệt/yếu, không phải mạch
    ("hoạt", False),                    # từ đơn, không có chữ mạch
    ("giãn tĩnh mạch chân", False),     # huyết quản, không phải mạch chẩn
    ("mạch máu bế tắc", False),         # huyết quản
    ("kinh mạch ứ trệ", False),         # kinh lạc
    ("rêu lưỡi vàng nhớt ; mạch hoạt sác", False),  # mở đầu bằng rêu -> giữ để khớp phần lưỡi
    ("đau đầu", False),
    ("", False),
]


def test_classify():
    ok = True
    for field, expect in CLASSIFY_CASES:
        got = F._is_pulse_field(field.lower())
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  [{status}] _is_pulse_field({field!r}) = {got} (kỳ vọng {expect})")
    return ok


def test_denominator():
    """Dòng giả lập: 3 field quan sát được + 2 field mạch. Khớp đúng 3 field quan sát ->
    ratio phải = 3/3 = 1.0 (không phải 3/5 = 0.6 như trước khi loại mạch)."""
    o = F.__new__(F)
    o.csv_rows = [{
        "benh_ly": "Bệnh Giả Lập",
        "hoi_chung": "Thể giả lập",
        "triệu_chứng": "lưỡi nhợt, rêu trắng mỏng, sợ lạnh, mạch trầm tế, mạch trì",
        "bai_thuoc": "", "vi_thuoc": "",
    }]
    o._build_symptom_idf()
    # Corpus giả lập 1 dòng khiến IDF = log(N/(1+df)) ra ÂM (rớt cổng peak_idf). Ép N lớn để IDF
    # dương như corpus thật (1038 dòng), cô lập đúng thứ đang test là MẪU SỐ (không phải cổng IDF).
    o._N_dis = 1038
    o._idf_min = 0.0
    o._idf_peak_min = 0.0

    res = F._find_matching_diseases(
        o,
        ["lưỡi nhợt", "rêu trắng mỏng", "sợ lạnh"],
        raw_user_text="lưỡi nhợt, rêu trắng mỏng, sợ lạnh",
    )
    ok = True
    if not res:
        print("  [FAIL] không tìm được ứng viên nào (kỳ vọng 1)")
        return False
    ratio = res[0]["ratio"]
    # Trước fix: 3/5 = 0.60. Sau fix: 3/3 = 1.00.
    status = "PASS" if abs(ratio - 1.0) < 1e-9 else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"  [{status}] ratio khớp = {ratio:.3f} (kỳ vọng 1.000 — mẫu số đã trừ 2 field mạch)")
    return ok


def run():
    print("── Phân loại field mạch ──")
    ok1 = test_classify()
    print("── Mẫu số tỷ lệ khớp (đã loại mạch) ──")
    ok2 = test_denominator()
    all_ok = ok1 and ok2
    print("\n" + ("✅ TẤT CẢ PASS" if all_ok else "❌ CÓ CA FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
