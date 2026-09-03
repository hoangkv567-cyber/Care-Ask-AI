# -*- coding: utf-8 -*-
"""Test [CHỐNG NHẠI RUBRIC] _strip_meta_commentary: gỡ câu model TỰ CHẤM ĐIỂM (nhại chữ luật 7)
nhưng KHÔNG đụng câu lâm sàng hợp lệ. Chạy: PYTHONIOENCODING=utf-8 python scripts/test_meta_commentary.py"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from src.fusion_pipeline import TCMFusionPipeline as F

p = F.__new__(F)
fails = []


def check(desc, cond, got=None):
    print(f"  [{'OK ' if cond else 'FAIL'}] {desc}")
    if not cond:
        fails.append(desc)
        if got is not None:
            print(f"         -> {got!r}")


print("== PHẢI GỠ (câu META nhại rubric) ==")
_STRIP = [
    ("câu META ca thật (ho ra máu)",
     "Ho cơn là do phong nhiệt làm rối loạn khí cơ. Tất cả các triệu chứng này đều được giải thích "
     "một cách mạch lạc, liên kết chặt chẽ với cơ chế bệnh sinh của Phong nhiệt phạm phế, không có "
     "dấu hiệu nào bị bỏ sót hay tự ý thêm vào.",
     "Ho cơn là do phong nhiệt làm rối loạn khí cơ."),
    ("tuyên bố tuân luật",
     "Tỳ khí hư sinh mệt mỏi. Bài biện luận đã tuân thủ đúng các luật và không bỏ sót triệu chứng nào.",
     "Tỳ khí hư sinh mệt mỏi."),
    ("tự khen 'như một danh y'",
     "Can khí uất kết gây hiếp thống. Phân tích được trình bày như một danh y thực thụ.",
     "Can khí uất kết gây hiếp thống."),
    ("khoe không tự ý thêm",
     "Thấp nhiệt hạ chú gây tiểu buốt. Tôi không tự ý thêm vào triệu chứng nào ngoài danh sách.",
     "Thấp nhiệt hạ chú gây tiểu buốt."),
]
for desc, inp, expect in _STRIP:
    out = p._strip_meta_commentary(inp)
    check(desc, out.strip() == expect, out)

print("\n== PHẢI GIỮ NGUYÊN (câu lâm sàng hợp lệ — chống gỡ oan) ==")
_KEEP = [
    ("tổng kết CƠ CHẾ 'tất cả triệu chứng đều DO...' (KHÔNG phải meta)",
     "Tất cả các triệu chứng cấp tính này đều do Thấp nhiệt ứ đọng ở hạ tiêu, gây rối loạn chức năng "
     "đại tràng và tổn thương tại chỗ."),
    ("'giải thích' dùng theo nghĩa lâm sàng",
     "Điều này giải thích vì sao bệnh nhân ho nhiều về đêm và đờm có màu vàng."),
    ("dấu nền + theo dõi thêm (câu của stripper lưỡi bệu)",
     "Lưỡi bệu là dấu thể trạng nền có sẵn, không thuộc bệnh cảnh ngoại cảm cấp lần này và nên theo dõi thêm."),
    ("mạch LẠC theo nghĩa kinh mạch/huyết mạch",
     "Nhiệt tà xâm nhập mạch lạc của Phế, làm huyết đi sai đường mà thành khái huyết."),
    ("'bỏ' theo nghĩa lâm sàng, không phải 'bỏ sót'",
     "Bệnh nhân bỏ bữa nhiều ngày khiến Tỳ vị hư tổn, vận hóa bất lực."),
    ("biện luận dài không có marker meta",
     "Phong nhiệt phạm phế làm Phế mất tuyên giáng, sinh ho cơn; nhiệt hun đốt tân dịch hóa đàm vàng; "
     "nhiệt bức huyết vong hành nên đờm lẫn máu tươi."),
]
for desc, inp in _KEEP:
    out = p._strip_meta_commentary(inp)
    check(desc, out.strip() == inp.strip(), out)

print("\n== Không đụng text rỗng/None ==")
check("text rỗng", p._strip_meta_commentary("") == "")
check("text None", p._strip_meta_commentary(None) is None)

print("\n" + ("✅ TẤT CẢ PASS" if not fails else f"❌ {len(fails)} FAIL: {fails}"))
sys.exit(1 if fails else 0)
