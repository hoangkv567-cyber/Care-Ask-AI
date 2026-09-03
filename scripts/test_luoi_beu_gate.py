# -*- coding: utf-8 -*-
"""Test [NHẤT QUÁN LƯỠI BỆU] _strip_luoi_beu_exterior_claims: ca NGOẠI CẢM + Bát Cương THUẦN THỰC
thì lưỡi bệu phải là 'dấu thể trạng nền', KHÔNG được quy cho bệnh cấp (kể cả quy gián tiếp qua
'phế khí uất trệ'). Ca có 'Hư' -> KHÔNG đụng (luật 13 xử).
Chạy: PYTHONIOENCODING=utf-8 python scripts/test_luoi_beu_gate.py"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from src.fusion_pipeline import TCMFusionPipeline as F

p = F.__new__(F)
fails = []
EXT = "Phong nhiệt phạm phế"          # core ngoại cảm
BC_THUC = "Biểu - Lý đồng bệnh - Nhiệt - Thực"
BC_HU = "Lý - Hàn - Hư"
NEN = "dấu thể trạng nền"


def check(desc, cond, got=None):
    print(f"  [{'OK ' if cond else 'FAIL'}] {desc}")
    if not cond:
        fails.append(desc)
        if got is not None:
            print(f"         -> {got!r}")


print("== PHẢI GỠ (ngoại cảm + thuần Thực, quy lưỡi bệu cho bệnh cấp) ==")
# REGRESSION THẬT: model lách danh sách tên-tà bằng 'phế khí uất trệ'
r1 = p._strip_luoi_beu_exterior_claims(
    "Ho cơn do phế mất tuyên giáng. Lưỡi bệu là do phế khí uất trệ, vận hóa bất thường, khiến thủy "
    "thấp ứ đọng, làm thân lưỡi căng phồng, sưng to.", EXT, BC_THUC)
check("quy GIÁN TIẾP qua 'phế khí uất trệ' (regression thật)", NEN in r1 and "uất trệ" not in r1, r1)

r2 = p._strip_luoi_beu_exterior_claims(
    "Lưỡi bệu do phong nhiệt làm rối loạn vận hóa của Tỳ.", EXT, BC_THUC)
check("quy ĐÍCH DANH ngoại tà (hành vi cũ vẫn chạy)", NEN in r2 and "phong nhiệt làm rối loạn" not in r2, r2)

r3 = p._strip_luoi_beu_exterior_claims(
    "Lưỡi bệu sinh ra bởi nhiệt tà hun đốt tân dịch.", EXT, BC_THUC)
check("quy nhân bằng 'sinh ra bởi'", NEN in r3, r3)

print("\n== PHẢI GIỮ (chống gỡ oan) ==")
# Câu sanctioned của chính stripper -> không được tự gỡ (chống vòng lặp/mất câu đúng)
s = ("Lưỡi bệu là dấu thể trạng nền có sẵn (Tỳ hư, thủy thấp chưa vận hóa), không thuộc bệnh cảnh "
     "ngoại cảm cấp lần này và nên theo dõi thêm.")
check("câu sanctioned KHÔNG tự bị gỡ", p._strip_luoi_beu_exterior_claims(s, EXT, BC_THUC).strip() == s,
      p._strip_luoi_beu_exterior_claims(s, EXT, BC_THUC))

# Ca có Hư -> luật 13 xử, stripper không đụng
h = "Lưỡi bệu là do Tỳ khí hư không vận hóa được thủy thấp."
check("Bát Cương có 'Hư' -> KHÔNG đụng (luật 13)",
      p._strip_luoi_beu_exterior_claims(h, "Tỳ khí hư", BC_HU).strip() == h)

# Core nội thương -> không đụng
check("core NỘI THƯƠNG -> KHÔNG đụng",
      p._strip_luoi_beu_exterior_claims(h, "Tỳ thận dương hư", BC_THUC).strip() == h)

# Không nhắc lưỡi bệu -> giữ nguyên
n = "Rêu trắng mỏng cho thấy tà mới xâm nhập, còn ở phần Biểu nông."
check("không có lưỡi bệu -> giữ nguyên",
      p._strip_luoi_beu_exterior_claims(n, EXT, BC_THUC).strip() == n)

# Câu mô tả TRUNG TÍNH (không quy nhân) -> giữ, không thay bừa
t = "Vọng chẩn ghi nhận lưỡi bệu và rêu trắng mỏng."
check("mô tả trung tính (không quy nhân) -> giữ nguyên",
      p._strip_luoi_beu_exterior_claims(t, EXT, BC_THUC).strip() == t,
      p._strip_luoi_beu_exterior_claims(t, EXT, BC_THUC))

print("\n" + ("✅ TẤT CẢ PASS" if not fails else f"❌ {len(fails)} FAIL: {fails}"))
sys.exit(1 if fails else 0)
