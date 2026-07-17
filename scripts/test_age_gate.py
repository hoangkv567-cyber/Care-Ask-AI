# -*- coding: utf-8 -*-
"""Test CỔNG TUỔI (data/disease_age.json + _age_conflict): bệnh nhi bị loại cho người lớn và
ngược lại, NO-OP khi không nhập tuổi. Chạy: PYTHONIOENCODING=utf-8 python scripts/test_age_gate.py"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from src.fusion_pipeline import TCMFusionPipeline as F

p = F.__new__(F); p._load_csv_data()
fails = []


def check(desc, cond):
    print(f"  [{'OK ' if cond else 'FAIL'}] {desc}")
    if not cond:
        fails.append(desc)


print("== _age_conflict (đơn vị) ==")
# Bách nhật khái = pediatric (loại khi >=16)
check("Bách nhật khái loại cho 40 tuổi", p._age_conflict("Bách nhật khái", 40) is True)
check("Bách nhật khái loại cho 16 tuổi (ngưỡng)", p._age_conflict("Bách nhật khái", 16) is True)
check("Bách nhật khái GIỮ cho 15 tuổi", p._age_conflict("Bách nhật khái", 15) is False)
check("Bách nhật khái GIỮ cho trẻ 5 tuổi", p._age_conflict("Bách nhật khái", 5) is False)
check("Cam tích (nhi) loại cho người lớn", p._age_conflict("Cam tích", 30) is True)
check("Nhi tiết tả (nhi) loại cho người lớn", p._age_conflict("Nhi tiết tả", 30) is True)
# Không nhập tuổi -> không lọc
check("Bách nhật khái KHÔNG lọc khi tuổi None", p._age_conflict("Bách nhật khái", None) is False)
# Bệnh không đặc thù tuổi -> không lọc
check("Viêm phế quản KHÔNG lọc (any) cho mọi tuổi", p._age_conflict("Viêm phế quản", 40) is False)
check("Tiêu khát KHÔNG lọc (any) cho trẻ", p._age_conflict("Tiêu khát", 8) is False)
# Bệnh ADULT (loại khi < 16)
check("Liệt dương loại cho trẻ 12 tuổi", p._age_conflict("Liệt dương", 12) is True)
check("Liệt dương GIỮ cho người lớn 45 tuổi", p._age_conflict("Liệt dương", 45) is False)
# Bệnh MATURE (mãn kinh) — loại khi < mature_min_age (40), kể cả thanh niên >=16
check("Canh niên kỳ (mãn kinh) loại cho trẻ 10 tuổi", p._age_conflict("Canh niên kỳ hội chứng", 10) is True)
check("Canh niên kỳ loại cho nữ 22 tuổi (< 40)", p._age_conflict("Canh niên kỳ hội chứng", 22) is True)
check("Canh niên kỳ loại cho 39 tuổi (ngưỡng dưới)", p._age_conflict("Canh niên kỳ hội chứng", 39) is True)
check("Canh niên kỳ GIỮ cho 40 tuổi (ngưỡng)", p._age_conflict("Canh niên kỳ hội chứng", 40) is False)
check("Canh niên kỳ GIỮ cho 52 tuổi", p._age_conflict("Canh niên kỳ hội chứng", 52) is False)
check("Canh niên kỳ KHÔNG lọc khi tuổi None", p._age_conflict("Canh niên kỳ hội chứng", None) is False)
check("Parkinson loại cho trẻ", p._age_conflict("Parkinson", 12) is True)
check("COPD loại cho trẻ", p._age_conflict("Mạn tính tắc nghẽn phế bệnh", 12) is True)
check("Liệt dương KHÔNG lọc khi tuổi None", p._age_conflict("Liệt dương", None) is False)
# Di tinh/Mộng tinh CỐ Ý 'any' (hậu dậy thì) — không được loại cho thiếu niên
check("Di tinh KHÔNG lọc (any) cho 14 tuổi", p._age_conflict("Di tinh", 14) is False)
# Kinh nguyệt CỐ Ý 'any' (thiếu nữ có kinh) — không được loại
check("Thống kinh KHÔNG lọc (any) cho 13 tuổi", p._age_conflict("Thống kinh", 13) is False)

print("\n== Tích hợp _find_matching_diseases (ca ho + ho cơn) ==")
base = "ho ngày càng nặng, đờm mầu vàng, ho đàm dính, khô miệng, ho nhiều, sợ lạnh, ho cơn"


def diseases(txt):
    p._patient_age = None
    if hasattr(p, "_disease_age_map"):
        del p._disease_age_map
    return {m["benh_ly"] for m in p._find_matching_diseases([s.strip() for s in txt.split(",")], txt)}


check("40 tuổi -> LOẠI Bách nhật khái", "Bách nhật khái" not in diseases(base + ", 40 tuổi, nam giới"))
check("40 tuổi -> GIỮ Viêm phế quản", "Viêm phế quản" in diseases(base + ", 40 tuổi, nam giới"))
check("10 tuổi -> GIỮ Bách nhật khái", "Bách nhật khái" in diseases(base + ", 10 tuổi"))
check("Không nhập tuổi -> GIỮ Bách nhật khái (NO-OP)", "Bách nhật khái" in diseases(base))

_male = "liệt dương, di tinh, lưng gối mỏi, mệt mỏi, sợ lạnh"
_adult_dz = lambda s: {d for d in s if d in ("Liệt dương", "Dương nuy", "Tảo tiết")}
check("12 tuổi (nam) -> LOẠI Liệt dương/Dương nuy (bệnh người lớn)",
      not _adult_dz(diseases(_male + ", 12 tuổi, nam giới")))
check("45 tuổi (nam) -> GIỮ bệnh nam khoa người lớn",
      bool(_adult_dz(diseases(_male + ", 45 tuổi, nam giới"))))
check("12 tuổi -> GIỮ Di tinh (any, hậu dậy thì)",
      "Di tinh" in diseases(_male + ", 12 tuổi, nam giới"))

print("\n" + ("✅ TẤT CẢ PASS" if not fails else f"❌ {len(fails)} FAIL: {fails}"))
sys.exit(1 if fails else 0)
