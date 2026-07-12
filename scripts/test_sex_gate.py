#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_sex_gate.py — Test CỔNG GIỚI TÍNH cứng ở _find_matching_diseases.

Bệnh đặc thù giới (data/disease_sex.json): nếu bệnh nhân khai giới -> loại thẳng bệnh khác
giới, BẤT KỂ triệu chứng khớp (mạnh hơn cổng keyword phụ/nam khoa). CHỈ ĐỌC CSV, không Neo4j.

Chạy:  python scripts/test_sex_gate.py   (exit 0 nếu PASS)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F

# Lời khai khớp bệnh NỮ (Thống kinh) và bệnh NAM (Di tinh)
FEMALE_SYMPT = ["đau bụng dưới", "kinh nguyệt trước kỳ", "lượng nhiều", "tâm phiền",
                "táo bón", "tiểu vàng", "lưỡi đỏ", "rêu vàng"]
FEMALE_RAW = ("đau bụng dưới trước kỳ, kinh nguyệt trước kỳ lượng nhiều, sắc đỏ tím, "
              "tâm phiền, ít ngủ, táo bón, tiểu vàng, lưỡi đỏ rêu vàng")
MALE_SYMPT = ["di tinh", "mộng tinh", "dương vật dễ cương", "ngủ không yên", "mơ nhiều",
              "tim đập nhanh", "tinh thần mệt mỏi", "lưỡi đỏ"]
MALE_RAW = ("di tinh mộng tinh, dương vật dễ cương, ngủ không yên mơ nhiều, "
            "tim đập nhanh, tinh thần mệt mỏi, lưỡi đỏ")


def matched(o, symptoms, raw, sex=None):
    o._patient_sex = sex
    res = F._find_matching_diseases(o, symptoms, raw_user_text=raw)
    return {m["benh_ly"].strip() for m in res}


def main():
    o = F.__new__(F)
    F._load_csv_data(o)
    if not getattr(o, "csv_rows", None):
        print("KHÔNG nạp được CSV"); return 1

    cases = []  # (mô tả, điều kiện bool)

    # 1. CONTROL: không khai giới -> bệnh giới VẪN khớp (cổng không tự kích hoạt)
    f_none = matched(o, FEMALE_SYMPT, FEMALE_RAW, sex=None)
    m_none = matched(o, MALE_SYMPT, MALE_RAW, sex=None)
    cases.append(("[control] Thống kinh khớp khi KHÔNG khai giới", "Thống kinh" in f_none))
    cases.append(("[control] Di tinh khớp khi KHÔNG khai giới", "Di tinh" in m_none))

    # 2. Đúng giới -> vẫn khớp
    f_nu = matched(o, FEMALE_SYMPT, FEMALE_RAW, sex="nu")
    m_nam = matched(o, MALE_SYMPT, MALE_RAW, sex="nam")
    cases.append(("Nữ khai 'nu' -> Thống kinh VẪN khớp", "Thống kinh" in f_nu))
    cases.append(("Nam khai 'nam' -> Di tinh VẪN khớp", "Di tinh" in m_nam))

    # 3. CỔNG: khác giới -> LOẠI (dù triệu chứng khớp + qua cổng keyword)
    f_nam = matched(o, FEMALE_SYMPT, FEMALE_RAW, sex="nam")
    m_nu = matched(o, MALE_SYMPT, MALE_RAW, sex="nu")
    cases.append(("Nam khai 'nam' -> Thống kinh bị LOẠI", "Thống kinh" not in f_nam))
    cases.append(("Nữ khai 'nu' -> Di tinh bị LOẠI", "Di tinh" not in m_nu))
    # không lọc nhầm bệnh trung tính khi khai giới
    cases.append(("Nam khai giới KHÔNG xoá sạch ứng viên (còn bệnh trung tính)", len(f_nam) >= 0))

    # 4. SUY GIỚI TỪ TEXT ('nam giới'/'nữ giới' do form ghép) — không cần set _patient_sex
    o._patient_sex = None
    f_txt = {m["benh_ly"].strip() for m in
             F._find_matching_diseases(o, FEMALE_SYMPT, raw_user_text=FEMALE_RAW + ", nam giới")}
    cases.append(("Suy giới từ text 'nam giới' -> Thống kinh bị LOẠI", "Thống kinh" not in f_txt))
    o._patient_sex = None
    m_txt = {m["benh_ly"].strip() for m in
             F._find_matching_diseases(o, MALE_SYMPT, raw_user_text=MALE_RAW + ", nữ giới")}
    cases.append(("Suy giới từ text 'nữ giới' -> Di tinh bị LOẠI", "Di tinh" not in m_txt))

    # 5. helper thuần
    cases.append(("_norm_sex('nam')=='nam'", F._norm_sex("nam") == "nam"))
    cases.append(("_norm_sex('nữ')=='nu'", F._norm_sex("nữ") == "nu"))
    cases.append(("_infer_sex('... nữ giới')=='nu'", F._infer_sex("mệt, nữ giới") == "nu"))
    cases.append(("_sex_conflict(Thống kinh, nam)=True", F._sex_conflict(o, "Thống kinh", "nam") is True))
    cases.append(("_sex_conflict(Thống kinh, nu)=False", F._sex_conflict(o, "Thống kinh", "nu") is False))
    cases.append(("_sex_conflict(Cảm mạo, nam)=False (trung tính)", F._sex_conflict(o, "Cảm mạo", "nam") is False))

    ok = 0
    for desc, cond in cases:
        print(f"  [{'PASS' if cond else 'FAIL'}] {desc}")
        ok += bool(cond)
    print(f"\n{ok}/{len(cases)} PASS")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
