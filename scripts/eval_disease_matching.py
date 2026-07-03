#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_disease_matching.py — BỘ ĐO (metrics) cho _find_matching_diseases, phủ TOÀN BỘ 992 bệnh.

Không cần Neo4j, không cần nhãn chuyên gia. Dùng chính dữ liệu CSV làm ground-truth self-consistency:
đưa triệu chứng của một bệnh vào -> matcher PHẢI tìm lại được bệnh đó (recall). Đây là lưới an toàn
để đo TRƯỚC/SAU khi đổi thuật toán khớp (vd thêm IDF symptom-weighting).

Chạy (từ thư mục gốc):  python scripts/eval_disease_matching.py
In các chỉ số. Dùng để so sánh baseline vs sau thay đổi.

Chỉ số:
  M1  Self@top1     : đưa TOÀN BỘ triệu chứng của bệnh -> bệnh đó có phải ứng viên #1? (sanity)
  M2  SpecRecall    : đưa CHỈ triệu chứng ĐẶC HIỆU (non-generic) của bệnh -> bệnh có trong ứng viên?
                      + phân bố hạng (top1/top3). Đây là chỉ số NHẠY nhất với thay đổi thuật toán.
  M3  OverMatch     : đưa toàn triệu chứng GENERIC -> số ứng viên (muốn = 0, đo over-matching).
  M4  Clinical      : các ca forbid/expect lâm sàng (đồng bộ test_disease_matching.py).
"""
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def fields(row):
    return [s.strip() for s in row.get("triệu_chứng", "").split(",") if s.strip()]


def specific_only(flds):
    return [x for x in flds if not F._is_generic_symptom(x.lower())]


def run():
    o = F.__new__(F)
    F._load_csv_data(o)
    rows = getattr(o, "csv_rows", None)
    if not rows:
        print("LỖI: không tải được CSV.")
        return 1

    # ---------- M1 + M2 ----------
    m1_hit = m1_tot = 0
    m2_in = m2_top1 = m2_top3 = m2_tot = 0
    m2_miss_examples = []
    ranks = []
    for row in rows:
        flds = fields(row)
        if len(flds) < 2:
            continue
        disease = row["benh_ly"].strip().lower()

        # M1: full symptoms -> disease should be #1
        res = F._find_matching_diseases(o, flds, raw_user_text=", ".join(flds))
        names = [m["benh_ly"].strip().lower() for m in res]
        m1_tot += 1
        if names and names[0] == disease:
            m1_hit += 1

        # M2: specific-only symptoms -> recall of the disease
        spec = specific_only(flds)
        if len(spec) >= 2:
            m2_tot += 1
            res2 = F._find_matching_diseases(o, spec, raw_user_text=", ".join(spec))
            names2 = [m["benh_ly"].strip().lower() for m in res2]
            if disease in names2:
                m2_in += 1
                r = names2.index(disease) + 1
                ranks.append(r)
                if r == 1:
                    m2_top1 += 1
                if r <= 3:
                    m2_top3 += 1
            elif len(m2_miss_examples) < 8:
                m2_miss_examples.append((row["benh_ly"], spec[:4]))

    # ---------- M3 over-match ----------
    generic_probes = [
        ["mệt mỏi", "mặt nhợt nhạt", "rêu trắng", "lưỡi hồng"],
        ["mệt mỏi", "chóng mặt", "ăn kém"],
        ["mặt vàng", "rêu lưỡi trắng", "mệt mỏi"],
        ["lưỡi nhợt", "mạch tế", "mệt mỏi"],
    ]
    m3 = []
    for p in generic_probes:
        res = F._find_matching_diseases(o, p, raw_user_text=", ".join(p))
        m3.append(len(res))

    # ---------- M4 clinical ----------
    clinical = [
        ("Hô hấp (không đau ngực)", ["ho nhiều", "khó thở", "mệt mỏi", "nốt mụn đỏ", "rêu trắng", "lưỡi hồng", "mặt nhợt nhạt"],
         "ho nhiều, khó thở, mệt mỏi",
         ["Áp xe gan", "Bạch tiển (nấm da)", "Dương nuy", "Hung tý", "Động thai"], ["Mạn tính tắc nghẽn phế bệnh"]),
        ("Đau ngực (Hung tý hợp lệ)", ["đau thắt ngực", "tức ngực", "khó thở", "hồi hộp", "mệt mỏi"],
         "đau ngực dữ dội, tức ngực", [], ["Hung tý"]),
        ("Nam khoa (liệt dương)", ["liệt dương", "đau lưng mỏi gối", "tiểu đêm", "mệt mỏi"],
         "liệt dương, yếu sinh lý", [], ["Dương nuy"]),
        ("Thai phụ (động thai)", ["động thai", "đau bụng", "ra huyết"],
         "đang mang thai, đau bụng, ra huyết", [], ["Động thai"]),
    ]
    m4_pass = m4_tot = 0
    m4_fail_detail = []
    for name, syms, raw, forbid, expect in clinical:
        res = F._find_matching_diseases(o, syms, raw_user_text=raw)
        got = [m["benh_ly"].strip().lower() for m in res]
        bad = [b for b in forbid if b.strip().lower() in got]
        miss = [e for e in expect if e.strip().lower() not in got]
        m4_tot += 1
        if not bad and not miss:
            m4_pass += 1
        else:
            m4_fail_detail.append((name, bad, miss))

    # ---------- IN KẾT QUẢ ----------
    print("=" * 70)
    print("BỘ ĐO KHỚP BỆNH DANH  (baseline / so sánh trước-sau IDF)")
    print("=" * 70)
    print(f"M1 Self@top1  : {m1_hit}/{m1_tot} = {100*m1_hit/max(1,m1_tot):.1f}%  "
          f"(đưa full triệu chứng -> bệnh là ứng viên #1)")
    print(f"M2 SpecRecall : bệnh có specific>=2: {m2_tot}")
    if m2_tot:
        print(f"     - có trong ứng viên : {m2_in}/{m2_tot} = {100*m2_in/m2_tot:.1f}%")
        print(f"     - top-3             : {m2_top3}/{m2_tot} = {100*m2_top3/m2_tot:.1f}%")
        print(f"     - top-1             : {m2_top1}/{m2_tot} = {100*m2_top1/m2_tot:.1f}%")
        print(f"     - hạng trung vị     : {median(ranks) if ranks else 'n/a'}")
        print(f"     - ví dụ MISS (bệnh không tự khớp từ triệu chứng đặc hiệu):")
        for b, s in m2_miss_examples:
            print(f"         · {b}  <- {s}")
    print(f"M3 OverMatch  : số ứng viên cho input toàn generic (muốn ~0): {m3}")
    print(f"M4 Clinical   : {m4_pass}/{m4_tot} PASS")
    for name, bad, miss in m4_fail_detail:
        print(f"     ✗ {name}: cấm-mà-có={bad} kỳ-vọng-mà-thiếu={miss}")
    print("=" * 70)
    return 0 if (m4_pass == m4_tot) else 1


if __name__ == "__main__":
    sys.exit(run())
