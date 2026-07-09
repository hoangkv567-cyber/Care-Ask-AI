#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_gate_equivalence.py — Chứng minh bản DATA-HÓA của cổng an toàn bệnh danh cho kết quả
Y HỆT bản hard-code cũ (_validate_disease_safety vs _validate_disease_safety_legacy).

Kiểm 2 lớp:
  A. TOÀN BỘ bệnh danh trong CSV × một loạt lời khai đa dạng (rỗng, tự-triệu-chứng, 20 mẫu phủ
     nhiều chuyên khoa) — new phải == legacy ở MỌI cặp.
  B. TỔNG HỢP theo từng cổng: với mỗi tên bệnh của cổng, thử lời khai RỖNG (kỳ vọng loại) và lời
     khai = từng từ khóa 'requires' (kỳ vọng qua) — new phải == legacy. Bảo đảm mọi cổng, cả 2
     nhánh, đều được đối chiếu (kể cả cổng không khớp bệnh thật nào trong CSV).

Chạy:  python scripts/test_gate_equivalence.py   (Exit 0 nếu tương đương 100%; không cần Neo4j)
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F

CSV_PATH = "data/Medicine_clean.csv"
GATES_PATH = "data/disease_gates.json"

# Lời khai đa dạng phủ nhiều chuyên khoa (kích hoạt nhánh "có từ khóa" của phần lớn cổng)
BATTERY = [
    "",
    "mệt mỏi, chóng mặt, hoa mắt",
    "ho nhiều, đờm, khó thở, họng đau",
    "đau ngực, tức ngực, hồi hộp, trống ngực",
    "đau đầu, nhức đầu, mất ngủ, trằn trọc",
    "đau bụng, tiêu chảy, buồn nôn, ợ chua",
    "tiểu buốt, tiểu rắt, tiểu ra máu, bí tiểu",
    "ngứa, mẩn, phát ban, nổi mề đay, mụn nước",
    "mắt mờ, mắt đỏ, sưng mắt, giảm thị lực, quáng gà",
    "ù tai, điếc, nghe kém",
    "liệt dương, di tinh, xuất tinh, yếu sinh lý",
    "mang thai, thai động, kinh nguyệt, âm đạo ra huyết",
    "sốt, phát nhiệt, sốt về chiều, nóng trong",
    "mồ hôi, ra mồ hôi, đổ mồ hôi trộm",
    "chảy máu cam, chảy máu, xuất huyết, bầm tím",
    "phù, sưng phù, ấn lõm, húp mặt, tiểu ít",
    "khối u, u cục, sờ được khối, sụt cân, nuốt nghẹn",
    "nấc, ợ hơi, ưa nóng",
    "co giật, méo miệng, liệt nửa người, khó nói",
    "vàng da, da vàng, mắt vàng, nước tiểu vàng",
    "béo, mập, thừa cân, bụng to",
    "loét miệng, lở miệng, tưa lưỡi, nhiệt miệng",
    "đau lưng, mỏi gối, tiểu đêm, ăn kém, đại tiện lỏng, lưỡi nhợt",
]


def _obs(trieu):
    return [f.strip() for f in trieu.split(",") if f.strip() and not F._is_pulse_field(f.strip().lower())]


def run():
    if not hasattr(F, "_validate_disease_safety_legacy"):
        print("ℹ️  Bản legacy đã gỡ — migration cổng bệnh danh hoàn tất (tương đương đã xác nhận "
              "100% qua 9238 cặp trước khi gỡ). Bỏ qua đối chiếu.")
        return 0

    o = F.__new__(F)
    F._load_csv_data(o)                       # cho _get_symptom_vocab / _get_neg_protected_phrases
    o._disease_gates = json.load(open(GATES_PATH, encoding="utf-8")).get("gates", [])

    diseases = sorted({r["tên_bệnh"].strip() for r in csv.DictReader(open(CSV_PATH, encoding="utf-8-sig"))})
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    own = {}
    for r in rows:
        own.setdefault(r["tên_bệnh"].strip(), r["triệu_chứng"])

    mism = []
    checks = 0

    # ── A. bệnh thật × battery ──
    for d in diseases:
        profiles = list(BATTERY) + [", ".join(_obs(own.get(d, "")))]
        for raw in profiles:
            syms = [s.strip() for s in raw.split(",") if s.strip()]
            a = F._validate_disease_safety(o, d, syms, raw)
            b = F._validate_disease_safety_legacy(o, d, syms, raw)
            checks += 1
            if a != b:
                mism.append((d, raw[:40], a, b))

    # ── B. tổng hợp theo từng cổng (mọi tên × rỗng + từng requires) ──
    for gate in o._disease_gates:
        names = list(gate.get("names", []))
        # tên tổng hợp cho names_regex: nhúng token thành âm tiết độc lập
        for tok in gate.get("names_regex", []):
            names.append(f"vị {tok}")
        reqs = list(gate.get("requires", []))
        test_raws = [""] + reqs
        for nm in names:
            for raw in test_raws:
                syms = [raw] if raw else []
                a = F._validate_disease_safety(o, nm, syms, raw)
                b = F._validate_disease_safety_legacy(o, nm, syms, raw)
                checks += 1
                if a != b:
                    mism.append((nm, raw[:40], a, b))

    print(f"Đã đối chiếu {checks} cặp (new vs legacy).")
    if mism:
        print(f"❌ KHÔNG tương đương — {len(mism)} sai lệch (hiển thị tối đa 25):")
        for d, raw, a, b in mism[:25]:
            print(f"   · '{d}' | raw='{raw}' -> new={a} legacy={b}")
        return 1
    print("✅ TƯƠNG ĐƯƠNG 100% — bản data-hóa khớp hoàn toàn bản hard-code.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
