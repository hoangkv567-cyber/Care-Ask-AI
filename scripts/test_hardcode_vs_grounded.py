#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_hardcode_vs_grounded.py
So sánh: mỗi CASE HARDCODE trong _generate_explainable_answer ÉP hội chứng gì, VS grounded scoring
(_score_syndromes_grounded) tự SINH ra gì cho đúng bộ triệu chứng trigger của case đó.

Mục đích: quyết định GỠ hardcode có an toàn không.
  - Nếu grounded tái tạo được hội chứng bị ép (nằm trong top-3)  -> GỠ AN TOÀN.
  - Nếu hội chứng ép KHÔNG có trong KG, hoặc grounded không xếp nó lên -> GỠ SẼ REGRESS
    (phải enrich symptom cho hội chứng đó, hoặc giữ hardcode).

HAI CHẾ ĐỘ:
  (mặc định) OFFLINE: replicate grounded scoring từ data/Medicine_clean.csv (xấp xỉ graph — nhanh,
             không cần Neo4j). Dùng để phân tích SƠ BỘ.
  --graph  : gợi ý chạy grounded THẬT trên Neo4j cho từng case bằng test_grounded_scoring.py
             (in sẵn lệnh để bạn chạy chốt).

Chạy:  python scripts/test_hardcode_vs_grounded.py
"""
import csv
import os
import re
import sys
from collections import defaultdict

CSV_CANDIDATES = [os.getenv("TCM_CSV_PATH"), "data/Medicine_clean.csv"]

# ---- 13 CASE HARDCODE: (tên, triệu chứng trigger đại diện, hội chứng CỐT LÕI bị ép) ----
# Triệu chứng lấy đúng theo điều kiện trigger trong src/fusion_pipeline.py (_generate_explainable_answer).
CASES = [
    ("is_ngoai_cam_phong_han", ["sổ mũi", "ho", "sợ lạnh"], "Phong hàn phạm biểu"),
    ("is_ty_than_duong_hu", ["sợ lạnh", "tay chân lạnh", "quầng thâm dưới mắt", "rêu lưỡi trắng dày", "mặt nhợt nhạt", "đau đầu"], "Tỳ thận dương hư"),
    ("is_ty_than_duong_hu_tieu_chay", ["tiêu chảy", "phân lỏng nát", "sợ lạnh", "tay chân lạnh", "đau lưng", "đau bụng"], "Tỳ thận dương hư"),
    ("is_dam_thap_huyet_ap", ["nặng đầu", "chóng mặt", "béo phì", "rêu lưỡi dày nhớt"], "Đàm thấp"),
    ("is_khi_huyet_hu", ["mất ngủ", "hay quên", "mệt mỏi", "mặt nhợt nhạt", "rêu lưỡi trắng dày", "quầng thâm dưới mắt"], "Khí huyết đều hư"),
    ("is_dam_nhiet_uan_phe", ["đờm vàng dính", "ho", "sốt"], "Đàm nhiệt uẩn phế"),
    ("is_an_duong", ["mệt mỏi", "chóng mặt", "chán ăn", "vùng đỏ trên mặt", "mặt nhợt nhạt"], "Khí huyết đều hư"),
    ("is_ban_hu_tieu_thuc", ["mệt mỏi", "chóng mặt", "chán ăn", "mặt nhợt nhạt", "nốt mụn đỏ"], "Khí huyết đều hư"),
    ("is_vi_han", ["nấc", "ưa nóng", "miệng nhạt không khát", "mạch trì", "rêu trắng nhuận"], "Vị hàn"),
    ("is_am_hanh_dam_hach", ["đàm hạch", "vết răng", "rêu lưỡi trắng nhạt", "mạch nhu"], "Đờm Trọc Ngưng Kết"),
    ("is_be_kinh_phong_han", ["bế kinh", "bụng dưới đau lạnh", "tay chân lạnh", "mạch trầm khẩn"], "Phong hàn"),
]


def find_csv():
    return next((p for p in CSV_CANDIDATES if p and os.path.exists(p)), None)


def wb_match(term, text):
    """Khớp 'term' như một từ độc lập trong 'text' (ranh giới không phải chữ cái) — xấp xỉ
    _word_boundary_pattern của app (bắt cả khi term nằm trong node triệu chứng ghép)."""
    term = term.lower().strip()
    if not term:
        return False
    start = 0
    while True:
        i = text.find(term, start)
        if i < 0:
            return False
        before = text[i - 1] if i > 0 else ""
        after = text[i + len(term)] if i + len(term) < len(text) else ""
        if (not before or not before.isalpha()) and (not after or not after.isalpha()):
            return True
        start = i + 1


def is_dirty(name):
    # cùng bộ lọc với _score_syndromes_grounded: tên có số/ngoặc hoặc bắt đầu 'thể '
    return bool(re.search(r"[0-9(]", name)) or name.lower().startswith("thể ")


def build_index(rows):
    """syndrome -> set(symptom field lowercase) gộp từ mọi dòng có hội chứng đó."""
    idx = defaultdict(set)
    for r in rows:
        syn = r["hội_chứng"].strip()
        if not syn or is_dirty(syn):
            continue
        for f in r["triệu_chứng"].split(","):
            f = f.strip().lower()
            if f:
                idx[syn].add(f)
    return idx


def grounded_score(terms, idx, top_n=6):
    """Replicate _score_syndromes_grounded: score = matched + sum(1/df). df = #hội chứng khớp term."""
    terms = list(dict.fromkeys([t.lower().strip() for t in terms if t.strip()]))
    # với mỗi term: tập hội chứng khớp
    term_syns = {}
    for t in terms:
        syns = {syn for syn, symps in idx.items() if any(wb_match(t, s) for s in symps)}
        term_syns[t] = syns
    score = defaultdict(float)
    matched = defaultdict(int)
    for t in terms:
        syns = term_syns[t]
        df = len(syns)
        if df == 0:
            continue
        for syn in syns:
            score[syn] += 1.0 / df
            matched[syn] += 1
    ranked = sorted(((syn, matched[syn] + score[syn], matched[syn]) for syn in score),
                    key=lambda x: (-x[1], x[0]))
    strong = [x for x in ranked if x[2] >= 2]
    return (strong if strong else ranked)[:top_n]


def main():
    path = find_csv()
    if not path:
        print("Không tìm thấy CSV.")
        return 1
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    idx = build_index(rows)
    kg_syndromes = set(idx.keys())

    print("=" * 88)
    print("HARDCODE vs GROUNDED (replica offline từ CSV — SƠ BỘ; chốt bằng test_grounded_scoring.py trên Neo4j)")
    print("=" * 88)
    safe = risky = 0
    for name, terms, forced in CASES:
        top = grounded_score(terms, idx)
        top_names = [s for s, _sc, _m in top]
        in_kg = forced in kg_syndromes
        # forced có nằm trong top grounded? (khớp mềm theo tên chuẩn hoá)
        rank = next((i + 1 for i, s in enumerate(top_names) if s.strip().lower() == forced.strip().lower()), None)
        if rank is None:
            # thử khớp lỏng (chứa)
            rank = next((i + 1 for i, s in enumerate(top_names)
                         if forced.lower() in s.lower() or s.lower() in forced.lower()), None)
        verdict = "GỠ AN TOÀN" if (rank and rank <= 3) else "RỦI RO (grounded không tái tạo)"
        if rank and rank <= 3:
            safe += 1
        else:
            risky += 1
        print(f"\n[{name}]  ép='{forced}'  (có trong KG: {'CÓ' if in_kg else 'KHÔNG'})")
        print(f"   grounded top: {[f'{s}({m})' for s, _sc, m in top][:6]}")
        print(f"   -> hạng của '{forced}' trong grounded: {rank if rank else 'KHÔNG có'}   => {verdict}")
    print("\n" + "=" * 88)
    print(f"TỔNG: {safe} ca GỠ AN TOÀN (grounded tái tạo top-3) | {risky} ca RỦI RO (cần enrich/giữ).")
    print("CHỐT bằng graph thật:  python scripts/test_grounded_scoring.py \"<các triệu chứng trigger>\"")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
