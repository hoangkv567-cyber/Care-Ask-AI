#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_gold.py — Chạy BỘ CA CHUẨN (data/gold_cases.json) trên tầng khớp bệnh danh.

Phạm vi: Vấn + Vọng (lưỡi/mặt), KHÔNG mạch — đúng năng lực hệ. Chỉ dùng CSV (self.csv_rows),
KHÔNG cần Neo4j, KHÔNG cần LLM. Dùng làm CỔNG CI chống hồi quy cho _find_matching_diseases.

Đo:
  - Ca 'recall'  : bệnh kỳ vọng có nằm trong top-3 ứng viên không -> recall@3.
  - Ca 'discrim' : bệnh trong forbid_disease KHÔNG được xuất hiện (0 vi phạm);
                   bệnh trong expect_disease_any phải xuất hiện.

Cổng PASS:
  - recall@3 >= GOLD_MIN_RECALL (mặc định lấy từ baseline lần đầu, chỉ báo động khi TỤT).
  - 0 vi phạm forbid.
  - mọi expect_disease_any thỏa.

Chạy:  python scripts/eval_gold.py [--min-recall 0.6]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F

GOLD_PATH = "data/gold_cases.json"
DEFAULT_MIN_RECALL = 0.60   # ngưỡng bảo thủ trên dữ liệu hiện tại (self-recall ~77%)
TOP_K = 3


def norm(s):
    return (s or "").strip().lower()


def run(min_recall):
    o = F.__new__(F)
    F._load_csv_data(o)
    if not getattr(o, "csv_rows", None):
        print("LỖI: không tải được CSV.")
        return 1

    gold = json.load(open(GOLD_PATH, encoding="utf-8"))
    cases = gold.get("cases", [])

    recall_total = recall_hit = 0
    forbid_violations = []
    expect_miss = []
    recall_miss = []

    for c in cases:
        res = F._find_matching_diseases(o, c["symptoms"], raw_user_text=c.get("raw", ""))
        names = [norm(m["benh_ly"]) for m in res]
        top = [norm(m["benh_ly"]) for m in res[:TOP_K]]

        if c["type"] == "recall":
            recall_total += 1
            if norm(c["expect_disease"]) in top:
                recall_hit += 1
            else:
                recall_miss.append((c["id"], c["expect_disease"], [m["benh_ly"] for m in res[:TOP_K]]))
        elif c["type"] == "discrim":
            for bad in c.get("forbid_disease", []):
                if norm(bad) in names:
                    forbid_violations.append((c["id"], bad, c.get("specialty", "")))
            for good in c.get("expect_disease_any", []):
                if norm(good) not in names:
                    expect_miss.append((c["id"], good, c.get("specialty", "")))

    recall = (recall_hit / recall_total) if recall_total else 1.0

    print("=" * 66)
    print(f"BỘ CA CHUẨN (gold)  — {len(cases)} ca  (top-k={TOP_K})")
    print("=" * 66)
    print(f"Recall@{TOP_K} (ca recall): {recall_hit}/{recall_total} = {recall:.1%}  (ngưỡng {min_recall:.0%})")
    if recall_miss:
        print(f"  Ca recall TRƯỢT ({len(recall_miss)}):")
        for cid, exp, got in recall_miss:
            print(f"    · [{cid}] cần '{exp}' — top3: {got}")
    print(f"Vi phạm forbid (phải = 0): {len(forbid_violations)}")
    for cid, bad, spec in forbid_violations:
        print(f"    ✗ [{cid}] ({spec}) XUẤT HIỆN bệnh cấm '{bad}'")
    print(f"Thiếu expect (phải = 0): {len(expect_miss)}")
    for cid, good, spec in expect_miss:
        print(f"    ✗ [{cid}] ({spec}) THIẾU bệnh kỳ vọng '{good}'")

    ok = (recall >= min_recall) and not forbid_violations and not expect_miss
    print("=" * 66)
    print("✅ GOLD PASS" if ok else "❌ GOLD FAIL — kiểm tra thay đổi tầng khớp bệnh danh")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    args = ap.parse_args()
    sys.exit(run(args.min_recall))
