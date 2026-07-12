#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/resync_herb_strings.py — Đồng bộ CHÍNH XÁC chuỗi vị thuốc CSV -> graph cho
các cặp (bệnh × hội chứng) chỉ định (targets JSON [{benh, hc}]).

reconcile_kg_formulas.py so khớp bằng norm() (bỏ ngoặc) nên MÙ với lệch CHỈ-chú-thích
(vd graph 'Tri mẫu' vs CSV 'Tri mẫu (tẩm muối)' — norm bằng nhau -> bỏ qua). Script này
so khớp EXACT string: nếu bộ ViThuoc của node lệch bản CSV -> gỡ BAO_GỒM cũ, gắn lại
theo chuỗi CSV (giữ chú thích bào chế). CHỈ đụng node của các cặp target -> an toàn,
KHÔNG quét toàn graph (tránh nhiễu tokenization).

An toàn: MẶC ĐỊNH DRY-RUN. --apply mới ghi.
    python scripts/resync_herb_strings.py --targets data/_dedup_reconcile_targets.json
    python scripts/resync_herb_strings.py --targets data/_dedup_reconcile_targets.json --apply
"""
import argparse
import csv
import json
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "scripts"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from kg_maintenance import get_driver, run_read, run_write  # noqa: E402
from reconcile_kg_formulas import split_herbs  # dùng CHUNG bản đã giữ ngoặc bào chế

CSV = os.path.join(_ROOT, "data", "Medicine_clean.csv")


def main():
    ap = argparse.ArgumentParser(description="Resync EXACT chuỗi vị CSV->graph cho target. Mặc định DRY-RUN.")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    targets = json.load(open(args.targets, encoding="utf-8"))
    driver = get_driver()
    changed = 0
    try:
        for t in targets:
            b, h = t["benh"], t["hc"]
            csv_h = None
            for r in rows:
                if r["tên_bệnh"].strip() == b and r["hội_chứng"].strip() == h:
                    csv_h = split_herbs(r["vị_thuốc"])
                    break
            if not csv_h:
                continue
            nodes = run_read(driver, """
                MATCH (p:BaiThuoc {benh_ly:$b, hoi_chung:$h})
                OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
                RETURN elementId(p) AS eid, collect(DISTINCT v.name) AS hb
            """, b=b, h=h)
            for nd in nodes:
                if set(nd["hb"]) == set(csv_h):
                    continue
                miss = sorted(set(csv_h) - set(nd["hb"]))
                extra = sorted(set(nd["hb"]) - set(csv_h))
                print(f"[{b} × {h}] gắn: {miss}  |  gỡ: {extra}")
                changed += 1
                if args.apply:
                    run_write(driver, """
                        MATCH (p:BaiThuoc) WHERE elementId(p)=$e
                        OPTIONAL MATCH (p)-[r:BAO_GỒM]->(:ViThuoc) DELETE r
                        WITH p UNWIND $herbs AS hn
                        MERGE (v:ViThuoc {name:hn}) MERGE (p)-[:BAO_GỒM]->(v)
                    """, e=nd["eid"], herbs=csv_h)
        if not changed:
            print("Tất cả node target đã khớp EXACT CSV — không cần sửa.")
        elif args.apply:
            print(f"\nXONG: resync {changed} node. (ViThuoc plain cũ giữ nguyên cho bài khác dùng.)")
        else:
            print(f"\n(DRY-RUN — {changed} node sẽ resync. Thêm --apply.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
