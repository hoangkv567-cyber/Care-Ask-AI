#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/dedup_baithuoc.py — Gộp node BaiThuoc TRÙNG LẶP trên graph (Neo4j).

Graph có node BaiThuoc trùng (cùng benh_ly + hoi_chung + name, >1 node — do import
nhiều lần / biến thể chính tả vị thuốc) -> Mục 5 IN BÀI 2 LẦN. Giữ node có bộ vị
thuốc KHỚP CSV nhất (hiện trạng đúng), xóa các node stale/biến thể.

An toàn: MẶC ĐỊNH DRY-RUN. --apply mới xóa. Chỉ xóa khi cụm có >1 node CÙNG
(benh_ly, hoi_chung, name) -> luôn giữ >=1 node/cụm.
    python scripts/dedup_baithuoc.py            # dry-run
    python scripts/dedup_baithuoc.py --apply
"""
import argparse
import csv
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

CSV = os.path.join(_ROOT, "data", "Medicine_clean.csv")


def norm(s):
    return re.sub(r"\([^)]*\)", "", s or "").strip().lower()


def csv_herbs(rows, b, h, n):
    for r in rows:
        if r["tên_bệnh"].strip() == b and r["hội_chứng"].strip() == h and r["bài_thuốc"].strip() == n:
            return {norm(x) for x in re.split(r"[,;]", r.get("vị_thuốc", "")) if norm(x)}
    return set()


def main():
    ap = argparse.ArgumentParser(description="Gộp BaiThuoc trùng. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    driver = get_driver()
    try:
        clusters = run_read(driver, """
            MATCH (p:BaiThuoc) WITH p.benh_ly AS b, p.hoi_chung AS h, p.name AS n, count(*) AS c
            WHERE c > 1 RETURN b, h, n ORDER BY b, h
        """)
        print(f"Cụm BaiThuoc trùng: {len(clusters)}")
        total_del = 0
        for cl in clusters:
            b, h, n = cl["b"], cl["h"], cl["n"]
            cs = csv_herbs(rows, b, h, n)
            nodes = run_read(driver, """
                MATCH (p:BaiThuoc) WHERE p.benh_ly=$b AND p.hoi_chung=$h AND p.name=$n
                OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
                RETURN elementId(p) AS eid, collect(DISTINCT v.name) AS herbs
            """, b=b, h=h, n=n)
            scored = sorted(((len({norm(x) for x in nd["herbs"]} & cs), nd["eid"]) for nd in nodes),
                            reverse=True)
            keep_ov, _ = scored[0]
            drop = scored[1:]
            print(f"  {b} × {h} -> {n}: {len(nodes)} node; giữ overlap={keep_ov}/{len(cs)} CSV, xóa {len(drop)}")
            if args.apply:
                for _, eid in drop:
                    run_write(driver, "MATCH (p:BaiThuoc) WHERE elementId(p)=$e DETACH DELETE p", e=eid)
                    total_del += 1
        if args.apply:
            print(f"XONG: xóa {total_del} node trùng.")
        else:
            print("\n(DRY-RUN — thêm --apply để xóa.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
