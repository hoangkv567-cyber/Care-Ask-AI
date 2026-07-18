#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/fix_duonghu_han_tag.py — Bù tag Bát Cương 'Hàn' cho các thể DƯƠNG HƯ trên Neo4j.

VÌ SAO: dương hư theo định nghĩa là DƯƠNG KHÍ BẤT TÚC -> không ôn ấm được -> SINH NỘI HÀN. Trên
graph, 'Thận dương hư' có đủ ['Hàn','Hư','Lý'] nhưng 20 thể dương hư khác chỉ có ['Hư','Lý'] —
thiếu trục Hàn. Hệ quả đo được: ca 'Ách nghịch × Tỳ thận dương hư' (nấc hư hàn, kê Phụ tử/Can
khương/Đinh hương) ra Bát Cương KHÔNG có Hàn, tức Mục 2 mâu thuẫn với chính bài thuốc ôn dương.

PHẠM VI (cố ý HẸP):
  - CHỈ node tên khớp mẫu dương-hư: dương (khí)? (hư|suy|nhược|thoát|bất túc); KHÔNG lấy 'âm hư
    dương cang/xung/thịnh' (âm hư dương vượng — KHÔNG phải hàn).
  - BẮT BUỘC node đã có tag 'Hư' (đã được xác lập là thể hư) — tránh gán mù vào node chưa phân loại.
  - LOẠI node có tag 'Nhiệt' (vd 'Dương hư phát nhiệt' — dương hư phù việt sinh nhiệt): thêm Hàn ở
    đó sẽ dựng Hàn+Nhiệt, phải do người quyết, không tự động.

AN TOÀN: mặc định DRY-RUN. Cạnh tạo mới gắn r.nguon='duonghu_han_fix' -> gỡ sạch bằng --undo
(nhãn riêng của script NÀY, không đụng cạnh sẵn có).

Chạy:
    python scripts/fix_duonghu_han_tag.py            # dry-run
    python scripts/fix_duonghu_han_tag.py --apply
    python scripts/fix_duonghu_han_tag.py --undo
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

NGUON = "duonghu_han_fix"
YANG_RE = re.compile(r'\bdương\s+(?:khí\s+)?(?:hư|suy|nhược|thoát|bất\s*túc)', re.IGNORECASE)


def _load_env():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="ghi graph thật (mặc định dry-run)")
    ap.add_argument("--undo", action="store_true", help="gỡ các cạnh script này đã tạo")
    args = ap.parse_args()

    _load_env()
    import warnings
    warnings.filterwarnings("ignore")
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    if not uri:
        print("LỖI: thiếu NEO4J_URI trong .env")
        return 1
    drv = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"),
                                          os.getenv("NEO4J_PASSWORD")))

    if args.undo:
        with drv.session() as s:
            n = s.run("MATCH (:HoiChung)-[r:CÓ_TÍNH_CHẤT]->(:BatCuong) WHERE r.nguon=$src "
                      "DELETE r RETURN count(*) AS n", src=NGUON).single()["n"]
        print(f"Đã gỡ {n} cạnh do script này tạo.")
        return 0

    with drv.session() as s:
        rows = s.run("""MATCH (h:HoiChung) OPTIONAL MATCH (h)-[]->(b:BatCuong)
                        RETURN h.name AS n, collect(DISTINCT b.name) AS tags""").data()

    todo, skipped = [], []
    for r in rows:
        name, tags = r["n"] or "", r["tags"] or []
        if not YANG_RE.search(name.lower()) or "âm" in name.lower():
            continue
        if "Hàn" in tags:
            continue
        if "Nhiệt" in tags:
            skipped.append((name, tags, "có tag Nhiệt — cần người quyết"))
        elif "Hư" not in tags:
            skipped.append((name, tags, "chưa có tag Hư — không gán mù"))
        else:
            todo.append((name, tags))

    print(f"Thể dương hư sẽ BÙ tag Hàn: {len(todo)} | bỏ qua: {len(skipped)}\n")
    for n, t in todo:
        print(f"  [FIX] {n:<40} {t} -> + Hàn")
    if skipped:
        print()
        for n, t, why in skipped:
            print(f"  [BỎ QUA] {n:<36} {t}  ({why})")

    if not args.apply:
        print("\n(DRY-RUN — thêm --apply để ghi. Gỡ lại: --undo)")
        return 0

    with drv.session() as s:
        for n, _t in todo:
            s.run("""MATCH (h:HoiChung {name:$n})
                     MERGE (b:BatCuong {name:'Hàn'})
                     MERGE (h)-[r:CÓ_TÍNH_CHẤT]->(b)
                     ON CREATE SET r.nguon=$src""", n=n, src=NGUON)
    print(f"\nĐã bù tag Hàn cho {len(todo)} thể dương hư. Gỡ lại: --undo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
