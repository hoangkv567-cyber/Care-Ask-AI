#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/reconcile_herb_base.py — Dọn ViThuoc BẨN trên graph do import CSV rối (ghi chú đuôi
bài-thay-thế/gia-giảm bị comma-split thành node rác 'Sa sâm)', 'Sinh Cam thảo. (Hoặc dùng...').

Với MỖI dòng CSV: base = _dedupe_herbs(_strip_herb_tail_notes(vị_thuốc)) = list vị SẠCH của bài
chính. Tìm BaiThuoc graph khớp (benh_ly, hoi_chung); nếu bộ ViThuoc (chuẩn hoá) LỆCH base -> gỡ
BAO_GỒM cũ, gắn lại base. Cũng dọn TÊN BaiThuoc có '(hoặc...)'/đuôi ghi chú.

An toàn: MẶC ĐỊNH DRY-RUN. --apply mới ghi (+ rollback JSON). Chỉ đụng node LỆCH.
    python scripts/reconcile_herb_base.py            # dry-run
    python scripts/reconcile_herb_base.py --apply
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
from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402

CSV = os.path.join(_ROOT, "data", "Medicine_clean.csv")
ROLLBACK = os.path.join(_ROOT, "data", "herb_base_reconcile_rollback.json")


def norm(s):
    s = re.sub(r"\([^)]*\)", "", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def base_herbs(vi):
    """List vị base SẠCH (đã cắt ghi chú đuôi + khử trùng)."""
    clean = F._dedupe_herbs(vi or "")
    return [h.strip() for h in clean.split(",") if h.strip()]


def clean_bai_name(name):
    """Bỏ đuôi '(hoặc ...)'/'. (...)'/ghi chú khỏi tên bài."""
    n = re.split(r'\s*\(\s*hoặc\b|\.\s*\(', name or "", maxsplit=1)[0]
    return n.strip().rstrip(".,; ").strip() or (name or "").strip()


def is_garbage_vithuoc(t):
    """ViThuoc RÁC do import blob: ngoặc LỆCH (bị comma-split), có '.'/';', hoặc chứa từ-ghi-chú.
    ViThuoc hợp lệ (kể cả bào chế 'Tri mẫu (tẩm muối)') có ngoặc CÂN BẰNG + không '.'/note-word."""
    if not t:
        return False
    if t.count("(") != t.count(")"):
        return True
    if "." in t or ";" in t:
        return True
    tl = t.lower()
    return any(k in tl for k in ("nếu ", "hoặc dùng", "kết hợp", "gia:", "tham khảo",
                                 "thể thực", "thể hư", " thêm "))


def main():
    ap = argparse.ArgumentParser(description="Reconcile ViThuoc base graph<-CSV. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    driver = get_driver()
    plan, rollback = [], []
    try:
        for r in rows:
            b = r["tên_bệnh"].strip()
            h = r["hội_chứng"].strip()
            vi = r.get("vị_thuốc", "").strip()
            if not b or not h or not vi:
                continue
            base = base_herbs(vi)
            if not base:
                continue
            base_norm = {norm(x) for x in base}
            nodes = run_read(driver, """
                MATCH (p:BaiThuoc {benh_ly:$b, hoi_chung:$h})
                OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
                RETURN elementId(p) AS eid, p.name AS n, collect(DISTINCT v.name) AS hb
            """, b=b, h=h)
            for nd in nodes:
                g_norm = {norm(x) for x in nd["hb"] if x}
                extra = g_norm - base_norm            # graph thừa (rác / bài thay thế)
                missing = base_norm - g_norm
                dirty_name = clean_bai_name(nd["n"]) != (nd["n"] or "").strip()
                # CHỈ dọn node có ViThuoc RÁC IMPORT (ngoặc lệch/'.'/note-word) hoặc TÊN bẩn —
                # BỎ QUA node chỉ lệch vị do nhiễu tokenization/chính tả (tránh ghi đè hỏng graph).
                has_garbage = any(is_garbage_vithuoc(x) for x in nd["hb"])
                if not (has_garbage or dirty_name):
                    continue
                plan.append({"eid": nd["eid"], "benh": b, "hc": h, "old_name": nd["n"],
                             "new_name": clean_bai_name(nd["n"]), "old_herbs": nd["hb"],
                             "new_herbs": base, "extra": sorted(extra), "missing": sorted(missing)})
        print(f"Node BaiThuoc cần dọn (ViThuoc lệch base HOẶC tên bẩn): {len(plan)}")
        for it in plan[:20]:
            tag = []
            if it["extra"]:
                tag.append(f"gỡ {len(it['extra'])} vị thừa")
            if it["missing"]:
                tag.append(f"thêm {len(it['missing'])} vị thiếu")
            if it["old_name"] != it["new_name"]:
                tag.append("dọn tên")
            print(f"   [{it['benh']} × {it['hc']}] {'; '.join(tag)}")
            if it["extra"]:
                print(f"       thừa: {it['extra'][:6]}")
        if args.apply:
            for it in plan:
                rollback.append({"eid": it["eid"], "old_name": it["old_name"], "old_herbs": it["old_herbs"]})
            json.dump(rollback, open(ROLLBACK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"\nĐã ghi rollback: {ROLLBACK}. Ghi graph...")
            for it in plan:
                run_write(driver, """
                    MATCH (p:BaiThuoc) WHERE elementId(p)=$e SET p.name=$nm
                    WITH p OPTIONAL MATCH (p)-[r:BAO_GỒM]->(:ViThuoc) DELETE r
                    WITH p UNWIND $herbs AS hn MERGE (v:ViThuoc {name:hn}) MERGE (p)-[:BAO_GỒM]->(v)
                """, e=it["eid"], nm=it["new_name"], herbs=it["new_herbs"])
            print(f"XONG: dọn {len(plan)} node.")
        else:
            print(f"\n(DRY-RUN — {len(plan)} node sẽ dọn. ⚠️ BACKUP Aura rồi thêm --apply.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
