#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/purge_garbage_vithuoc.py — Dọn nốt ViThuoc RÁC còn sót (blob note import) trên graph.

Với mỗi ViThuoc rác (ngoặc lệch / '.'/';'/':' / note-word) nối tới BaiThuoc: bóc VỊ ĐẦU thật
(text trước dấu '.;:(' + bỏ ngoặc note dẫn đầu + bỏ ')' cuối), MERGE vị sạch + nối BaiThuoc,
rồi XÓA cạnh BAO_GỒM tới node rác. An toàn: chỉ đụng node RÁC; vị đầu là thuốc thật -> không
mất thuốc. Node rác mồ côi (hết cạnh) -> xóa hẳn.

MẶC ĐỊNH DRY-RUN. --apply mới ghi.
    python scripts/purge_garbage_vithuoc.py            # dry-run
    python scripts/purge_garbage_vithuoc.py --apply
"""
import argparse
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from kg_maintenance import get_driver, run_read, run_write  # noqa: E402

GARBAGE = ("v.name CONTAINS '.' OR v.name CONTAINS ';' OR v.name CONTAINS ':' "
           "OR (v.name CONTAINS ')' AND NOT v.name CONTAINS '(') "
           "OR (v.name CONTAINS '(' AND NOT v.name CONTAINS ')')")


_FORMULA_END = re.compile(r'(?:thang|tán|hoàn|ẩm|đan|tễ|cao|phương|bổ|tán)\s*$', re.IGNORECASE)


def lead_herb(name):
    """Bóc VỊ đầu thật từ token rác. '<Bài>: vị' -> lấy vị sau ':'; cắt tại .;( và ' nếu/gia/thêm'."""
    s = re.sub(r'^\s*\([^)]*\)\s*', '', name or "")           # bỏ ngoặc note dẫn đầu
    if ":" in s:                                              # '<Bài>: vị' -> lấy sau ':' nếu trước là tên bài
        head, rest = s.split(":", 1)
        if _FORMULA_END.search(head.strip()) or "thể thực" in head.lower() or "thể hư" in head.lower():
            s = rest
    s = re.split(r'\s*[.;:(]|\s+(?:nếu|gia|thêm|hợp|bỏ|bớt)\b', s, maxsplit=1, flags=re.IGNORECASE)[0]
    return s.strip().rstrip(") ").strip()


def valid_herb(h):
    """Vị thuốc hợp lệ để MERGE (không phải tên bài / nhãn / cụm 'tham khảo')."""
    if not h or len(h) < 2 or len(h) > 20:
        return False
    hl = h.lower()
    if _FORMULA_END.search(hl):
        return False
    return not any(k in hl for k in ("thể thực", "thể hư", "tham khảo", "không liệt", "dùng", "/"))


def main():
    ap = argparse.ArgumentParser(description="Dọn ViThuoc rác sót. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    d = get_driver()
    try:
        rows = run_read(d, f"""
            MATCH (p:BaiThuoc)-[:BAO_GỒM]->(v:ViThuoc) WHERE {GARBAGE}
            RETURN elementId(p) AS pe, p.benh_ly AS b, p.hoi_chung AS h,
                   elementId(v) AS ve, v.name AS vn
        """)
        print(f"Cạnh BAO_GỒM tới ViThuoc rác: {len(rows)}")
        fixed = 0
        for r in rows:
            lead = lead_herb(r["vn"])
            keep = valid_herb(lead)
            action = f"bóc '{lead}'" if keep else "(bỏ hẳn — tên bài/nhãn)"
            print(f"   [{r['b']} × {r['h']}] rác {r['vn'][:45]!r} -> {action} + xóa cạnh rác")
            if args.apply:
                if keep:
                    run_write(d, """
                        MATCH (p:BaiThuoc) WHERE elementId(p)=$pe
                        MERGE (nv:ViThuoc {name:$lead}) MERGE (p)-[:BAO_GỒM]->(nv)
                    """, pe=r["pe"], lead=lead)
                run_write(d, """
                    MATCH (p:BaiThuoc)-[rel:BAO_GỒM]->(v:ViThuoc)
                    WHERE elementId(p)=$pe AND elementId(v)=$ve DELETE rel
                """, pe=r["pe"], ve=r["ve"])
                fixed += 1
        if args.apply:
            # xóa hẳn ViThuoc rác mồ côi (không còn cạnh nào)
            run_write(d, f"MATCH (v:ViThuoc) WHERE ({GARBAGE}) AND NOT (v)--() DELETE v")
            print(f"XONG: xử {fixed} cạnh rác + xóa node rác mồ côi.")
        else:
            print(f"\n(DRY-RUN — {len(rows)} cạnh sẽ xử. Thêm --apply.)")
    finally:
        d.close()


if __name__ == "__main__":
    main()
