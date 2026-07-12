#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/normalize_whitespace_nodes.py — Dọn node có TÊN/PROP dính whitespace/tab (Neo4j).

Import cũ tạo node SHADOW tên bẩn ('\\tĐầu thống', 'Nhĩ lung ', 'Khí trệ huyết ứ ',
'Long đởm tả can thang '...) song song node SẠCH cùng tên. Node bẩn:
  - BenhLy/HoiChung/BaiThuoc bẩn -> pipeline bind p.benh_ly=b.name (tên SẠCH) nên
    node bẩn không khớp -> hoặc CHẾT (thừa) hoặc làm Mục 5 TRẮNG (vd Nhĩ lung).
  - BaiThuoc còn PROP benh_ly/hoi_chung dính whitespace -> lệch bind.

Script GỘP node bẩn vào twin SẠCH: dời MỌI quan hệ (4 loại) sang node sạch, TRIM prop
benh_ly/hoi_chung của BaiThuoc, xóa node bẩn. Sau đó gợi ý chạy dedup_baithuoc.py.

Quan hệ: BenhLy-[CHIA_THÀNH]->HoiChung-[ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->BaiThuoc-[BAO_GỒM]->ViThuoc
         HoiChung-[CÓ_BIỂU_HIỆN {benh_ly}]->TrieuChung

An toàn: MẶC ĐỊNH DRY-RUN. --apply mới ghi. ⚠️ BACKUP Aura Snapshot trước --apply.
    python scripts/normalize_whitespace_nodes.py            # dry-run
    python scripts/normalize_whitespace_nodes.py --apply
"""
import argparse
import os
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

# Node có tên bẩn -> (label, direction, rel_type) cần dời. Dùng elementId để định danh
# node bẩn (tên trùng có thể mơ hồ), khớp twin sạch theo tên đã trim (+ prop cho BaiThuoc).
REPOINT = {
    # label: [(direction 'out'/'in', rel_type)]
    "BenhLy":   [("out", "CHIA_THÀNH")],
    "HoiChung": [("in", "CHIA_THÀNH"), ("out", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG"), ("out", "CÓ_BIỂU_HIỆN")],
    "BaiThuoc": [("in", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG"), ("out", "BAO_GỒM")],
}


def find_dirty(driver, label):
    return run_read(driver, f"""
        MATCH (n:{label}) WHERE n.name <> trim(n.name) OR n.name CONTAINS '\t'
        WITH n, trim(replace(n.name,'\t','')) AS clean
        OPTIONAL MATCH (m:{label}) WHERE m.name = clean AND elementId(m) <> elementId(n)
        RETURN elementId(n) AS eid, n.name AS dirty, clean,
               n.benh_ly AS benh_ly, n.hoi_chung AS hoi_chung,
               collect(elementId(m)) AS clean_ids
    """)


def clean_twin_id(driver, label, row):
    """Chọn elementId twin sạch. BaiThuoc: khớp thêm benh_ly+hoi_chung (đã trim)."""
    if label != "BaiThuoc":
        return row["clean_ids"][0] if row["clean_ids"] else None
    r = run_read(driver, """
        MATCH (m:BaiThuoc) WHERE m.name=$cn AND trim(replace(m.benh_ly,'\t',''))=$bl
          AND trim(replace(m.hoi_chung,'\t',''))=$hc AND elementId(m)<>$eid
        RETURN elementId(m) AS eid LIMIT 1
    """, cn=row["clean"], bl=(row["benh_ly"] or "").replace("\t", "").strip(),
        hc=(row["hoi_chung"] or "").replace("\t", "").strip(), eid=row["eid"])
    return r[0]["eid"] if r else None


def repoint(driver, label, dirty_eid, clean_eid):
    for direction, rt in REPOINT[label]:
        if rt == "CÓ_BIỂU_HIỆN":
            # BẢO TOÀN benh_ly: chỉ tạo edge (target, benh_ly) mà clean CHƯA có (giữ
            # granularity theo bệnh cho scorer), rồi xóa hết edge bẩn. MERGE thường
            # gộp mất benh_ly khi trùng target -> hỏng chấm điểm bệnh dùng node này.
            run_write(driver, """
                MATCH (d) WHERE elementId(d)=$de
                MATCH (c) WHERE elementId(c)=$ce
                MATCH (d)-[r:CÓ_BIỂU_HIỆN]->(y)
                WHERE NOT EXISTS {
                    MATCH (c)-[r3:CÓ_BIỂU_HIỆN]->(y)
                    WHERE coalesce(r3.benh_ly,'') = coalesce(r.benh_ly,'')
                }
                CREATE (c)-[r2:CÓ_BIỂU_HIỆN]->(y) SET r2.benh_ly = r.benh_ly
            """, de=dirty_eid, ce=clean_eid)
            run_write(driver, """
                MATCH (d) WHERE elementId(d)=$de
                MATCH (d)-[r:CÓ_BIỂU_HIỆN]->() DELETE r
            """, de=dirty_eid)
        elif direction == "out":
            # (dirty)-[r]->(y)  =>  (clean)-[:rt]->(y)
            run_write(driver, f"""
                MATCH (d) WHERE elementId(d)=$de
                MATCH (c) WHERE elementId(c)=$ce
                MATCH (d)-[r:{rt}]->(y)
                MERGE (c)-[r2:{rt}]->(y)
                  ON CREATE SET r2 = properties(r)
                DELETE r
            """, de=dirty_eid, ce=clean_eid)
        else:
            run_write(driver, f"""
                MATCH (d) WHERE elementId(d)=$de
                MATCH (c) WHERE elementId(c)=$ce
                MATCH (x)-[r:{rt}]->(d)
                MERGE (x)-[r2:{rt}]->(c)
                  ON CREATE SET r2 = properties(r)
                DELETE r
            """, de=dirty_eid, ce=clean_eid)
    run_write(driver, "MATCH (d) WHERE elementId(d)=$de DETACH DELETE d", de=dirty_eid)


def main():
    ap = argparse.ArgumentParser(description="Gộp node tên bẩn vào twin sạch. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    driver = get_driver()
    try:
        total = 0
        for label in ("BenhLy", "HoiChung", "BaiThuoc"):
            rows = find_dirty(driver, label)
            # gom theo eid (BaiThuoc query có thể lặp dòng)
            seen = {}
            for r in rows:
                seen[r["eid"]] = r
            print(f"=== {label}: {len(seen)} node tên bẩn ===")
            for eid, r in seen.items():
                twin = clean_twin_id(driver, label, r)
                if not twin:
                    print(f"   ⚠️ {r['dirty']!r}: KHÔNG có twin sạch khớp -> chỉ TRIM tên")
                    if args.apply:
                        run_write(driver, "MATCH (n) WHERE elementId(n)=$e SET n.name=$c",
                                  e=eid, c=r["clean"])
                    total += 1
                    continue
                extra = ""
                if label == "BaiThuoc":
                    extra = f" [{r['benh_ly']!r}×{r['hoi_chung']!r}]"
                print(f"   {r['dirty']!r}{extra} -> gộp vào twin sạch {r['clean']!r}")
                if args.apply:
                    repoint(driver, label, eid, twin)
                total += 1

        # TRIM prop benh_ly/hoi_chung còn dính whitespace trên MỌI BaiThuoc
        dirty_props = run_read(driver, """
            MATCH (p:BaiThuoc)
            WHERE p.benh_ly<>trim(p.benh_ly) OR p.hoi_chung<>trim(p.hoi_chung)
               OR p.benh_ly CONTAINS '\t' OR p.hoi_chung CONTAINS '\t'
            RETURN count(*) AS c
        """)[0]["c"]
        print(f"\n=== BaiThuoc còn PROP benh_ly/hoi_chung dính whitespace: {dirty_props} ===")
        if args.apply and dirty_props:
            run_write(driver, """
                MATCH (p:BaiThuoc)
                WHERE p.benh_ly<>trim(p.benh_ly) OR p.hoi_chung<>trim(p.hoi_chung)
                   OR p.benh_ly CONTAINS '\t' OR p.hoi_chung CONTAINS '\t'
                SET p.benh_ly=trim(replace(p.benh_ly,'\t','')),
                    p.hoi_chung=trim(replace(p.hoi_chung,'\t',''))
            """)
            print("   -> đã trim prop.")

        if args.apply:
            print(f"\nXONG: xử lý {total} node bẩn + trim prop. "
                  f"KHUYẾN NGHỊ: chạy `python scripts/dedup_baithuoc.py --apply` để gộp BaiThuoc trùng phát sinh.")
        else:
            print(f"\n(DRY-RUN — {total} node sẽ gộp/trim. ⚠️ BACKUP Aura rồi thêm --apply.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
