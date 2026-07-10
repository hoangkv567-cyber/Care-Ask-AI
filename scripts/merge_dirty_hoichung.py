#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
GỘP NODE HoiChung BẨN vào node SẠCH cùng bệnh (Neo4j)
================================================================================
Đồng bộ graph với CSV đã chuẩn hoá (②): CSV đổi 'Viêm yết hầu × Cấp (Phong nhiệt)'
-> 'Phong nhiệt' và '× Âm hư (mạn tính)' -> 'Âm hư'. Graph vẫn giữ node HoiChung
bẩn + BaiThuoc gắn nhãn bẩn. Script repoint bài thuốc + dời cạnh triệu chứng sang
node sạch rồi XÓA node bẩn.

TARGETS lấy đúng 2 cặp đã sửa trong CSV — KHÔNG quét mù toàn graph.

An toàn:
  - MẶC ĐỊNH DRY-RUN: chỉ liệt kê. Thêm --apply mới ghi.
  - --apply ghi ROLLBACK data/merge_hoichung_rollback.json để tái tạo node bẩn.
  - Chỉ gộp node bẩn ĐƠN-BỆNH (không dùng chung bệnh khác) -> an toàn xóa.

⚠️  BACKUP graph TRƯỚC --apply (Aura Snapshot).

Chạy:  python scripts/merge_dirty_hoichung.py            # dry-run
        python scripts/merge_dirty_hoichung.py --apply    # ghi (+ rollback)
        python scripts/merge_dirty_hoichung.py --rollback data/merge_hoichung_rollback.json
"""
import argparse
import json
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

ROLLBACK_PATH = os.path.join(_ROOT, "data", "merge_hoichung_rollback.json")

# (bệnh, hội chứng BẨN, hội chứng SẠCH)
TARGETS = [
    ("Viêm yết hầu", "Cấp (Phong nhiệt)", "Phong nhiệt"),
    ("Viêm yết hầu", "Âm hư (mạn tính)", "Âm hư"),
]


def inspect(driver, benh, dirty, clean):
    """Trả trạng thái trước-gộp: node bẩn/sạch tồn tại?, số bệnh dùng node bẩn, bài & triệu chứng."""
    q = """
    OPTIONAL MATCH (dh:HoiChung {name:$dirty})
    OPTIONAL MATCH (ch:HoiChung {name:$clean})
    WITH dh, ch
    OPTIONAL MATCH (bd:BenhLy)-[:CHIA_THÀNH]->(dh)
    OPTIONAL MATCH (dh)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
    OPTIONAL MATCH (dh)-[rt:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
    RETURN dh IS NOT NULL AS dirty_exists, ch IS NOT NULL AS clean_exists,
           collect(DISTINCT bd.name) AS dirty_benh,
           collect(DISTINCT p.name) AS bai,
           collect(DISTINCT {t:t.name, benh_ly:rt.benh_ly}) AS trieuchung
    """
    r = run_read(driver, q, dirty=dirty, clean=clean)
    return r[0] if r else None


def do_merge(driver, benh, dirty, clean):
    # 1) repoint BaiThuoc: prop + cạnh ĐƯỢC_ĐIỀU_TRỊ_BẰNG
    run_write(driver, """
    MATCH (dh:HoiChung {name:$dirty})-[r:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
    MATCH (ch:HoiChung {name:$clean})
    SET p.hoi_chung = $clean
    MERGE (ch)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p)
    DELETE r
    """, dirty=dirty, clean=clean)
    # 2) dời cạnh CÓ_BIỂU_HIỆN (giữ prop benh_ly)
    run_write(driver, """
    MATCH (dh:HoiChung {name:$dirty})-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
    MATCH (ch:HoiChung {name:$clean})
    MERGE (ch)-[r2:CÓ_BIỂU_HIỆN]->(t)
      ON CREATE SET r2.benh_ly = r.benh_ly
    DELETE r
    """, dirty=dirty, clean=clean)
    # 3) đảm bảo BenhLy-CHIA_THÀNH->clean, rồi xóa hẳn node bẩn (DETACH gỡ CHIA_THÀNH còn lại)
    run_write(driver, """
    MATCH (b:BenhLy {name:$benh}), (ch:HoiChung {name:$clean})
    MERGE (b)-[:CHIA_THÀNH]->(ch)
    WITH 1 AS _
    MATCH (dh:HoiChung {name:$dirty})
    DETACH DELETE dh
    """, benh=benh, clean=clean, dirty=dirty)


def do_rollback(driver, path):
    data = json.load(open(path, encoding="utf-8"))
    for it in data:
        benh, dirty, clean = it["benh"], it["dirty"], it["clean"]
        run_write(driver, "MERGE (dh:HoiChung {name:$dirty})", dirty=dirty)
        run_write(driver, """
        MATCH (b:BenhLy {name:$benh}), (dh:HoiChung {name:$dirty})
        MERGE (b)-[:CHIA_THÀNH]->(dh)
        """, benh=benh, dirty=dirty)
        for tc in it.get("trieuchung", []):
            run_write(driver, """
            MATCH (dh:HoiChung {name:$dirty}) MATCH (t:TrieuChung {name:$t})
            MERGE (dh)-[r:CÓ_BIỂU_HIỆN]->(t) ON CREATE SET r.benh_ly=$bl
            """, dirty=dirty, t=tc["t"], bl=tc.get("benh_ly"))
        for bai in it.get("bai", []):
            run_write(driver, """
            MATCH (dh:HoiChung {name:$dirty}) MATCH (p:BaiThuoc {name:$bai})
            SET p.hoi_chung=$dirty
            MERGE (dh)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p)
            """, dirty=dirty, bai=bai)
        print(f"  ↩ tái tạo node bẩn {dirty!r}")
    print(f"XONG: hoàn tác {len(data)} node.")


def main():
    ap = argparse.ArgumentParser(description="Gộp HoiChung bẩn vào node sạch. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true", help="Ghi graph thật (+ rollback).")
    ap.add_argument("--rollback", metavar="FILE", help="Hoàn tác từ file rollback JSON")
    args = ap.parse_args()

    driver = get_driver()
    try:
        if args.rollback:
            do_rollback(driver, args.rollback)
            return
        rollback, plan = [], []
        for benh, dirty, clean in TARGETS:
            st = inspect(driver, benh, dirty, clean)
            print(f"[{benh}] gộp {dirty!r} -> {clean!r}")
            if not st or not st["dirty_exists"]:
                print("    (node bẩn KHÔNG tồn tại — bỏ qua)")
                continue
            if not st["clean_exists"]:
                print("    ⚠️ node SẠCH chưa tồn tại — BỎ QUA (an toàn; cần tạo trước).")
                continue
            others = [b for b in st["dirty_benh"] if b and b != benh]
            if others:
                print(f"    ⚠️ node bẩn dùng chung bởi bệnh khác {others} — BỎ QUA (không xóa an toàn).")
                continue
            tcs = [t for t in st["trieuchung"] if t.get("t")]
            print(f"    bài repoint: {st['bai']}")
            print(f"    triệu chứng dời: {len(tcs)}  |  xóa node bẩn.")
            plan.append((benh, dirty, clean))
            rollback.append({"benh": benh, "dirty": dirty, "clean": clean,
                             "bai": st["bai"], "trieuchung": tcs})

        if not plan:
            print("\nKhông có gì để gộp.")
            return
        if args.apply:
            json.dump(rollback, open(ROLLBACK_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"\nĐã ghi rollback: {ROLLBACK_PATH}")
            print("⚠️  Đảm bảo đã BACKUP graph. --apply: ghi...")
            for benh, dirty, clean in plan:
                do_merge(driver, benh, dirty, clean)
                print(f"  ✓ gộp {dirty!r} -> {clean!r}")
            print(f"XONG: gộp {len(plan)} node.")
        else:
            print("\n(DRY-RUN — chưa ghi graph. Thêm --apply sau khi backup Aura.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
