#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/unflag_clean_hoichung.py — Gỡ cờ _flagged_dirty BỊ DÍNH OAN trên node HoiChung TÊN SẠCH.

BỐI CẢNH: `clean-dirty --apply` gắn h._flagged_dirty=true theo tên-nghi-rác (_DIRTY_HOICHUNG_WHERE:
tên null/rỗng, >40 ký tự, có SỐ, có NGOẶC, hoặc chứa ' hoặc | là | gây | do | nếu '). Sau đó
`normalize-names` MERGE node bẩn vào node SẠCH cùng tên (APOC) -> thuộc tính _flagged_dirty=true
THEO SANG node sạch. Hậu quả: hội chứng TRUNG TÂM tên sạch (Khí huyết hư, Thận âm hư...) bị cờ oan
-> _score_syndromes_grounded (fusion_pipeline) và get_all_syndromes (qa_system) đều LOẠI khỏi chấm
điểm/danh sách -> core rơi về hội chứng ngoại lai + Mục 5 trắng.

Ca kích hoạt: "tiểu nhiều, lượng ít, mệt mỏi, ít ngủ" -> Đái tháo nhạt: Thận âm hư (deg 61) bị cờ
oan -> scorer bỏ -> core = Huyết hư ngoại lai, Mục 5 trắng. Gỡ cờ -> core = Thận âm hư (grounded).

AN TOÀN: chỉ gỡ cờ ở node ĐANG flagged NHƯNG tên SẠCH (không khớp bất kỳ tiêu chí bẩn nào). Node
tên THẬT bẩn giữ nguyên cờ. Cờ chỉ dùng làm bộ lọc read-only (không xóa/không đổi dữ liệu khác)
nên gỡ cờ hoàn toàn đảo được. MẶC ĐỊNH DRY-RUN; --apply mới ghi (+ rollback JSON).

    python scripts/unflag_clean_hoichung.py            # dry-run: liệt kê node sẽ gỡ cờ
    python scripts/unflag_clean_hoichung.py --apply     # gỡ cờ (cần backup Aura Snapshot)
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
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from kg_maintenance import _DIRTY_HOICHUNG_WHERE, get_driver, run_read, run_write  # noqa: E402

ROLLBACK = os.path.join(_ROOT, "data", "unflag_clean_hoichung_rollback.json")


def main():
    ap = argparse.ArgumentParser(description="Gỡ cờ _flagged_dirty oan ở HoiChung tên sạch. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    d = get_driver()
    try:
        # Node ĐANG flagged NHƯNG KHÔNG khớp bất kỳ tiêu chí bẩn nào (tên sạch) -> cờ oan.
        rows = run_read(d, f"""
            MATCH (h:HoiChung) WHERE h._flagged_dirty
              AND NOT ({_DIRTY_HOICHUNG_WHERE})
            OPTIONAL MATCH (h)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
            RETURN elementId(h) AS eid, h.name AS name, count(DISTINCT t) AS deg
            ORDER BY deg DESC
        """)
        print(f"Node HoiChung TÊN SẠCH bị cờ _flagged_dirty oan: {len(rows)}")
        for r in rows:
            print(f"   deg={r['deg']:>3}  {r['name']!r}")
        # Vẫn còn flagged mà tên THẬT bẩn (giữ nguyên) — báo cho minh bạch
        dirty_kept = run_read(d, f"""
            MATCH (h:HoiChung) WHERE h._flagged_dirty AND ({_DIRTY_HOICHUNG_WHERE})
            RETURN count(*) AS c
        """)
        print(f"(Giữ nguyên cờ ở {dirty_kept[0]['c'] if dirty_kept else 0} node tên THẬT bẩn.)")
        if not rows:
            print("Không có node nào cần gỡ.")
            return
        if not args.apply:
            print("\n(DRY-RUN — thêm --apply để gỡ cờ. Nên backup Aura Snapshot trước.)")
            return
        json.dump([{"eid": r["eid"], "name": r["name"]} for r in rows],
                  open(ROLLBACK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nĐã ghi rollback: {ROLLBACK}. Gỡ cờ...")
        summary = run_write(d, f"""
            MATCH (h:HoiChung) WHERE h._flagged_dirty AND NOT ({_DIRTY_HOICHUNG_WHERE})
            REMOVE h._flagged_dirty
        """)
        print(f"XONG: gỡ cờ {summary.counters.properties_set} node "
              f"(để khôi phục: MATCH (h) WHERE elementId(h) IN [...] SET h._flagged_dirty=true).")
    finally:
        d.close()


if __name__ == "__main__":
    main()
