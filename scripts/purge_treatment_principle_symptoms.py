#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/purge_treatment_principle_symptoms.py — Dọn node TrieuChung là PHÁP TRỊ (dưỡng âm/bổ thận/
hoạt huyết hóa ứ/sinh tân...) lọt vào graph do import CSV bẩn (cột triệu_chứng chứa pháp trị).

Đây là việc THẦY THUỐC LÀM, KHÔNG phải triệu chứng người bệnh -> làm bẩn gợi ý '/api/related-symptoms'
(đọc graph). Bộ lọc _is_treatment_principle đã chặn lúc HIỂN THỊ; script này dọn TẬN GỐC trong graph.

An toàn: chỉ đụng node mà `TCMFusionPipeline._is_treatment_principle(name)` = True (bắt đầu bằng
động-từ-pháp-trị đặc trưng — đã kiểm 0 false-positive trên triệu chứng thật). DETACH DELETE node
(gỡ luôn cạnh CÓ_BIỂU_HIỆN sai). MẶC ĐỊNH DRY-RUN; --apply mới ghi (+ rollback JSON để khôi phục).

    python scripts/purge_treatment_principle_symptoms.py            # dry-run: liệt kê
    python scripts/purge_treatment_principle_symptoms.py --apply     # xóa (cần backup Aura Snapshot)
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
from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402

ROLLBACK = os.path.join(_ROOT, "data", "treatment_principle_symptoms_rollback.json")


def main():
    ap = argparse.ArgumentParser(description="Dọn TrieuChung PHÁP-TRỊ khỏi graph. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    d = get_driver()
    try:
        rows = run_read(d, "MATCH (t:TrieuChung) RETURN elementId(t) AS eid, t.name AS name")
        pt = [r for r in rows if r.get("name") and F._is_treatment_principle(r["name"])]
        print(f"Tổng TrieuChung: {len(rows)} | là PHÁP TRỊ (sẽ xóa): {len(pt)}")
        plan = []
        for r in pt:
            edges = run_read(d, """
                MATCH (h:HoiChung)-[rel:CÓ_BIỂU_HIỆN]->(t:TrieuChung) WHERE elementId(t)=$e
                RETURN h.name AS hc, rel.benh_ly AS benh_ly
            """, e=r["eid"])
            el = [{"hc": x["hc"], "benh_ly": x["benh_ly"]} for x in edges]
            plan.append({"name": r["name"], "edges": el})
            print(f"   {r['name']!r}  <- CÓ_BIỂU_HIỆN từ {[(x['hc'], x['benh_ly']) for x in el]}")
        if not plan:
            print("Không có node pháp-trị nào — graph đã sạch.")
            return
        if not args.apply:
            print(f"\n(DRY-RUN — {len(plan)} node sẽ bị DETACH DELETE. ⚠️ Backup Aura rồi thêm --apply.)")
            return
        json.dump(plan, open(ROLLBACK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nĐã ghi rollback: {ROLLBACK}. Xóa node...")
        deleted = 0
        for it in plan:
            summary = run_write(d, "MATCH (t:TrieuChung {name:$n}) DETACH DELETE t", n=it["name"])
            deleted += summary.counters.nodes_deleted
        print(f"XONG: xóa {deleted} node pháp-trị. "
              f"Khôi phục: dựng lại node + cạnh CÓ_BIỂU_HIỆN theo {os.path.basename(ROLLBACK)}.")
    finally:
        d.close()


if __name__ == "__main__":
    main()
