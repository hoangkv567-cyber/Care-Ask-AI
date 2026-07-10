#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
ĐỒNG BỘ SỬA ĐỔI BÀI THUỐC: CSV -> KNOWLEDGE GRAPH (Neo4j)
================================================================================
kg_import_csv_gaps.py chỉ THÊM phần graph THIẾU — KHÔNG sửa được node đã lệch.
Script này bù đúng chỗ đó: tìm mọi BaiThuoc trong graph có TÊN hoặc BỘ VỊ THUỐC
KHÁC với data/Medicine_clean.csv (đã sửa), rồi cập nhật node cho khớp CSV.

Ví dụ điển hình: graph vẫn giữ 'Viêm phế quản × Phong hàn -> Tang bạch thang' (bài
LƯƠNG cho chứng HÀN) trong khi CSV đã sửa thành 'Hạnh tô tán' (tân ôn).

An toàn:
  - MẶC ĐỊNH DRY-RUN: chỉ liệt kê drift + ghi kế hoạch data/kg_formula_reconcile_plan.json.
    Thêm --apply mới ghi graph.
  - --apply ghi ROLLBACK data/kg_formula_reconcile_rollback.json (tên+vị cũ) để hoàn tác:
        python scripts/reconcile_kg_formulas.py --rollback data/kg_formula_reconcile_rollback.json
  - CHỈ đụng node có CẶP (benh_ly, hoi_chung) khớp DUY NHẤT một dòng CSV (an toàn, không mơ hồ).
    Cặp mơ hồ (nhiều dòng CSV) hoặc node không có dòng CSV -> BÁO CÁO, KHÔNG sửa.

⚠️  BACKUP graph TRƯỚC --apply (Neo4j Aura: Console -> Snapshot).

Chạy từ THƯ MỤC GỐC dự án:
    python scripts/reconcile_kg_formulas.py                 # dry-run (liệt kê drift)
    python scripts/reconcile_kg_formulas.py --apply         # ghi thật (+ rollback)
    python scripts/reconcile_kg_formulas.py --rollback data/kg_formula_reconcile_rollback.json
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
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _p in (_SCRIPTS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kg_maintenance import get_driver, run_read, run_write  # noqa: E402

CSV_PATH = os.getenv("TCM_CSV_PATH") or os.path.join(_ROOT, "data", "Medicine_clean.csv")
PLAN_PATH = os.path.join(_ROOT, "data", "kg_formula_reconcile_plan.json")
ROLLBACK_PATH = os.path.join(_ROOT, "data", "kg_formula_reconcile_rollback.json")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def split_herbs(vi):
    """GIỐNG scripts/audit_thermal_polarity.py: bỏ ghi chú ngoặc, tách , ; . ; loại mệnh đề chú thích."""
    if not vi:
        return []
    vi = re.sub(r"\([^)]*\)", " ", vi)
    out = []
    for p in re.split(r"[,;.]", vi):
        h = p.strip().strip(".").strip()
        if not h:
            continue
        if re.search(r"gia\b|giảm|nếu|hoặc|liều|thêm|bớt", h.lower()):
            continue
        if len(h) > 30:
            continue
        out.append(h)
    return out


def git_changed_pairs():
    """Tập (benh_norm, hc_norm) của những dòng CSV ĐÃ ĐỔI so với HEAD (git diff).
    Dùng để CHỈ đồng bộ đúng các bản sửa, tránh ghi đè 200+ node lệch do tokenization."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "diff", "HEAD", "--", "data/Medicine_clean.csv"],
            cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
        ).stdout
    except Exception as e:
        print(f"Không chạy được git diff: {e}", file=sys.stderr)
        return set()
    pairs = set()
    for line in out.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        cells = next(csv.reader([line[1:]]))  # tôn trọng dấu ngoặc kép CSV
        if len(cells) >= 2 and cells[0].strip() and cells[1].strip():
            pairs.add((norm(cells[0]), norm(cells[1])))
    return pairs


def load_csv_index():
    """(benh_norm, hc_norm) -> list[{bai, herbs(list), herbs_norm(set), row_line}]."""
    idx = {}
    with open(CSV_PATH, encoding="utf-8-sig") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            benh = (r.get("tên_bệnh") or "").strip()
            hc = (r.get("hội_chứng") or "").strip()
            bai = (r.get("bài_thuốc") or "").strip()
            vi = (r.get("vị_thuốc") or "").strip()
            if not benh or not hc or not bai:
                continue
            herbs = split_herbs(vi)
            idx.setdefault((norm(benh), norm(hc)), []).append({
                "bai": bai, "herbs": herbs,
                "herbs_norm": {norm(h) for h in herbs}, "line": i + 2,
            })
    return idx


def fetch_graph_formulas(driver):
    q = """
    MATCH (p:BaiThuoc)
    OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
    RETURN elementId(p) AS eid, p.benh_ly AS benh, p.hoi_chung AS hc,
           p.name AS bai, collect(DISTINCT v.name) AS herbs
    """
    return run_read(driver, q)


def build_plan(driver, targets=None):
    """targets: nếu là set (benh_norm, hc_norm) thì CHỈ xét các cặp đó; None = toàn bộ."""
    csv_idx = load_csv_index()
    plan, manual, orphan = [], [], []
    for g in fetch_graph_formulas(driver):
        key = (norm(g["benh"]), norm(g["hc"]))
        if targets is not None and key not in targets:
            continue
        rows = csv_idx.get(key)
        if not rows:
            orphan.append(g)                       # graph có, CSV không -> không đụng
            continue
        g_herbs_norm = {norm(h) for h in (g["herbs"] or [])}
        # Có dòng CSV nào TRÙNG TÊN bài với node graph không?
        same_name = [r for r in rows if norm(r["bai"]) == norm(g["bai"])]
        if same_name:
            r = same_name[0]
            if r["herbs_norm"] != g_herbs_norm:    # chỉ lệch VỊ THUỐC
                plan.append(_mk(g, r, g_herbs_norm, "HERB_DRIFT"))
            continue
        # Không trùng tên: chỉ sửa khi cặp (benh,hc) có DUY NHẤT một dòng CSV (không mơ hồ)
        if len(rows) == 1:
            r = rows[0]
            dt = "NAME+HERB_DRIFT" if r["herbs_norm"] != g_herbs_norm else "NAME_DRIFT"
            plan.append(_mk(g, r, g_herbs_norm, dt))
        else:
            manual.append({"graph": g, "csv_bais": [r["bai"] for r in rows]})
    return plan, manual, orphan


def _mk(g, r, g_herbs_norm, drift):
    return {
        "eid": g["eid"], "benh": g["benh"], "hc": g["hc"],
        "old_bai": g["bai"], "new_bai": r["bai"],
        "old_herbs": g["herbs"] or [], "new_herbs": r["herbs"],
        "added": sorted(r["herbs_norm"] - g_herbs_norm),
        "removed": sorted(g_herbs_norm - r["herbs_norm"]),
        "drift": drift, "csv_line": r["line"],
    }


def apply_plan(driver, plan):
    rollback = []
    for it in plan:
        rollback.append({"eid": it["eid"], "old_bai": it["old_bai"], "old_herbs": it["old_herbs"]})
    with open(ROLLBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(rollback, f, ensure_ascii=False, indent=1)
    print(f"Đã ghi rollback: {ROLLBACK_PATH}")
    for it in plan:
        _write_node(driver, it["eid"], it["new_bai"], it["new_herbs"])
        print(f"  ✓ [{it['benh']} × {it['hc']}] {it['old_bai']} -> {it['new_bai']}")
    print(f"XONG: cập nhật {len(plan)} node.")


def _write_node(driver, eid, new_bai, new_herbs):
    run_write(driver, """
    MATCH (p:BaiThuoc) WHERE elementId(p) = $eid
    SET p.name = $bai
    WITH p
    OPTIONAL MATCH (p)-[r:BAO_GỒM]->(:ViThuoc) DELETE r
    WITH p
    UNWIND $herbs AS hn
    MERGE (v:ViThuoc {name: hn})
    MERGE (p)-[:BAO_GỒM]->(v)
    """, eid=eid, bai=new_bai, herbs=new_herbs)


def do_rollback(driver, path):
    data = json.load(open(path, encoding="utf-8"))
    for it in data:
        _write_node(driver, it["eid"], it["old_bai"], it["old_herbs"])
        print(f"  ↩ khôi phục -> {it['old_bai']}")
    print(f"XONG: hoàn tác {len(data)} node.")


def main():
    ap = argparse.ArgumentParser(description="Đồng bộ sửa đổi bài thuốc CSV -> graph. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true", help="Ghi graph thật (+ rollback). Mặc định chỉ liệt kê.")
    ap.add_argument("--rollback", metavar="FILE", help="Hoàn tác từ file rollback JSON")
    ap.add_argument("--changed-only", action="store_true",
                    help="CHỈ đồng bộ các cặp (bệnh × hội chứng) đã đổi trong git diff CSV (AN TOÀN — khuyến nghị).")
    ap.add_argument("--targets", metavar="FILE",
                    help="JSON [{benh, hc}] giới hạn phạm vi đồng bộ.")
    ap.add_argument("--all", action="store_true",
                    help="Cho phép --apply TOÀN BỘ drift (kể cả lệch do tokenization — NGUY HIỂM).")
    args = ap.parse_args()

    driver = get_driver()
    try:
        if args.rollback:
            do_rollback(driver, args.rollback)
            return

        targets = None
        if args.changed_only:
            targets = git_changed_pairs()
            print(f"[--changed-only] {len(targets)} cặp CSV đã đổi so với HEAD.")
        elif args.targets:
            raw = json.load(open(args.targets, encoding="utf-8"))
            targets = {(norm(x["benh"]), norm(x["hc"])) for x in raw}
            print(f"[--targets] {len(targets)} cặp mục tiêu.")

        plan, manual, orphan = build_plan(driver, targets)
        json.dump(plan, open(PLAN_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        name_drift = [p for p in plan if "NAME" in p["drift"]]
        herb_only = [p for p in plan if p["drift"] == "HERB_DRIFT"]
        print(f"CSV: {CSV_PATH}")
        print(f"Node lệch CSV: {len(plan)}  (đổi TÊN bài: {len(name_drift)} | chỉ lệch vị: {len(herb_only)})")
        print(f"Cặp mơ hồ (nhiều dòng CSV, BỎ QUA): {len(manual)}")
        print(f"Node không có dòng CSV (orphan, BỎ QUA): {len(orphan)}")
        print(f"-> kế hoạch: {PLAN_PATH}")
        print()

        # Đổi TÊN bài = tín hiệu cao (thay bài thật). Liệt kê ĐẦY ĐỦ.
        show = plan if targets is not None else name_drift
        heading = "TẤT CẢ (phạm vi mục tiêu)" if targets is not None else "ĐỔI TÊN BÀI (tín hiệu cao)"
        print(f"=== {heading} — {len(show)} node ===")
        for it in show:
            print(f"[{it['drift']}] {it['benh']} × {it['hc']}  (CSV dòng {it['csv_line']})")
            print(f"    bài:  {it['old_bai']}  ->  {it['new_bai']}")
            if it["removed"]:
                print(f"    bỏ vị:   {', '.join(it['removed'])}")
            if it["added"]:
                print(f"    thêm vị: {', '.join(it['added'])}")
            print()
        if targets is None and herb_only:
            print(f"(*) {len(herb_only)} node CHỈ lệch vị thuốc — PHẦN LỚN là nhiễu tokenization/"
                  f"chính tả (tên bài lọt vào ViThuoc, 'thủy diệt'≈'thủy điệt'...). KHÔNG tự sửa; "
                  f"xem {PLAN_PATH} nếu cần rà tay.")

        if args.apply:
            if not plan:
                print("\nKhông có gì để cập nhật.")
                return
            if targets is None and not args.all:
                print("\n[TỪ CHỐI] --apply TOÀN BỘ dễ ghi đè nhiễu tokenization. "
                      "Dùng --changed-only (khuyến nghị) hoặc --targets, hoặc ép --all nếu chắc chắn.")
                return
            print("\n⚠️  Đảm bảo đã BACKUP graph (Aura Snapshot). --apply: bắt đầu ghi...")
            apply_plan(driver, plan)
        else:
            print("\n(DRY-RUN — chưa ghi graph. Thêm --apply để thực thi sau khi backup Aura.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
