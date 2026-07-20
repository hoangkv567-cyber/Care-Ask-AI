#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/rename_diseases_tcm.py — Đổi bệnh danh TÂY Y sang quy ước 'Tên Đông y (tên Tây y)'.

VÌ SAO quy ước này chứ không đổi hẳn: đo trên KB, 7/11 cặp thử sẽ TRÙNG với bệnh danh Đông y đã
tồn tại (Viêm đại tràng->Tiết tả, Parkinson->Chiến chấn, Suy tim->Tâm quý...). Đổi hẳn = GỘP hai
bệnh làm một chứ không phải đổi nhãn. Dạng 'ĐY (TY)' tạo chuỗi MỚI nên không va chạm, và giữ được
đường tra theo tên Tây y — cách bệnh nhân Việt tự mô tả bệnh mình. KB đã dùng quy ước này cho 22
bệnh sẵn có (Phấn thích (trứng cá), Ma chẩn (sởi), Tỵ cứu (viêm mũi dị ứng)...).

⚠ TÊN BỆNH LÀ KHÓA Ở 6 NƠI. Script này migrate ĐỒNG BỘ cả 6; thiếu một nơi là cổng đó IM LẶNG
mất hiệu lực (bệnh không còn khớp gate/giới/tuổi mà không báo lỗi):
    data/Medicine_clean.csv        (nguồn thật)
    data/disease_gates.json        (cổng bệnh danh)
    data/disease_sex.json          (cổng giới)
    data/disease_age.json          (cổng tuổi)
    data/gold_cases.json           (kỳ vọng eval_gold)
    data/batcuong_golden.json      (harness Bát Cương)
Neo4j: node BenhLy KHÔNG đổi ở đây — chạy scripts/kg_import_csv_gaps.py sau, hoặc dùng --cypher để
xuất câu lệnh đổi tên node.

AN TOÀN: mặc định DRY-RUN. --apply mới ghi. Mọi file được sao lưu .bak trước khi ghi.

Chạy:
    python scripts/rename_diseases_tcm.py                 # xem trước
    python scripts/rename_diseases_tcm.py --include-uncertain   # gồm cả nhóm cần xác nhận
    python scripts/rename_diseases_tcm.py --apply
    python scripts/rename_diseases_tcm.py --cypher         # in lệnh đổi tên node Neo4j
    python scripts/rename_diseases_tcm.py --undo           # khôi phục từ .bak
"""
import argparse
import io
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

MAP_FILE = os.path.join(ROOT, "data", "disease_tcm_names.json")
CSV_FILE = os.path.join(ROOT, "data", "Medicine_clean.csv")
JSON_KEYED = ["disease_gates.json", "disease_sex.json", "disease_age.json",
              "gold_cases.json", "batcuong_golden.json"]


def load_mapping(include_uncertain: bool) -> dict:
    d = json.load(io.open(MAP_FILE, encoding="utf-8"))
    m = dict(d.get("cao") or {})
    if include_uncertain:
        for old, info in (d.get("can_xac_nhan") or {}).items():
            m[old] = info["de_xuat"]
    return m


def _walk_replace(obj, mapping, stats):
    """Thay tên bệnh ở MỌI vị trí trong cây JSON: khóa dict, phần tử list, và chuỗi lá.

    Thay theo KHỚP TOÀN PHẦN (==), KHÔNG thay chuỗi con: 'Viêm gan cấp' là chuỗi con của
    'Viêm gan cấp/hồi phục' (một bệnh KHÁC đã có sẵn) — thay chuỗi con sẽ làm hỏng nó.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = mapping.get(k, k)
            if nk != k:
                stats["key"] += 1
            out[nk] = _walk_replace(v, mapping, stats)
        return out
    if isinstance(obj, list):
        return [_walk_replace(x, mapping, stats) for x in obj]
    if isinstance(obj, str):
        nv = mapping.get(obj, obj)
        if nv != obj:
            stats["val"] += 1
        return nv
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--include-uncertain", action="store_true",
                    help="gồm cả nhóm 'can_xac_nhan' (mặc định CHỈ nhóm 'cao')")
    ap.add_argument("--cypher", action="store_true", help="in lệnh Cypher đổi tên node BenhLy")
    ap.add_argument("--undo", action="store_true", help="khôi phục mọi file từ .bak")
    args = ap.parse_args()

    targets = [CSV_FILE] + [os.path.join(ROOT, "data", f) for f in JSON_KEYED]

    if args.undo:
        n = 0
        for p in targets:
            if os.path.exists(p + ".bak"):
                shutil.copy2(p + ".bak", p)
                n += 1
                print(f"  khôi phục {os.path.basename(p)}")
        print(f"Đã khôi phục {n} file.")
        return 0

    mapping = load_mapping(args.include_uncertain)
    print(f"Ánh xạ: {len(mapping)} bệnh"
          f"{' (gồm nhóm cần xác nhận)' if args.include_uncertain else ' (chỉ nhóm tin cậy CAO)'}\n")

    if args.cypher:
        for old, new in sorted(mapping.items()):
            print(f'MATCH (b:BenhLy) WHERE b.name = "{old}" SET b.name = "{new}";')
        return 0

    # --- CSV: cột tên_bệnh, khớp TOÀN PHẦN ---
    raw = io.open(CSV_FILE, encoding="utf-8-sig").read()
    import csv
    rows = list(csv.DictReader(io.StringIO(raw)))
    col = list(rows[0].keys())[0]
    hit = {}
    for r in rows:
        cur = (r.get(col) or "").strip()
        if cur in mapping:
            hit[cur] = hit.get(cur, 0) + 1
    print("CSV — bệnh sẽ đổi tên:")
    for old in sorted(hit):
        print(f"   {old:<30} -> {mapping[old]:<40} ({hit[old]} dòng)")
    missing = [o for o in mapping if o not in hit]
    if missing:
        print(f"\n   ⚠ {len(missing)} tên trong ánh xạ KHÔNG có trong CSV (đã đổi rồi, hoặc sai chính tả):")
        for o in missing:
            print(f"      {o}")

    # --- các file JSON có khóa là tên bệnh ---
    print("\nCác file khóa theo tên bệnh:")
    json_plan = {}
    for f in JSON_KEYED:
        p = os.path.join(ROOT, "data", f)
        if not os.path.exists(p):
            print(f"   {f:<26} (không có)")
            continue
        data = json.load(io.open(p, encoding="utf-8"))
        stats = {"key": 0, "val": 0}
        new = _walk_replace(data, mapping, stats)
        json_plan[p] = new
        print(f"   {f:<26} {stats['key']} khóa + {stats['val']} giá trị sẽ đổi")

    if not args.apply:
        print("\n(DRY-RUN — thêm --apply để ghi. Mọi file sẽ được sao lưu .bak; hoàn tác: --undo)")
        return 0

    for p in targets:
        if os.path.exists(p) and not os.path.exists(p + ".bak"):
            shutil.copy2(p, p + ".bak")
    for r in rows:
        cur = (r.get(col) or "").strip()
        if cur in mapping:
            r[col] = mapping[cur]
    with io.open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for p, new in json_plan.items():
        io.open(p, "w", encoding="utf-8").write(json.dumps(new, ensure_ascii=False, indent=1))
    print(f"\nĐã ghi CSV + {len(json_plan)} file JSON. Hoàn tác: --undo")
    print("TIẾP THEO (bắt buộc): đồng bộ Neo4j — chạy --cypher rồi áp, hoặc kg_import_csv_gaps.py.")
    print("SAU ĐÓ: chạy eval_gold.py, eval_disease_matching.py, test_batcuong_golden.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
