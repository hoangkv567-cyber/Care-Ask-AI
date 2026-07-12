#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/dedup_csv_rows.py — Gộp DÒNG CSV TRÙNG LẶP (cùng bệnh × hội chứng × bài).

CSV có nhiều dòng CÙNG (tên_bệnh, hội_chứng, bài_thuốc) với danh sách vị thuốc gần
trùng (biến thể chính tả/bào chế: 'Tri mẫu (tẩm muối)' vs 'Tri mẫu Tẩm muối'). Khi
import graph, các dòng này gộp vào MỘT node BaiThuoc -> node ôm UNION vị thuốc ->
Mục 5 in trùng vị. Giữ dòng NHIỀU VỊ NHẤT mỗi cụm (tiebreak: dòng đầu), xóa còn lại.

CHỈ gộp cụm CÙNG BÀI (exact triple). Cụm khác bài (2+ bài/hội chứng) là hợp lệ (nhiều
phương án) -> GIỮ NGUYÊN.

An toàn: MẶC ĐỊNH DRY-RUN. --apply mới ghi (tự backup .bak).
    python scripts/dedup_csv_rows.py            # dry-run
    python scripts/dedup_csv_rows.py --apply
"""
import argparse
import collections
import csv
import os
import re
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(_ROOT, "data", "Medicine_clean.csv")


def main():
    ap = argparse.ArgumentParser(description="Gộp dòng CSV trùng bệnh×hội chứng×bài. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    raw = open(CSV, encoding="utf-8-sig").read()
    lines = raw.split("\n")
    grp = collections.defaultdict(list)          # (benh,hc,bai) -> [(line_idx, cols)]
    for idx, ln in enumerate(lines):
        if idx == 0 or not ln.strip():
            continue
        try:
            c = next(csv.reader([ln]))
        except Exception:
            continue
        if len(c) >= 8:
            grp[(c[0].strip(), c[1].strip(), c[6].strip())].append((idx, c))

    def nherb(c):
        return len([x for x in c[7].split(",") if x.strip()])

    def hset(c):
        return {re.sub(r"\s+", " ", x.strip().lower()) for x in c[7].split(",") if x.strip()}

    drop = set()
    kept_diff = []                             # bài TRÙNG TÊN nhưng vị KHÁC (không gộp)
    for (b, h, n), v in grp.items():
        if len(v) <= 1:
            continue
        # giữ dòng nhiều vị nhất; tiebreak dòng đầu
        keep, *rest = sorted(v, key=lambda t: (-nherb(t[1]), t[0]))
        kh = hset(keep[1])
        dropped = []
        for idx, c in rest:
            rh = hset(c)
            jac = len(kh & rh) / len(kh | rh) if (kh | rh) else 1.0
            if jac >= 0.6:                     # chỉ gộp khi GẦN TRÙNG (biến thể chính tả/bào chế)
                drop.add(idx)
                dropped.append(idx)
            else:                              # KHÁC bài thật (cùng principle-name) -> GIỮ
                kept_diff.append((b, h, n, idx, jac))
        if dropped:
            print(f"  {b} × {h} -> {n}: giữ dòng {keep[0]+1} ({nherb(keep[1])} vị), "
                  f"xóa {[i+1 for i in dropped]}")
    for b, h, n, idx, jac in kept_diff:
        print(f"  ⚠️ GIỮ (khác bài, overlap {jac:.0%}): {b} × {h} -> {n} dòng {idx+1} "
              f"(bài trùng tên nhưng vị thuốc khác — KHÔNG gộp)")

    print(f"\nTổng dòng trùng sẽ XÓA: {len(drop)}")
    if not drop:
        print("Không có gì để gộp.")
        return
    if not args.apply:
        print("\n(DRY-RUN — thêm --apply để ghi.)")
        return

    shutil.copyfile(CSV, CSV + ".bak")
    out = [ln for idx, ln in enumerate(lines) if idx not in drop]
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        f.write("\n".join(out))
    print(f"\nĐã ghi (backup: {os.path.basename(CSV)}.bak). Còn "
          f"{len([l for l in out if l.strip()]) - 1} dòng data.")


if __name__ == "__main__":
    main()
