#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/clean_csv_symptoms.py — Làm sạch BẢO THỦ cột triệu_chứng của data/Medicine_clean.csv.

TUÂN THỦ RUNBOOK (scripts/README.md): KHÔNG tách field (tách làm phình mẫu số match_ratio
-> mất recall im lặng, đã mô phỏng 135 dòng). Chỉ làm các biến đổi mà số field GIẢM hoặc
GIỮ NGUYÊN -> mẫu số chỉ giảm -> recall chỉ có thể tốt lên:
  1. Bóc nhóm ngoặc '(...)' — kể cả nhóm VẮT QUA DẤU PHẨY (nguồn field rác kiểu 'ăn nhiều)'
     đang bị matcher coi là triệu chứng độc lập dù là ghi chú điều kiện).
  2. Vá ngoặc lửng: '(' không đóng -> cắt từ '(' tới hết cell.
  3. Cắt dấu câu thừa đầu/cuối field ('mạch tế nhược.' -> 'mạch tế nhược'), gộp khoảng trắng,
     bỏ field rỗng, khử field trùng trong cùng cell.
MẶC ĐỊNH DRY-RUN (in mọi thay đổi); --apply ghi backup timestamp rồi ghi đè CSV.
SAU --apply BẮT BUỘC chạy: eval_disease_matching.py + test_disease_matching.py so với baseline.
"""
import csv
import os
import re
import shutil
import sys
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

CSV_PATH = "data/Medicine_clean.csv"


def clean_cell(cell: str) -> str:
    """Làm sạch 1 cell triệu_chứng (chuỗi các triệu chứng cách nhau bằng dấu phẩy)."""
    s = cell
    # 1. Bóc nhóm ngoặc cân bằng — làm TRÊN CẢ CELL để bắt nhóm vắt qua dấu phẩy
    s = re.sub(r'\([^()]*\)', ' ', s)
    # 2. Ngoặc lửng: '(' không có ')' phía sau -> cắt tới hết cell; ')' mồ côi -> xóa
    s = re.sub(r'\([^)]*$', ' ', s)
    s = s.replace('(', ' ').replace(')', ' ')
    # 3. Từng field: cắt dấu câu thừa, gộp khoảng trắng, bỏ rỗng, khử trùng
    out = []
    for f in s.split(','):
        f = re.sub(r'\s+', ' ', f).strip()
        f = f.strip('.…;:"\'-–— ').strip()
        f = re.sub(r'\s+', ' ', f)
        if len(f) < 2:
            continue
        if f.lower() not in [x.lower() for x in out]:
            out.append(f)
    return ", ".join(out)


def main():
    apply = "--apply" in sys.argv
    if not os.path.exists(CSV_PATH):
        print(f"LỖI: không thấy {CSV_PATH} (chạy từ thư mục gốc dự án).")
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = []
    grew = 0
    for i, r in enumerate(rows):
        old = r.get("triệu_chứng") or ""
        new = clean_cell(old)
        if new != old.strip():
            n_old = len([x for x in old.split(",") if x.strip()])
            n_new = len([x for x in new.split(",") if x.strip()])
            if n_new > n_old:
                grew += 1  # bất biến bị vi phạm -> không được xảy ra
                continue
            changed.append((i, r.get("tên_bệnh", ""), old, new, n_old, n_new))
            r["triệu_chứng"] = new

    print(f"{len(rows)} dòng; đề xuất sửa {len(changed)} cell "
          f"(giảm field: {sum(1 for c in changed if c[5] < c[4])}, giữ nguyên số field: "
          f"{sum(1 for c in changed if c[5] == c[4])}; VI PHẠM tăng field: {grew} — phải là 0).")
    for i, benh, old, new, n_old, n_new in changed[:15]:
        print(f"\n  [{benh}] ({n_old} -> {n_new} field)")
        print(f"    CŨ : {old[:160]}")
        print(f"    MỚI: {new[:160]}")
    if len(changed) > 15:
        print(f"\n  ... và {len(changed) - 15} cell nữa.")

    if grew:
        print("\nLỖI BẤT BIẾN: có cell tăng số field — dừng, không apply.")
        sys.exit(1)
    if not apply:
        print("\nDRY-RUN — chưa ghi gì. Thêm --apply để thực thi (sẽ backup trước).")
        return

    backup = f"data/Medicine_clean.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy2(CSV_PATH, backup)
    print(f"\nĐã backup: {backup}")
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi {CSV_PATH} ({len(changed)} cell được làm sạch).")
    print("BẮT BUỘC chạy tiếp: python scripts/eval_disease_matching.py && python scripts/test_disease_matching.py")


if __name__ == "__main__":
    main()
