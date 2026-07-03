#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_csv_dirt.py — AUDIT (CHỈ ĐỌC) độ bẩn của data/Medicine_clean.csv.

MỤC ĐÍCH: định lượng chính xác dirt trong cột triệu_chứng & hội_chứng, VÀ mô phỏng
regression của _find_matching_diseases NẾU ta tách triệu chứng run-on (bằng chứng cho
việc TẠI SAO không được tách ngây thơ). 100% NON-DESTRUCTIVE: không sửa CSV, không chạm DB.

Chạy:  python scripts/audit_csv_dirt.py
Ghi báo cáo:  data/csv_dirt_audit.txt   (chỉ tạo file báo cáo mới, không đụng dữ liệu gốc)

Bối cảnh (đã kiểm chứng đối kháng, xem scripts/README.md mục "Runbook làm sạch an toàn"):
- CSV là phụ thuộc RUNTIME: src/fusion_pipeline.py._load_csv_data đọc trực tiếp, và
  _find_matching_diseases tính match_ratio = matched_count / len(split(',')). Do đó TÁCH
  thêm field làm PHÌNH mẫu số -> tụt ratio dưới ngưỡng 0.30 -> bệnh đúng bị loại im lặng.
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

# ---- Cấu hình khớp với app (src/fusion_pipeline.py) ----
MATCH_THRESHOLD = 0.30          # _find_matching_diseases: match_ratio >= 0.30
CSV_CANDIDATES = [
    os.getenv("TCM_CSV_PATH"),
    "data/Medicine_clean.csv",
]

STAGING_PREFIXES = (
    "biến chứng", "di chứng", "cấp tính", "mạn tính", "mãn tính", "thể ",
    "giai đoạn", "sơ nhiệt kỳ", "thời kỳ", "kỳ ", "thể nhẹ", "thể vừa", "thể nặng",
)
# Từ khoá cho thấy tên hội chứng là GHÉP HỢP LỆ (không phải staging rác)
LEGIT_COMPOUND_MARKERS = (
    "kèm", "đều hư", "lưỡng hư", "ứ trệ", "uất kết", "uất hoá", "hư hoả",
    "hạ chú", "nội động", "thượng cang", "bất hoà", "bất hòa", "thất kiện",
    "nung nấu", "huyết ứ", "khí trệ", "thấp nhiệt", "âm hư", "dương hư",
)


def find_csv():
    return next((p for p in CSV_CANDIDATES if p and os.path.exists(p)), None)


def split_fields(cell):
    """Tách theo dấu phẩy — GIỐNG HỆT _find_matching_diseases."""
    return [s.strip() for s in cell.split(",") if s.strip()]


def split_after_dot_semi(field):
    """Mô phỏng việc TÁCH THÊM run-on trên '.'/';' (KHÔNG áp dụng thật — chỉ để đo regression).
    Bỏ qua '...' (ellipsis) và số thập phân \\d\\.\\d để không tách sai."""
    # bảo vệ ellipsis và số thập phân
    tmp = field.replace("...", "")
    tmp = re.sub(r"(\d)\.(\d)", r"\1\2", tmp)
    parts = re.split(r"[.;]", tmp)
    out = [p.strip() for p in parts if p.strip()]
    return out or [field]


def is_runon(field):
    # có '.' hoặc ';' nội bộ, loại ellipsis đứng cuối và số thập phân
    f = field.replace("...", "")
    f = re.sub(r"(\d)\.(\d)", r"\1\2", f)
    return ("." in f) or (";" in f)


def is_tongue_pulse_mash(field):
    fl = field.lower()
    return ("mạch" in fl) and ("lưỡi" in fl or "rêu" in fl)


def is_prose(field, n=8):
    return len(field.split()) >= n


def unbalanced_paren(field):
    return field.count("(") != field.count(")")


def hr(f, title):
    f.write("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78 + "\n")


def main():
    path = find_csv()
    if not path:
        print("Không tìm thấy CSV (thử TCM_CSV_PATH, data/Medicine_clean.csv).")
        sys.exit(1)

    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    out_path = os.path.join("data", "csv_dirt_audit.txt")
    f = open(out_path, "w", encoding="utf-8")

    def emit(*a):
        line = " ".join(str(x) for x in a)
        f.write(line + "\n")

    emit(f"AUDIT ĐỘ BẨN — {path}  ({len(rows)} dòng)")
    emit("CHỈ ĐỌC — không sửa CSV, không chạm DB.")

    # ================= 1. TRIỆU_CHỨNG =================
    hr(f, "1. CỘT triệu_chứng")
    total_fields = 0
    c_runon = c_mash = c_prose = c_paren = c_unbal = 0
    dirty_rows = set()
    ex = defaultdict(list)
    for i, r in enumerate(rows):
        for fld in split_fields(r.get("triệu_chứng", "")):
            total_fields += 1
            bad = False
            if is_runon(fld):
                c_runon += 1; bad = True
                if len(ex["runon"]) < 6: ex["runon"].append((i, fld))
            if is_tongue_pulse_mash(fld):
                c_mash += 1; bad = True
                if len(ex["mash"]) < 6: ex["mash"].append((i, fld))
            if is_prose(fld):
                c_prose += 1; bad = True
                if len(ex["prose"]) < 6: ex["prose"].append((i, fld))
            if "(" in fld or ")" in fld:
                c_paren += 1
                if unbalanced_paren(fld):
                    c_unbal += 1
                    if len(ex["unbal"]) < 6: ex["unbal"].append((i, fld))
            if bad:
                dirty_rows.add(i)
    emit(f"Tổng field (tách theo ','): {total_fields}")
    emit(f"  - run-on ('.'/';'): {c_runon}")
    emit(f"  - mash mạch+lưỡi/rêu: {c_mash}")
    emit(f"  - prose >=8 từ:      {c_prose}")
    emit(f"  - field có ngoặc:    {c_paren}  (trong đó ngoặc LỆCH cặp: {c_unbal})")
    emit(f"  - số DÒNG có >=1 field bẩn: {len(dirty_rows)} / {len(rows)}")
    for k, lbl in [("runon", "run-on"), ("mash", "mash mạch+lưỡi"), ("prose", "prose dài"), ("unbal", "ngoặc lệch cặp")]:
        emit(f"  ví dụ [{lbl}]:")
        for i, s in ex[k]:
            emit(f"    dòng{i+2}: {s[:100]}")

    # ================= 2. REGRESSION nếu tách run-on =================
    hr(f, "2. MÔ PHỎNG REGRESSION — nếu TÁCH triệu chứng run-on trên '.'/';'")
    emit("Quy tắc app: match_ratio = matched_count / len(fields), PASS khi ratio>=0.30 & matched>=2 & specific>=1.")
    emit("Tách làm TĂNG mẫu số => ngưỡng matched tối thiểu để PASS tăng => bệnh đang khớp có thể rớt.")
    import math
    at_risk = []
    for i, r in enumerate(rows):
        fields = split_fields(r.get("triệu_chứng", ""))
        n_now = len(fields)
        if n_now == 0:
            continue
        n_after = sum(len(split_after_dot_semi(fld)) for fld in fields)
        if n_after <= n_now:
            continue
        # matched tối thiểu để đạt ratio 0.30 (và >=2)
        need_now = max(2, math.ceil(MATCH_THRESHOLD * n_now))
        need_after = max(2, math.ceil(MATCH_THRESHOLD * n_after))
        if need_after > need_now:
            at_risk.append((i, r.get("tên_bệnh", ""), n_now, n_after, need_now, need_after))
    emit(f"Số DÒNG mà việc tách LÀM TĂNG ngưỡng matched tối thiểu (nguy cơ rớt bệnh): {len(at_risk)}")
    emit("  (dòng | bệnh | #field trước->sau | matched tối thiểu trước->sau)")
    for i, b, na, nb, ra, rb in at_risk[:25]:
        emit(f"    dòng{i+2}: {b[:34]:34s} {na:2d}->{nb:2d}   cần {ra}->{rb}")
    if len(at_risk) > 25:
        emit(f"    ... và {len(at_risk)-25} dòng nữa.")
    emit("\n=> KẾT LUẬN: KHÔNG tách triệu chứng trong CSV nếu chưa re-tune ngưỡng 0.30 + test recall.")

    # ================= 3. HỘI_CHỨNG =================
    hr(f, "3. CỘT hội_chứng")
    hc = Counter(r.get("hội_chứng", "").strip() for r in rows if r.get("hội_chứng", "").strip())
    uniq = sorted(hc)
    emit(f"{sum(hc.values())} ô có giá trị | {len(uniq)} tên distinct")

    def classify_hc(name):
        nl = name.lower()
        legit = any(m in nl for m in LEGIT_COMPOUND_MARKERS)
        if any(nl.startswith(p) for p in STAGING_PREFIXES) and (" - " in name or "(" in name):
            return "STAGING (nghi rác — nên bóc/sửa tay)"
        if " - " in name:
            return "GHÉP HỢP LỆ (giữ)" if legit else "DẤU ' - ' (cần phân loại tay)"
        if "/" in name:
            return "DẤU '/' 2 hội chứng (chỉ report, tách phá {name:$syn})"
        if "(" in name or ")" in name:
            return "NGOẶC chú thích (normalize-names xử lý được phần lớn)"
        if any(ch.isdigit() for ch in name):
            return "CHỨA SỐ (nghi Bài N/Nghiệm phương)"
        if len(name) > 40:
            return "GHÉP HỢP LỆ >40 ký tự (giữ)" if legit else ">40 ký tự (rà tay — có thể false-positive)"
        return "CLEAN"

    buckets = defaultdict(list)
    for name in uniq:
        buckets[classify_hc(name)].append(name)
    for cat in sorted(buckets, key=lambda c: -len(buckets[c])):
        emit(f"\n  [{cat}] : {len(buckets[cat])}")
        for name in buckets[cat][:12]:
            emit(f"    - {name[:80]}")
        if len(buckets[cat]) > 12:
            emit(f"    ... +{len(buckets[cat])-12} nữa")

    f.close()
    # In tóm tắt ra stdout
    print(f"[OK] Đã ghi báo cáo: {out_path}")
    print(f"  triệu_chứng: {total_fields} field, {len(dirty_rows)} dòng có field bẩn "
          f"(run-on {c_runon}, mash {c_mash}, prose {c_prose}, ngoặc lệch {c_unbal})")
    print(f"  REGRESSION: {len(at_risk)} dòng sẽ tăng ngưỡng matched nếu tách run-on (=> KHÔNG tách ngây thơ)")
    print(f"  hội_chứng: {len(uniq)} distinct -> phân loại:")
    for cat in sorted(buckets, key=lambda c: -len(buckets[c])):
        print(f"    {len(buckets[cat]):4d}  {cat}")


if __name__ == "__main__":
    main()
