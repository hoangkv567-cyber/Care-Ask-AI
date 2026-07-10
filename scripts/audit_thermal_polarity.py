#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit_thermal_polarity.py — AUDIT (CHỈ ĐỌC) xung khắc CỰC HÀN/NHIỆT giữa
NHÃN hội chứng và TÍNH VỊ THUỐC trong data/Medicine_clean.csv.

VÌ SAO: một dòng có thể bị dán nhãn hội chứng HÀN (cần bài tân ÔN) nhưng bài thuốc lại
toàn vị HÀN/LƯƠNG (vd dòng cũ 'Viêm phế quản × Phong hàn → Tang bạch thang' — Tang diệp/
Sa sâm/Xuyên bối mẫu/Chi tử, herb-mean ≈ -0.88). Guard nhãn-vs-nhãn (_syndromes_thermal_
conflict) KHÔNG bắt được vì mâu thuẫn nằm ở NHÃN vs VỊ THUỐC. Audit này tính cực nhiệt học
của bài THEO TÍNH TỪNG VỊ rồi đối chiếu hướng pháp trị mà nhãn hội chứng đòi hỏi.

CỔNG HƯ/THỰC (điểm mấu chốt so với shortlist.json cũ): CHỈ soi hội chứng THỰC có cực hàn/
nhiệt rõ. Hội chứng HƯ (Âm hư/Huyết hư/Dương hư...) bị LOẠI khỏi phạm vi vì hư nhiệt được
TƯ BỔ (bài tư âm có thể ấm nhẹ, vd Nhục quế dẫn hỏa quy nguyên) chứ không THANH — nếu soi
sẽ cờ oan hàng loạt ca hư đang ĐÚNG (đó chính là dương-tính-giả #2..#7 của shortlist.json).

100% NON-DESTRUCTIVE: không sửa CSV, không chạm DB.

Chạy:   python scripts/audit_thermal_polarity.py
Cần:    data/herb_thermal.json  (từ điển tính vị; sinh bởi workflow herb-thermal-lexicon)
Xuất:   data/thermal_audit_worklist.json  (danh sách nghi vấn, xếp theo mức nghiêm trọng)
        data/thermal_audit_report.txt     (báo cáo người đọc)
"""
import csv
import json
import os
import re
import sys

# ---- Cấu hình ----
CSV_CANDIDATES = [os.getenv("TCM_CSV_PATH"), "data/Medicine_clean.csv"]
LEXICON_PATH = "data/herb_thermal.json"
OUT_WORKLIST = "data/thermal_audit_worklist.json"
OUT_REPORT = "data/thermal_audit_report.txt"

# Ngưỡng cờ theo herb-mean (điểm vị: nhiệt=+2 ôn=+1 bình=0 lương=-1 hàn=-2).
FLAG_WEAK = 0.20      # sai hướng & |mean| >= 0.20  -> XEM LẠI
FLAG_STRONG = 0.50    # sai hướng & |mean| >= 0.50  -> NGHI VẤN MẠNH
MIN_KNOWN = 3         # cần >=3 vị có trong từ điển mới đủ tin
MIN_COVERAGE = 0.40   # và >=40% số vị được biết tính


# ---- Phân loại hội chứng: PORT NGUYÊN VĂN từ src/fusion_pipeline.py ----
def syndrome_is_hu(name: str) -> bool:
    """Khớp src/fusion_pipeline.py::_syndrome_is_hu (ranh giới từ)."""
    return bool(re.search(r'\b(hư|suy|bất túc|nhược|khuy|tổn|thiếu)\b', (name or "").lower()))


def syndrome_thermal_sign(name):
    """Khớp src/fusion_pipeline.py::_syndrome_thermal_sign.
    'han' | 'nhiet' | None (trung tính hoặc lẫn cực). Táo/ôn/thử/hỏa xếp NHIỆT."""
    nl = (name or "").lower()
    han = bool(re.search(r'\b(hàn|lạnh)\b', nl))
    nhiet = bool(re.search(r'\b(nhiệt|hỏa|hoả|ôn|thử|táo)\b', nl))
    if han and not nhiet:
        return "han"
    if nhiet and not han:
        return "nhiet"
    return None


def split_herbs(vi):
    """Tách vị thuốc: bỏ ghi chú trong ngoặc (Gia/nếu...), tách theo , ; . ;
    loại mệnh đề chú thích và câu dài (không phải tên vị)."""
    if not vi:
        return []
    vi = re.sub(r'\([^)]*\)', ' ', vi)
    out = []
    for p in re.split(r'[,;.]', vi):
        h = p.strip().strip('.').strip()
        if not h:
            continue
        if re.search(r'gia\b|giảm|nếu|hoặc|liều|thêm|bớt', h.lower()):
            continue
        if len(h) > 30:
            continue
        out.append(h)
    return out


def col(row, name):
    return (row.get(name) or "").strip()


def find_csv():
    return next((p for p in CSV_CANDIDATES if p and os.path.exists(p)), None)


def main():
    csv_path = find_csv()
    if not csv_path:
        print("Không tìm thấy CSV.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(LEXICON_PATH):
        print(f"Thiếu từ điển tính vị {LEXICON_PATH} — chạy workflow herb-thermal-lexicon trước.",
              file=sys.stderr)
        sys.exit(1)

    lex_raw = json.load(open(LEXICON_PATH, encoding="utf-8"))
    # chấp nhận {herb: ...} phẳng HOẶC bọc trong khóa 'herbs' (có _meta)
    if isinstance(lex_raw, dict) and "herbs" in lex_raw and isinstance(lex_raw["herbs"], dict):
        lex_raw = lex_raw["herbs"]
    # chấp nhận cả {herb: score} lẫn {herb: {score,...}}
    lex = {}
    for k, v in lex_raw.items():
        lex[k.strip().lower()] = (v if isinstance(v, (int, float)) else v.get("score", 0))

    with open(csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    flagged = []
    n_thuc_scope = 0        # số dòng trong phạm vi soi (Thực + cực rõ + có bài/vị)
    n_hu_skipped = 0
    n_neutral = 0
    n_lowcov = 0
    unknown_herbs = {}

    for i, r in enumerate(rows):
        ln = i + 2  # header = dòng 1
        hc = col(r, "hội_chứng")
        benh = col(r, "tên_bệnh")
        bai = col(r, "bài_thuốc")
        vi = col(r, "vị_thuốc")

        if syndrome_is_hu(hc):
            n_hu_skipped += 1
            continue
        sign = syndrome_thermal_sign(hc)
        if not sign:
            n_neutral += 1
            continue
        herbs = split_herbs(vi)
        if not bai or not herbs:
            continue

        n_thuc_scope += 1
        scores, known, unknown = [], [], []
        for h in herbs:
            key = h.lower().strip()
            if key in lex:
                scores.append(lex[key])
                known.append(h)
            else:
                unknown.append(h)
                unknown_herbs[key] = unknown_herbs.get(key, 0) + 1

        n_known, n_total = len(scores), len(herbs)
        if n_known < MIN_KNOWN or (n_total and n_known / n_total < MIN_COVERAGE):
            n_lowcov += 1
            continue

        mean = sum(scores) / n_known

        # HÀN cần bài ẤM (mean>0); NHIỆT (thực) cần bài LẠNH (mean<0).
        # wrong = độ lệch SAI HƯỚNG (>0 nghĩa là sai cực).
        if sign == "han":
            wrong = -mean          # bài càng lạnh (mean âm) -> wrong dương -> càng sai
            need = "ẤM (tân ôn)"
        else:  # nhiet
            wrong = mean           # bài càng ấm (mean dương) -> wrong dương -> càng sai
            need = "LẠNH (thanh/lương)"

        if wrong >= FLAG_WEAK:
            sev = "STRONG" if wrong >= FLAG_STRONG else "REVIEW"
            cold = [h for h in known if lex[h.lower().strip()] < 0]
            warm = [h for h in known if lex[h.lower().strip()] > 0]
            flagged.append({
                "line": ln,
                "benh": benh,
                "hc": hc,
                "sign": sign.upper(),
                "need": need,
                "bai": bai,
                "mean": round(mean, 3),
                "wrong": round(wrong, 3),
                "severity": sev,
                "n_known": n_known,
                "n_total": n_total,
                "warm": warm,
                "cold": cold,
                "unknown": unknown,
            })

    flagged.sort(key=lambda x: -x["wrong"])

    os.makedirs("data", exist_ok=True)
    json.dump(flagged, open(OUT_WORKLIST, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    strong = [f for f in flagged if f["severity"] == "STRONG"]
    review = [f for f in flagged if f["severity"] == "REVIEW"]

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        w = lambda *a: f.write(" ".join(str(x) for x in a) + "\n")
        w(f"AUDIT XUNG CỰC HÀN/NHIỆT (nhãn hội chứng vs tính vị thuốc) — {csv_path}")
        w(f"Tổng dòng: {len(rows)} | Từ điển tính vị: {len(lex)} vị")
        w(f"Loại HƯ (ngoài phạm vi, tư bổ): {n_hu_skipped}")
        w(f"Trung tính/không cực: {n_neutral}")
        w(f"THỰC + cực rõ + có bài/vị (phạm vi soi): {n_thuc_scope}")
        w(f"Bỏ vì thiếu phủ tính vị (<{MIN_KNOWN} vị biết hoặc <{int(MIN_COVERAGE*100)}%): {n_lowcov}")
        w(f"=> NGHI VẤN: {len(flagged)}  (STRONG={len(strong)}, REVIEW={len(review)})")
        w("")
        for title, items in [("NGHI VẤN MẠNH (STRONG)", strong), ("XEM LẠI (REVIEW)", review)]:
            w("=" * 78)
            w(title)
            w("=" * 78)
            for x in items:
                w(f"[dòng {x['line']}] {x['benh']} × {x['hc']}  (cần bài {x['need']})")
                w(f"    Bài: {x['bai']}   herb-mean={x['mean']}  (biết {x['n_known']}/{x['n_total']} vị)")
                if x["cold"]:
                    w(f"    Vị HÀN/LƯƠNG: {', '.join(x['cold'])}")
                if x["warm"]:
                    w(f"    Vị ÔN/NHIỆT: {', '.join(x['warm'])}")
                w("")
        # vị chưa có trong từ điển (để mở rộng lexicon sau)
        if unknown_herbs:
            miss = sorted(unknown_herbs.items(), key=lambda t: -t[1])
            w("=" * 78)
            w(f"VỊ CHƯA CÓ TRONG TỪ ĐIỂN ({len(miss)}) — cân nhắc bổ sung nếu xuất hiện nhiều:")
            w("=" * 78)
            w(", ".join(f"{h}({n})" for h, n in miss[:60]))

    print(f"THỰC trong phạm vi soi: {n_thuc_scope} | HƯ loại: {n_hu_skipped} | "
          f"low-coverage bỏ: {n_lowcov}")
    print(f"NGHI VẤN: {len(flagged)} (STRONG={len(strong)}, REVIEW={len(review)})")
    print(f"-> {OUT_WORKLIST}")
    print(f"-> {OUT_REPORT}")
    print()
    print("TOP nghi vấn:")
    for x in flagged[:15]:
        print(f"  [{x['severity']:6}] dòng {x['line']:4}  {x['benh']} × {x['hc']} "
              f"| cần {x['need']} | mean={x['mean']:+.2f} | {x['bai']}")


if __name__ == "__main__":
    main()
