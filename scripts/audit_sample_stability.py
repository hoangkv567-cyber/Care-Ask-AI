# -*- coding: utf-8 -*-
"""Audit ổn định MẪU: chọn ~40 bệnh đa dạng (NGOÀI gold), sinh lời khai từ chủ_chứng CSV,
chạy pipeline SỐNG, tính CỜ lỗi tự động (Mục 5 trắng / bài trái cực / Tiêu-Thực mâu thuẫn /
không tự gọi tên bệnh). Xuất data/audit_sample_report.json. Chạy nền (~20-40 phút)."""
import sys, io, json, os, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from src.fusion_pipeline import TCMFusionPipeline as F

N = int(os.getenv("AUDIT_N", "40"))
random.seed(42)

p = F()
# distinct diseases + their primary row (dòng đầu tiên)
first_row = {}
for r in p.csv_rows:
    b = r.get("benh_ly", "").strip()
    if b and b not in first_row:
        first_row[b] = r
diseases = list(first_row.keys())

# gold-covered để LOẠI (audit phần chưa test)
gold_cov = set()
try:
    gd = json.load(open("data/gold_cases.json", encoding="utf-8"))
    gcs = gd if isinstance(gd, list) else gd.get("cases", gd.get("gold", []))
    for c in gcs:
        e = c.get("expect_disease") or c.get("expect") or ""
        if isinstance(e, list): gold_cov.update(x.strip().lower() for x in e)
        elif e: gold_cov.add(e.strip().lower())
except Exception:
    pass

cands = [d for d in diseases if d.strip().lower() not in gold_cov]
sample = sorted(random.sample(cands, min(N, len(cands))))


def gen_input(row):
    cc = [s.strip() for s in row.get("chu_chung", "").split(",") if s.strip()]
    tc = [s.strip() for s in row.get("triệu_chứng", "").split(",") if s.strip()]
    # ưu tiên chủ chứng; bù thêm vài triệu chứng để đủ ngưỡng khớp
    terms = []
    for s in (cc + tc):
        sl = s.lower()
        if sl not in [t.lower() for t in terms] and not F._is_pulse_field(sl):
            terms.append(s)
        if len(terms) >= 6:
            break
    return ", ".join(terms[:6])


THERMAL_STRONG = 0.6


def analyze(disease, ans):
    low = ans.lower()
    flags = []
    # bệnh danh dòng
    m_bd = re.search(r"bệnh danh[^\n:]*:\s*([^\n]+)", low)
    bd = m_bd.group(1).strip() if m_bd else ""
    m_core = re.search(r"cốt lõi[^\n:]*:\s*([^\n]+)", low)
    core = m_core.group(1).strip() if m_core else ""
    m_bc = re.search(r"thuộc chứng[^\n:]*:\s*([^\n]+)", low)
    bc = m_bc.group(1).strip() if m_bc else ""
    # 1. Mục 5 trắng
    if "chưa có bài thuốc" in low or "chưa cập nhật bài thuốc" in low or "hiện chưa có bài" in low:
        flags.append("MUC5_TRANG")
    # 2. không tự gọi tên bệnh (self-name miss)
    dn = disease.lower().split("(")[0].strip()
    if dn and dn not in bd:
        flags.append("KHONG_TU_GOI_TEN")
    # 3. Tiêu-Thực mâu thuẫn: Mục 4 'Không có Tiêu Thực' NHƯNG Mục 5 có dòng 'Tiêu'
    has_no_tieu = "không có tiêu thực" in low
    has_tieu_formula = bool(re.search(r"tiêu\s*[–-]\s*(nhánh|thể kb khớp hội chứng kèm)", low))
    if has_no_tieu and has_tieu_formula:
        flags.append("TIEU_THUC_MAU_THUAN")
    # 4. bài trái cực: lấy vị thuốc từng dòng, so tính vị với Bát Cương
    polar = None
    if ("hàn" in bc and "nhiệt" not in bc):
        polar = "han"   # cần bài ẤM
    elif ("nhiệt" in bc and "hàn" not in bc):
        polar = "nhiet"  # cần bài LẠNH
    thermal_bad = []
    for mvi in re.finditer(r"\*vị thuốc:\*\s*([^\n]+)", low):
        mean = F._formula_thermal_mean(mvi.group(1))
        if polar == "han" and mean <= -THERMAL_STRONG:
            thermal_bad.append(round(mean, 2))
        elif polar == "nhiet" and mean >= THERMAL_STRONG:
            thermal_bad.append(round(mean, 2))
    if thermal_bad:
        flags.append(f"BAI_TRAI_CUC({polar}:{thermal_bad})")
    return {"benh_danh": bd, "core": core, "bat_cuong": bc, "flags": flags}


results = []
for i, d in enumerate(sample):
    inp = gen_input(first_row[d])
    rec = {"idx": i, "disease": d, "input": inp}
    try:
        out = p.run_diagnosis(user_symptoms=inp)
        ans = out["diagnosis_result"]["answer"] if isinstance(out, dict) and "diagnosis_result" in out else str(out)
        rec.update(analyze(d, ans))
        rec["answer"] = ans
    except Exception as e:
        rec["error"] = repr(e)[:300]
        rec["flags"] = ["EXCEPTION"]
    results.append(rec)
    print(f"[{i+1}/{len(sample)}] {d[:40]:40s} flags={rec.get('flags')}", flush=True)

json.dump(results, open("data/audit_sample_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# summary
flagged = [r for r in results if r.get("flags")]
print(f"\n==== XONG: {len(results)} ca, {len(flagged)} ca CÓ CỜ ====")
from collections import Counter
cnt = Counter(f.split("(")[0] for r in flagged for f in r["flags"])
for k, v in cnt.most_common():
    print(f"  {k}: {v}")
print("-> data/audit_sample_report.json")
