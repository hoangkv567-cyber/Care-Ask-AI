#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
ĐỒNG BỘ TRIỆU CHỨNG CSV -> KNOWLEDGE GRAPH (có THÊM và có XÓA)
================================================================================
scripts/kg_import_csv_gaps.py chỉ BÙ phần graph còn THIẾU — nó không bao giờ xóa. Nên khi một dòng
CSV được SỬA (triệu chứng cũ thay bằng triệu chứng mới), graph giữ nguyên cái cũ và hai bên lệch
nhau âm thầm — đúng lớp lỗi "graph không tự đồng bộ khi sửa CSV".

Script này đồng bộ HAI CHIỀU cho phạm vi HẸP do người gọi chỉ định (một bệnh, hoặc một cặp
bệnh × hội chứng): cạnh (HoiChung)-[:CÓ_BIỂU_HIỆN {benh_ly}]->(TrieuChung) phải khớp ĐÚNG danh
sách triệu chứng của dòng CSV tương ứng.

⚠️ PHẠM VI DÙNG — ĐỌC TRƯỚC KHI CHẠY DIỆN RỘNG:
Bản nhập graph GỐC tách triệu chứng KHÁC với CSV: nó tách theo cả dấu CHẤM và bỏ liên từ đầu cụm.
    CSV  : "chóng mặt ù tai. Chất lưỡi đỏ ít tân dịch"   (một ô)
    GRAPH: "chóng mặt ù tai" + "Chất lưỡi đỏ ít tân dịch" (hai node)
    CSV  : "kèm ngũ tâm phiền nhiệt"  ->  GRAPH: "ngũ tâm phiền nhiệt"
tức GRAPH CHUẨN HOÁ TỐT HƠN CSV. Đã đo: tách theo dấu phẩy đơn thuần cho 288 cặp "lệch"; tách
phẩy+chấm+bỏ liên từ (hàm dưới) còn 244 — VẪN chưa tái tạo hết quy ước gốc.

=> Công cụ này chỉ dùng cho SỬA ĐIỂM một dòng CSV vừa đổi, và phải NHÌN kế hoạch dry-run rồi mới
--apply. TUYỆT ĐỐI không quét cả KB: phần "graph dư" phần lớn là node đã tách đúng, xoá đi là làm
graph TỆ ĐI. Vì vậy có ngưỡng chặn số cạnh XOÁ mỗi lần chạy (--max-change, mặc định 3).

AN TOÀN:
  - MẶC ĐỊNH DRY-RUN. Phải có --apply mới ghi.
  - BẮT BUỘC --disease: không có chế độ "đồng bộ tất cả" để tránh một lệnh quét sạch graph.
  - THÊM+XOÁ quá --max-change cạnh một lần chạy -> DỪNG (dấu hiệu đang so sai quy ước tách).
  - Rollback ghi RIÊNG theo từng bệnh, không đè lẫn nhau.
  - CHỈ đụng cạnh có r.benh_ly = đúng tên bệnh đó -> triệu chứng dùng chung với BỆNH KHÁC không suy
    suyển. TUYỆT ĐỐI không xóa node TrieuChung (node còn được bệnh khác dùng).
  - So khớp KHÔNG phân biệt hoa/thường (tái dùng node và tên hội chứng sẵn có, không đẻ biến thể).
  - Ghi data/<...>_rollback.json ghi lại CHÍNH XÁC cạnh đã thêm/xóa kèm thuộc tính gốc -> --undo
    khôi phục nguyên trạng.

Dùng:
    python scripts/sync_csv_symptoms_to_kg.py --disease "Kinh hành đầu thống"
    python scripts/sync_csv_symptoms_to_kg.py --disease "Kinh hành đầu thống" --syndrome "Âm hư" --apply
    python scripts/sync_csv_symptoms_to_kg.py --undo data/kg_symptom_sync_<bệnh>_rollback.json
================================================================================
"""
import argparse
import csv
import json
import os
import re
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(_ROOT, "data", "Medicine_clean.csv")
def _rollback_path(disease):
    """Rollback RIÊNG cho từng bệnh. Tên file cố định từng làm lần chạy sau ĐÈ MẤT rollback lần
    trước — mất luôn đường lui của thay đổi đã ghi."""
    slug = re.sub(r"[^\w]+", "_", (disease or "").strip().lower()).strip("_")[:60]
    return os.path.join(_ROOT, "data", f"kg_symptom_sync_{slug}_rollback.json")
NGUON = "csv_symptom_sync"


def _load_env():
    p = os.path.join(_ROOT, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _driver():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI")
    if not uri:
        print("LỖI: thiếu NEO4J_URI trong .env")
        sys.exit(2)
    return GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"),
                                           os.getenv("NEO4J_PASSWORD")))


def split_symptoms(s):
    """Tách triệu chứng THEO QUY ƯỚC GRAPH (xem đầu file): dấu phẩy VÀ dấu chấm/chấm phẩy, bỏ liên
    từ mở đầu ('kèm', 'kèm theo', 'và', 'còn'). Vẫn chưa khớp 100% bản nhập gốc — đó là lý do có
    ngưỡng --max-remove."""
    import re
    out = []
    for chunk in re.split(r"[,.;]", s or ""):
        c = re.sub(r"^\s*(?:kèm theo|kèm|và|còn)\s+", "", chunk.strip(), flags=re.I).strip()
        if c:
            out.append(c)
    return out


def _csv_rows(disease, syndrome=None):
    """Dòng CSV của bệnh (và hội chứng nếu chỉ định) -> [(hoi_chung, [triệu chứng...])]."""
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    if not rows:
        return []
    k = list(rows[0].keys())
    col_b, col_h, col_t = k[0], k[1], k[3]      # tên_bệnh, hội_chứng, triệu_chứng
    out = []
    for r in rows:
        if (r.get(col_b) or "").strip().lower() != disease.strip().lower():
            continue
        hc = (r.get(col_h) or "").strip()
        if syndrome and hc.lower() != syndrome.strip().lower():
            continue
        tcs = split_symptoms(r.get(col_t) or "")
        if hc and tcs:
            out.append((hc, tcs))
    return out


def _graph_state(sess, disease, hc):
    """Cạnh CÓ_BIỂU_HIỆN hiện có của (hội chứng × bệnh) -> {tên triệu chứng: props}."""
    q = ("MATCH (h:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung) "
         "WHERE toLower(h.name) = toLower($h) AND r.benh_ly = $b "
         "RETURN t.name AS tc, properties(r) AS p")
    return {x["tc"]: x["p"] for x in sess.run(q, h=hc, b=disease)}


def plan(sess, disease, syndrome):
    todo = []
    for hc, want in _csv_rows(disease, syndrome):
        have = _graph_state(sess, disease, hc)
        have_l = {t.lower(): t for t in have}
        want_l = {t.lower(): t for t in want}
        add = [want_l[x] for x in want_l if x not in have_l]
        rem = [have_l[x] for x in have_l if x not in want_l]
        if add or rem:
            todo.append({"hoi_chung": hc, "add": add,
                         "remove": [{"tc": t, "props": have[t]} for t in rem]})
    return todo


def show(disease, todo):
    if not todo:
        print(f"Graph ĐÃ khớp CSV cho bệnh '{disease}' — không có gì để làm.")
        return
    for item in todo:
        print(f"  [{item['hoi_chung']}]")
        for t in item["add"]:
            print(f"      + THÊM  : {t}")
        for r in item["remove"]:
            print(f"      - XÓA   : {r['tc']}   (props gốc: {r['props']})")


def apply_changes(sess, disease, todo):
    log = {"disease": disease, "added": [], "removed": []}
    for item in todo:
        hc = item["hoi_chung"]
        for tc in item["add"]:
            # MERGE node theo tên sẵn có (không phân biệt hoa/thường) để không đẻ node trùng
            sess.run(
                "MATCH (h:HoiChung) WHERE toLower(h.name) = toLower($h) "
                "WITH h LIMIT 1 "
                "OPTIONAL MATCH (old:TrieuChung) WHERE toLower(old.name) = toLower($t) "
                "WITH h, collect(old)[0] AS old "
                "FOREACH (_ IN CASE WHEN old IS NULL THEN [1] ELSE [] END | "
                "  MERGE (n:TrieuChung {name:$t}) SET n._nguon = $src "
                "  MERGE (h)-[r:CÓ_BIỂU_HIỆN {benh_ly:$b}]->(n) ON CREATE SET r.nguon = $src) "
                "FOREACH (o IN CASE WHEN old IS NULL THEN [] ELSE [old] END | "
                "  MERGE (h)-[r:CÓ_BIỂU_HIỆN {benh_ly:$b}]->(o) ON CREATE SET r.nguon = $src)",
                h=hc, t=tc, b=disease, src=NGUON)
            log["added"].append({"hoi_chung": hc, "tc": tc})
        for r in item["remove"]:
            # CHỈ xóa CẠNH đúng phạm vi bệnh này — node giữ nguyên cho bệnh khác dùng
            sess.run(
                "MATCH (h:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung) "
                "WHERE toLower(h.name) = toLower($h) AND r.benh_ly = $b AND t.name = $t "
                "DELETE r", h=hc, t=r["tc"], b=disease)
            log["removed"].append({"hoi_chung": hc, "tc": r["tc"], "props": r["props"]})
    return log


def undo(sess, path):
    log = json.load(open(path, encoding="utf-8"))
    disease = log["disease"]
    for a in log.get("added", []):
        sess.run("MATCH (h:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung) "
                 "WHERE toLower(h.name)=toLower($h) AND r.benh_ly=$b AND t.name=$t AND r.nguon=$src "
                 "DELETE r", h=a["hoi_chung"], t=a["tc"], b=disease, src=NGUON)
    for r in log.get("removed", []):
        sess.run("MATCH (h:HoiChung) WHERE toLower(h.name)=toLower($h) WITH h LIMIT 1 "
                 "MATCH (t:TrieuChung {name:$t}) "
                 "MERGE (h)-[e:CÓ_BIỂU_HIỆN {benh_ly:$b}]->(t) SET e += $p",
                 h=r["hoi_chung"], t=r["tc"], b=disease,
                 p={k: v for k, v in (r.get("props") or {}).items() if v is not None})
    print(f"Đã hoàn tác: gỡ {len(log.get('added', []))} cạnh thêm, khôi phục "
          f"{len(log.get('removed', []))} cạnh xóa.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", help="tên bệnh (BẮT BUỘC trừ khi --undo)")
    ap.add_argument("--syndrome", help="chỉ đồng bộ một hội chứng của bệnh đó")
    ap.add_argument("--apply", action="store_true", help="ghi graph thật (mặc định dry-run)")
    ap.add_argument("--max-change", type=int, default=3,
                    help="chặn: THÊM+XOÁ quá N cạnh trong một lần chạy thì DỪNG (mặc định 3)")
    ap.add_argument("--undo", metavar="ROLLBACK_JSON", help="hoàn tác theo file rollback")
    args = ap.parse_args()

    _load_env()
    drv = _driver()

    if args.undo:
        with drv.session() as s:
            undo(s, args.undo)
        return 0

    if not args.disease:
        print("LỖI: cần --disease (không có chế độ đồng bộ toàn bộ — quá nguy hiểm).")
        return 2

    with drv.session() as s:
        todo = plan(s, args.disease, args.syndrome)
        print(f"Bệnh: {args.disease}" + (f" | hội chứng: {args.syndrome}" if args.syndrome else ""))
        show(args.disease, todo)
        if not todo:
            return 0
        # Ngưỡng tính CẢ THÊM lẫn XOÁ. Đo được khi lỡ chạy --apply lên 'Hậu môn nứt kẽ': tách sai
        # sinh ra mảnh vụn vô nghĩa ('lượng ít', 'kéo dài hàng giờ') và chúng được THÊM vào graph
        # dù KHÔNG xoá cạnh nào. Thêm rác làm hỏng graph y hệt xoá nhầm -> ngưỡng phải chặn cả hai.
        n_chg = sum(len(i["remove"]) + len(i["add"]) for i in todo)
        if n_chg > args.max_change:
            print(f"\nDỪNG: kế hoạch đổi {n_chg} cạnh (> --max-change={args.max_change}).")
            print("Quy ước tách triệu chứng của graph KHÁC CSV (xem đầu file), nên số thay đổi lớn")
            print("thường nghĩa là đang so sai — ghi vào sẽ làm graph TỆ HƠN chứ không tốt hơn.")
            print("Rà TỪNG dòng ở trên; chắc chắn rồi mới nâng --max-change.")
            return 1
        if not args.apply:
            print("\n(DRY-RUN — thêm --apply để ghi thật.)")
            return 0
        log = apply_changes(s, args.disease, todo)

    rb = _rollback_path(args.disease)
    with open(rb, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"\nĐã ghi graph. Rollback: {rb}")
    print(f"Hoàn tác: python scripts/sync_csv_symptoms_to_kg.py --undo {rb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
