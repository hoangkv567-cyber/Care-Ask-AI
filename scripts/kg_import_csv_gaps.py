#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
ĐỒNG BỘ KHOẢNG TRỐNG CSV -> KNOWLEDGE GRAPH (Neo4j)
================================================================================
So từng dòng data/Medicine_clean.csv với graph; dòng nào graph THIẾU
(cạnh BenhLy-CHIA_THÀNH->HoiChung, node BaiThuoc, cạnh CÓ_BIỂU_HIỆN, BAO_GỒM)
thì bù đắp — CHỈ THÊM, không sửa/xóa gì sẵn có.

An toàn:
  - MẶC ĐỊNH DRY-RUN: chỉ liệt kê kế hoạch. Thêm --apply mới ghi.
  - Khớp node theo tên KHÔNG phân biệt hoa/thường -> tái dùng node sẵn có,
    KHÔNG tạo biến thể trùng tên mới (KG đang có ~24 nhóm HoiChung lệch hoa/thường).
  - Mọi node/cạnh tạo mới gắn nhãn nguồn (_nguon / r.nguon = NGUON) -> --undo gỡ được:
        python scripts/kg_import_csv_gaps.py --undo

Chạy từ thư mục gốc dự án:
    python scripts/kg_import_csv_gaps.py            # dry-run
    python scripts/kg_import_csv_gaps.py --apply    # ghi thật
    python scripts/kg_import_csv_gaps.py --undo     # gỡ những gì script đã tạo
================================================================================
"""
import argparse
import csv
import os
import re
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kg_maintenance import get_driver, run_read, run_write  # noqa: E402

NGUON = "csv_gap_sync"
CSV_PATH = os.path.join(_ROOT, "data", "Medicine_clean.csv")


def norm(s):
    return " ".join((s or "").split()).strip()


def split_vi_thuoc(raw: str) -> list:
    """Tách vị thuốc theo dấu phẩy NGOÀI ngoặc; bỏ token ghi chú (bắt đầu bằng '(' hoặc chứa 'gia:')."""
    out, buf, depth = [], [], 0
    for ch in raw:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    cleaned = []
    for tok in out:
        tok = tok.strip().strip(".").strip()
        if not tok or tok.startswith("("):
            continue
        # cắt MỌI ghi chú trong ngoặc dính ở đuôi token, dù CÓ hay KHÔNG có dấu chấm phía trước
        # (vd 'Kê huyết đằng (Bệnh chi trên gia...)' — ngoặc liền sau tên, không có '.').
        tok = re.sub(r"\s*\(.*$", "", tok).strip().strip(".").strip()
        # bỏ token ghi chú gia giảm nằm NGOÀI ngoặc (vd 'gia: Trần bì' — đầu ngữ, không phải vị thuốc)
        if not tok or re.match(r"(?i)^(?:nếu|gia|thêm|bỏ|hoặc|tùy)\b", tok):
            continue
        if tok:
            cleaned.append(tok)
    return cleaned


def canonical_node_name(driver, label: str, name: str):
    """Tìm node theo tên không phân biệt hoa/thường. Ưu tiên khớp EXACT; nếu chỉ có biến thể
    khác hoa/thường, chọn biến thể có BẬC (số quan hệ) cao nhất. Trả None nếu chưa có."""
    rows = run_read(
        driver,
        f"""MATCH (n:{label}) WHERE toLower(n.name) = toLower($name)
            RETURN n.name AS name, COUNT {{ (n)--() }} AS deg ORDER BY deg DESC""",
        name=name,
    )
    if not rows:
        return None
    for r in rows:
        if r["name"] == name:
            return name
    return rows[0]["name"]


def find_gaps(driver, rows):
    """Trả về danh sách kế hoạch bù đắp cho từng dòng CSV thiếu dữ liệu graph."""
    plans = []
    for row in rows:
        b, hc, bt = norm(row["tên_bệnh"]), norm(row["hội_chứng"]), norm(row["bài_thuốc"])
        if not (b and hc and bt):
            continue
        # Gộp trên MỌI biến thể hoa/thường của hội chứng: chỉ coi là thiếu khi KHÔNG biến thể
        # nào có (tránh tạo cạnh thừa lên biến thể chưa nối trong khi biến thể khác đã nối).
        res = run_read(
            driver,
            """
            OPTIONAL MATCH (b:BenhLy) WHERE toLower(b.name) = toLower($b)
            WITH b LIMIT 1
            OPTIONAL MATCH (h:HoiChung) WHERE toLower(h.name) = toLower($hc)
            OPTIONAL MATCH (b)-[ct:CHIA_THÀNH]->(h)
            OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
              WHERE toLower(p.name) = toLower($bt) AND toLower(p.benh_ly) = toLower($b)
            RETURN b.name AS benh,
                   count(DISTINCT ct) > 0 AS linked, count(DISTINCT p) > 0 AS has_bt
            """,
            b=b.lower(), hc=hc.lower(), bt=bt.lower(),
        )
        r = res[0] if res else {}
        if not r.get("benh"):
            plans.append({"row": row, "action": "SKIP", "reason": f"BenhLy '{b}' chưa có trong graph (ngoài phạm vi script này)"})
            continue
        need_link = not r.get("linked")
        need_bt = not r.get("has_bt")
        if need_link or need_bt:
            plans.append({
                "row": row,
                "action": "FIX",
                "benh": r["benh"],
                "need_link": need_link,
                "need_bt": need_bt,
            })
    return plans


def apply_plan(driver, plan):
    row = plan["row"]
    b_name = plan["benh"]
    hc_csv = norm(row["hội_chứng"])
    bt_name = norm(row["bài_thuốc"])

    # 1. HoiChung: ưu tiên biến thể ĐÃ NỐI với bệnh này (để BaiThuoc treo đúng đường truy hồi);
    #    rồi tới node sẵn có (exact trước, bậc cao nhất sau); chưa có thì tạo + tag
    linked = run_read(driver, """
        MATCH (b:BenhLy {name:$b})-[:CHIA_THÀNH]->(h:HoiChung)
        WHERE toLower(h.name) = toLower($hc)
        RETURN h.name AS name LIMIT 1
        """, b=b_name, hc=hc_csv)
    h_name = (linked[0]["name"] if linked else None) or canonical_node_name(driver, "HoiChung", hc_csv)
    if not h_name:
        run_write(driver, "MERGE (h:HoiChung {name:$n}) ON CREATE SET h._nguon = $src",
                  n=hc_csv, src=NGUON)
        h_name = hc_csv
        print(f"      + tạo HoiChung '{hc_csv}'")

    # 2. Cạnh BenhLy-CHIA_THÀNH->HoiChung
    if plan["need_link"]:
        run_write(driver, """
            MATCH (b:BenhLy {name:$b}), (h:HoiChung {name:$h})
            MERGE (b)-[r:CHIA_THÀNH]->(h) ON CREATE SET r.nguon = $src
            """, b=b_name, h=h_name, src=NGUON)
        print(f"      + cạnh ({b_name})-[:CHIA_THÀNH]->({h_name})")

    # 3. Node BaiThuoc (đúng ràng buộc truy hồi: p.benh_ly=b.name, p.hoi_chung=h.name) + cạnh
    if plan["need_bt"]:
        run_write(driver, """
            MATCH (h:HoiChung {name:$h})
            MERGE (p:BaiThuoc {name:$bt, benh_ly:$b, hoi_chung:$h})
              ON CREATE SET p._nguon = $src
            MERGE (h)-[r:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p) ON CREATE SET r.nguon = $src
            """, h=h_name, bt=bt_name, b=b_name, src=NGUON)
        print(f"      + BaiThuoc '{bt_name}' (benh_ly='{b_name}', hoi_chung='{h_name}')")

        # 3b. Vị thuốc BAO_GỒM
        for vt in split_vi_thuoc(row.get("vị_thuốc", "")):
            v_name = canonical_node_name(driver, "ViThuoc", vt) or vt
            run_write(driver, """
                MATCH (p:BaiThuoc {name:$bt, benh_ly:$b, hoi_chung:$h})
                MERGE (v:ViThuoc {name:$v}) ON CREATE SET v._nguon = $src
                MERGE (p)-[r:BAO_GỒM]->(v) ON CREATE SET r.nguon = $src
                """, bt=bt_name, b=b_name, h=h_name, v=v_name, src=NGUON)
        print(f"        vị thuốc: {split_vi_thuoc(row.get('vị_thuốc', ''))}")

    # 4. Cạnh CÓ_BIỂU_HIỆN (benh_ly = tên bệnh) cho các triệu chứng của dòng — chỉ bù nếu thiếu
    for tc in [t.strip() for t in row.get("triệu_chứng", "").split(",") if t.strip()]:
        t_name = canonical_node_name(driver, "TrieuChung", tc) or tc
        run_write(driver, """
            MATCH (h:HoiChung {name:$h})
            MERGE (t:TrieuChung {name:$t}) ON CREATE SET t._nguon = $src
            MERGE (h)-[r:CÓ_BIỂU_HIỆN {benh_ly:$b}]->(t) ON CREATE SET r.nguon = $src
            """, h=h_name, t=t_name, b=b_name, src=NGUON)


def undo(driver):
    print("Gỡ mọi cạnh r.nguon =", NGUON)
    run_write(driver, "MATCH ()-[r]-() WHERE r.nguon = $src DELETE r", src=NGUON)
    print("Gỡ mọi node mồ côi _nguon =", NGUON)
    run_write(driver, "MATCH (n) WHERE n._nguon = $src AND NOT (n)--() DELETE n", src=NGUON)
    print("Xong. (Node _nguon còn quan hệ khác sẽ được giữ lại an toàn)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="ghi thật (mặc định dry-run)")
    ap.add_argument("--undo", action="store_true", help="gỡ những gì script đã tạo")
    args = ap.parse_args()

    driver = get_driver()
    try:
        if args.undo:
            undo(driver)
            return 0

        with open(CSV_PATH, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        print(f"Đối chiếu {len(rows)} dòng CSV với graph...")
        plans = find_gaps(driver, rows)
        fixes = [p for p in plans if p["action"] == "FIX"]
        skips = [p for p in plans if p["action"] == "SKIP"]

        print(f"\nDòng cần bù đắp: {len(fixes)} | bỏ qua: {len(skips)}")
        for p in skips:
            print(f"  [SKIP] {p['reason']}")
        for p in fixes:
            row = p["row"]
            need = []
            if p["need_link"]:
                need.append("CHIA_THÀNH")
            if p["need_bt"]:
                need.append("BaiThuoc+BAO_GỒM")
            print(f"  [FIX] {norm(row['tên_bệnh'])} | {norm(row['hội_chứng'])} | {norm(row['bài_thuốc'])[:45]}")
            print(f"        thiếu: {', '.join(need)}")
            if not args.apply:
                print(f"        vị thuốc sẽ tạo: {split_vi_thuoc(row.get('vị_thuốc', ''))}")

        if not fixes:
            print("\nGraph đã đủ — không có gì để làm.")
            return 0
        if not args.apply:
            print("\n(DRY-RUN — thêm --apply để ghi; mọi thứ tạo mới gắn nhãn nguon để --undo)")
            return 0

        print("\n>>> APPLY:")
        for p in fixes:
            row = p["row"]
            print(f"  -> {norm(row['tên_bệnh'])} | {norm(row['hội_chứng'])}")
            apply_plan(driver, p)
        print("\nXong. Chạy lại không --apply để xác nhận hết gap; hoặc --undo để gỡ.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
