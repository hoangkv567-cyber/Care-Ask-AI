#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CHUẨN HÓA TÊN HoiChung BẨN (staging) -> hội chứng THẬT (Neo4j)
================================================================================
5 bệnh có MỌI hội chứng dán tên staging (vd Cam tích × 'Cam khí (thể nhẹ) - Tỳ vị
bất hòa'). Scorer _score_syndromes_grounded LOẠI node có '('/số/'thể ' -> các bệnh
này KHÔNG BAO GIỜ chấm được hội chứng của mình -> core luôn ngoại lai -> Mục 5 sai.
Script đổi tên node HoiChung sang hội chứng thật (merge-or-create + repoint bài +
dời triệu chứng + xóa node bẩn). Đồng bộ với bản CSV đã chuẩn hóa.

An toàn: MẶC ĐỊNH DRY-RUN. --apply ghi + rollback JSON. ⚠️ BACKUP Aura trước --apply.
    python scripts/normalize_dirty_hoichung.py            # dry-run
    python scripts/normalize_dirty_hoichung.py --apply     # ghi (+ rollback)
    python scripts/normalize_dirty_hoichung.py --rollback data/normalize_hoichung_rollback.json
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

ROLLBACK_PATH = os.path.join(_ROOT, "data", "normalize_hoichung_rollback.json")

# (bệnh, tên BẨN, tên SẠCH) — khớp MAP đã áp cho CSV.
TARGETS = [
    ("Cam tích", "Cam khí (thể nhẹ) - Tỳ vị bất hòa", "Tỳ vị bất hòa"),
    ("Cam tích", "Cam tích (thể vừa) - Tỳ hư tích trệ", "Tỳ hư tích trệ"),
    ("Cam tích", "Can cam / Cam khô (thể nặng) - Khí huyết lưỡng hư, Tỳ vị hư suy", "Khí huyết lưỡng hư kèm Tỳ vị hư suy"),
    ("Lâm chứng", "Nhiệt lâm (Bàng quang thấp nhiệt)", "Nhiệt lâm"),
    ("Lâm chứng", "Thạch lâm (Sỏi tiết niệu - thấp nhiệt kết tụ, sa thạch nội trở)", "Thạch lâm"),
    ("Lâm chứng", "Khí lâm (thực chứng - Khí trệ; hư chứng - Trung khí hạ hãm)", "Khí lâm"),
    ("Lâm chứng", "Huyết lâm (thực chứng - Hạ tiêu thấp nhiệt, nhiệt thịnh bức huyết; hư chứng - Âm hư hỏa vượng)", "Huyết lâm"),
    ("Lâm chứng", "Cao lâm (thực chứng - Thấp nhiệt hạ chú, thanh trọc bất phân; hư chứng - Thận hư hạ nguyên bất cố)", "Cao lâm"),
    ("Lâm chứng", "Lao lâm (Tỳ Thận lưỡng hư, lao quyện tức phát)", "Lao lâm"),
    ("Ma chẩn (sởi)", "Sơ nhiệt kỳ (thời kỳ tiền phát/khởi phát - tà uất phế vệ)", "Tà uất phế vệ"),
    ("Ma chẩn (sởi)", "Kiến hình kỳ (thời kỳ toàn phát/ra ban - phế vị nhiệt thịnh, độc tà ngoại thấu)", "Phế vị nhiệt thịnh"),
    ("Ma chẩn (sởi)", "Thu một kỳ (thời kỳ ban lặn/hồi phục - âm tân hao tổn, dư nhiệt vị thanh)", "Âm tân hao tổn"),
    ("Ma chẩn (sởi)", "Ma độc bế phế (thể nghịch chứng - độc tà nội hãm, bế trở phế khí)", "Ma độc bế phế"),
    ("Ma chẩn (sởi)", "Nhiệt độc công hầu (thể nghịch chứng - độc tà công xung yết hầu)", "Nhiệt độc công hầu"),
    ("Sán khí", "Hàn sán (hàn ngưng can mạch)", "Hàn ngưng can mạch"),
    ("Sán khí", "Khí sán (can uất khí trệ)", "Can uất khí trệ"),
    ("Sán khí", "Hồ sán (thoát vị bẹn)", "Hồ sán"),
    ("Thủy đậu", "Phong nhiệt (thể nhẹ)", "Phong nhiệt"),
    ("Thủy đậu", "Thấp nhiệt độc thịnh (thể nặng) / Khí dinh (huyết) lưỡng phần nhiệt độc", "Thấp nhiệt độc thịnh"),
]


def inspect(driver, benh, old, new):
    q = """
    OPTIONAL MATCH (dh:HoiChung {name:$old})
    OPTIONAL MATCH (b:BenhLy {name:$benh})-[:CHIA_THÀNH]->(dh)
    OPTIONAL MATCH (dh)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
    OPTIONAL MATCH (dh)-[rt:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
    OPTIONAL MATCH (od:BenhLy)-[:CHIA_THÀNH]->(dh)
    RETURN dh IS NOT NULL AS exists, b IS NOT NULL AS in_benh,
           collect(DISTINCT p.name) AS bai,
           collect(DISTINCT {t:t.name, benh_ly:rt.benh_ly}) AS tc,
           collect(DISTINCT od.name) AS all_benh
    """
    r = run_read(driver, q, old=old, benh=benh)
    return r[0] if r else None


def normalize(driver, benh, old, new):
    run_write(driver, """
    MATCH (b:BenhLy {name:$benh}) MERGE (ch:HoiChung {name:$new}) MERGE (b)-[:CHIA_THÀNH]->(ch)
    """, benh=benh, new=new)
    run_write(driver, """
    MATCH (dh:HoiChung {name:$old})-[r:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
    MATCH (ch:HoiChung {name:$new})
    SET p.hoi_chung=$new MERGE (ch)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p) DELETE r
    """, old=old, new=new)
    run_write(driver, """
    MATCH (dh:HoiChung {name:$old})-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
    MATCH (ch:HoiChung {name:$new})
    MERGE (ch)-[r2:CÓ_BIỂU_HIỆN]->(t) ON CREATE SET r2.benh_ly=r.benh_ly DELETE r
    """, old=old, new=new)
    # node bẩn đơn-bệnh -> xóa hẳn
    run_write(driver, "MATCH (dh:HoiChung {name:$old}) DETACH DELETE dh", old=old)


def do_rollback(driver, path):
    for it in json.load(open(path, encoding="utf-8")):
        benh, old, new = it["benh"], it["old"], it["new"]
        run_write(driver, "MERGE (dh:HoiChung {name:$old})", old=old)
        run_write(driver, "MATCH (b:BenhLy {name:$benh}),(dh:HoiChung {name:$old}) MERGE (b)-[:CHIA_THÀNH]->(dh)", benh=benh, old=old)
        for tc in it.get("tc", []):
            run_write(driver, """MATCH (dh:HoiChung {name:$old}) MATCH (t:TrieuChung {name:$t})
            MERGE (dh)-[r:CÓ_BIỂU_HIỆN]->(t) ON CREATE SET r.benh_ly=$bl""", old=old, t=tc["t"], bl=tc.get("benh_ly"))
        for bai in it.get("bai", []):
            run_write(driver, """MATCH (dh:HoiChung {name:$old}) MATCH (p:BaiThuoc {name:$bai})
            SET p.hoi_chung=$old MERGE (dh)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p)""", old=old, bai=bai)
        print(f"  ↩ tái tạo {old!r}")
    print("XONG rollback.")


def main():
    ap = argparse.ArgumentParser(description="Chuẩn hóa HoiChung bẩn. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rollback", metavar="FILE")
    args = ap.parse_args()
    driver = get_driver()
    try:
        if args.rollback:
            do_rollback(driver, args.rollback)
            return
        plan, rollback = [], []
        for benh, old, new in TARGETS:
            st = inspect(driver, benh, old, new)
            if not st or not st["exists"]:
                print(f"[BỎ QUA] {benh}: {old!r} (node không tồn tại)")
                continue
            others = [x for x in st["all_benh"] if x and x != benh]
            if others:
                print(f"⚠️ [BỎ QUA] {old!r} dùng chung bệnh {others} — không xóa an toàn.")
                continue
            tcs = [t for t in st["tc"] if t.get("t")]
            print(f"[{benh}] {old!r} -> {new!r}  (bài {st['bai']}, {len(tcs)} triệu chứng)")
            plan.append((benh, old, new))
            rollback.append({"benh": benh, "old": old, "new": new, "bai": st["bai"], "tc": tcs})
        if not plan:
            print("\nKhông có gì để chuẩn hóa.")
            return
        if args.apply:
            json.dump(rollback, open(ROLLBACK_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"\nĐã ghi rollback: {ROLLBACK_PATH}\n⚠️ Đảm bảo đã BACKUP Aura. Ghi...")
            for benh, old, new in plan:
                normalize(driver, benh, old, new)
                print(f"  ✓ {old!r} -> {new!r}")
            print(f"XONG: chuẩn hóa {len(plan)} node.")
        else:
            print(f"\n(DRY-RUN — {len(plan)} node sẽ đổi. Thêm --apply sau khi backup Aura.)")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
