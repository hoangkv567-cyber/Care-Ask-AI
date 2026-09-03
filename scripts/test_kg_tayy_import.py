#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_kg_tayy_import.py — An toàn import bệnh Tây y lên Neo4j (kg_import_tayy.py).

BẤT BIẾN SỐNG CÒN cần khóa bằng test:
  1) VOCAB KHÔNG NHIỄM: import không được tạo/đổi bất kỳ node TrieuChung nào —
     src/qa_system.py:535 quét TẤT CẢ TrieuChung làm vocab trích triệu chứng từ lời
     khai; một node triệu chứng Tây y dịch máy lọt vào là đầu độc tầng trích.
  2) Cột cấm (tỉ_lệ_chữa_khỏi, đề_xuất_thuốc...) không xuất hiện trong script import
     (nguồn đọc là index đã lọc, không phải TayY_clean.csv).
  3) Cầu substring phải cap fanout (mảnh dịch máy 'hoặc' tạo hub 641 cạnh nếu thả trần).
  4) Chatbot QA không sinh được Cypher GHI trên label mới.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_kg_tayy_import.py   (exit != 0 nếu fail)
Phần online (đếm graph) tự SKIP nếu thiếu env Neo4j.
"""
import io
import os
import re
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for p in (_SCRIPTS, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from kg_import_tayy import (  # noqa: E402
    NGUON,
    build_disease_links,
    build_related_links,
    build_symptom_bridge,
    item_chk,
    norm_key,
    norm_text,
    parse_paren_name,
)


def chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return not cond


def main():
    fails = 0

    # ---- 1. Chuẩn hóa ----
    fails += chk("norm_text collapse whitespace/NFKC",
                 norm_text("  Viêm   phổi  ") == "Viêm phổi")
    fails += chk("norm_key idempotent", norm_key(norm_key("Đau Đầu")) == norm_key("Đau Đầu"))
    fails += chk("parse_paren_name lấy ngoặc đuôi",
                 parse_paren_name("Tâm quý (suy tim)") == "suy tim")
    fails += chk("parse_paren_name tên không ngoặc -> None",
                 parse_paren_name("Cảm mạo") is None)
    fails += chk("parse_paren_name ngoặc giữa tên -> None",
                 parse_paren_name("Bệnh (cũ) thể mới") is None)

    # ---- 2. Cầu triệu chứng: exact giữ, hub bị cap, denylist chặn ----
    tcm = ["ho", "ho khan", "đau đầu", "sốt cao", "hoặc sốt", "hoặc ho", "hoặc đau",
           "hoặc mệt", "hoặc nôn", "hoặc khát", "đau bụng"]
    bridge = build_symptom_bridge({"đau đầu", "hoặc", "sốt"}, tcm, max_fanout=5)
    pairs = {(b["tayy"], b["tcm"], b["pp"]) for b in bridge}
    fails += chk("exact luôn giữ", ("đau đầu", "đau đầu", "exact") in pairs)
    fails += chk("denylist chặn 'hoặc' làm cầu substring",
                 not any(b["tayy"] == "hoặc" and b["pp"] == "substring" for b in bridge))
    # hub: symptom khớp quá max_fanout đích -> bỏ toàn bộ substring của nó
    many_tcm = [f"đau khớp vai {i}" for i in range(9)]
    bridge2 = build_symptom_bridge({"đau khớp"}, many_tcm, max_fanout=5)
    fails += chk("fanout > cap -> bỏ substring của symptom đó", bridge2 == [])
    bridge3 = build_symptom_bridge({"đau khớp"}, many_tcm[:4], max_fanout=5)
    fails += chk("fanout <= cap -> giữ", len(bridge3) == 4)
    fails += chk("substring đòi >=4 ký tự",
                 build_symptom_bridge({"ho"}, ["ho khan"], max_fanout=5) == [])

    # ---- 3. Cầu bệnh: exact ưu tiên, contains word-boundary, trộn nguồn ----
    items = [
        {"id": "TAYY-00001", "ten": "Suy tim", "_tc_keys": []},
        {"id": "TAYY-00002", "ten": "Suy tim mạn tính", "_tc_keys": []},
        {"id": "TAYY-00003", "ten": "Bệnh trứng cá", "_tc_keys": []},
        {"id": "TAYY-00004", "ten": "Trứng cá đỏ", "_tc_keys": []},
    ]
    mapping = {"cao": {"Suy tim": "Tâm quý (suy tim)"}, "can_xac_nhan": {}}
    links, misses = build_disease_links(
        ["Tâm quý (suy tim)", "Phấn thích (trứng cá)"], items, mapping, max_targets=3
    )
    by = {(l["benh_ly"], l["id"]): l for l in links}
    tam_quy = by.get(("Tâm quý (suy tim)", "TAYY-00001"))
    fails += chk("exact match -> xac_nhan + trộn 2 nguồn",
                 tam_quy is not None and tam_quy["trang_thai"] == "xac_nhan"
                 and "ten_ngoac" in tam_quy["nguon"] and "mapping_json" in tam_quy["nguon"],
                 str(tam_quy))
    fails += chk("contains word-boundary -> can_xac_nhan (trứng cá)",
                 by.get(("Phấn thích (trứng cá)", "TAYY-00003"), {}).get("trang_thai")
                 == "can_xac_nhan")
    # word-boundary: 'sởi' không được khớp 'khởi phát'
    _, misses2 = build_disease_links(
        ["Ma chẩn (sởi)"], [{"id": "TAYY-00009", "ten": "Khởi phát động kinh", "_tc_keys": []}],
        {}, max_targets=3,
    )
    fails += chk("'sởi' KHÔNG khớp 'khởi phát' (word-boundary)", len(misses2) == 1)

    # ---- 4. Link máy đề xuất: ngưỡng >=2, top-k, loại cặp đã xác nhận ----
    rel_items = [
        {"id": "TAYY-00010", "ten": "A", "_tc_keys": ["ho khan", "sốt cao", "đau đầu"]},
        {"id": "TAYY-00011", "ten": "B", "_tc_keys": ["ho khan"]},
    ]
    rel_bridge = [
        {"tayy": "ho khan", "tcm": "ho khan", "pp": "exact"},
        {"tayy": "sốt cao", "tcm": "sốt cao", "pp": "exact"},
        {"tayy": "đau đầu", "tcm": "đau đầu", "pp": "exact"},
    ]
    related = build_related_links(
        rel_items, rel_bridge,
        {"Cảm mạo": {"ho khan", "sốt cao", "đau đầu"}},
        confirmed=set(), top_k=3, min_shared=2,
    )
    fails += chk("máy đề xuất: A đạt (3 chung), B loại (<2 chung)",
                 [r["id"] for r in related] == ["TAYY-00010"], str(related))
    related2 = build_related_links(
        rel_items, rel_bridge, {"Cảm mạo": {"ho khan", "sốt cao", "đau đầu"}},
        confirmed={("Cảm mạo", "TAYY-00010")}, top_k=3, min_shared=2,
    )
    fails += chk("cặp đã xác nhận không bị lặp ở link máy", related2 == [])

    # ---- 5. item_chk nhạy với thay đổi nội dung ----
    a = {"ten": "X", "icd": "J45", "tt": "chua_kiem_duyet", "khoa": "", "mo_ta": "",
         "co": [], "tc": ["ho"]}
    b = dict(a, icd="J46")
    fails += chk("item_chk đổi khi icd đổi", item_chk(a) != item_chk(b))
    fails += chk("item_chk ổn định", item_chk(a) == item_chk(dict(a)))

    # ---- 6. Cột cấm không có trong script import ----
    src = io.open(os.path.join(_SCRIPTS, "kg_import_tayy.py"), encoding="utf-8").read()
    for banned in ("tỉ_lệ_chữa_khỏi", "đề_xuất_thuốc", "thuốc_phổ_biến", "thông_tin_thuốc",
                   "đề_xuất_món_ăn", "cách_phòng_tránh"):
        fails += chk(f"script import KHÔNG nhắc cột cấm {banned!r}", banned not in src)
    fails += chk("script không parse nội dung TayY_clean.csv (chỉ sha256)",
                 "DictReader" not in src and "import csv" not in src)
    # TƯƠNG_ĐƯƠNG chỉ MATCH phía TrieuChung — không MERGE/CREATE node TCM
    fails += chk("Cypher KHÔNG MERGE/CREATE node TrieuChung",
                 not re.search(r"(?:MERGE|CREATE)\s*\((?:\w+)?:TrieuChung\s*[{)]", src))

    # ---- 7. Chatbot QA từ chối Cypher ghi trên label mới ----
    from src.qa_system import TCMQA
    for evil in (
        "MERGE (b:BenhTayY {disease_id:'x'}) RETURN b",
        "CREATE (:TrieuChungTayY {name:'x'})",
        "MATCH (b:BenhTayY) SET b.name = 'x' RETURN b",
        "MATCH (b:BenhTayY) DETACH DELETE b",
    ):
        fails += chk(f"QA chặn Cypher ghi: {evil[:42]!r}",
                     not TCMQA._is_read_only_cypher(evil))
    fails += chk("QA cho phép Cypher đọc BenhTayY",
                 TCMQA._is_read_only_cypher(
                     "MATCH (b:BenhTayY) WHERE b.icd10_code STARTS WITH 'J45' RETURN b.name LIMIT 5"))

    # ---- 8. ONLINE (skip nếu thiếu env): bất biến vocab + đếm khớp ----
    load_env_lines = os.path.join(_ROOT, ".env")
    if os.path.exists(load_env_lines):
        for line in io.open(load_env_lines, encoding="utf-8"):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    if os.environ.get("NEO4J_URI"):
        from kg_maintenance import get_driver, run_read
        driver = get_driver()
        try:
            n = run_read(driver,
                         "MATCH (t:TrieuChung) WHERE t._nguon = $src RETURN count(t) AS c",
                         src=NGUON)[0]["c"]
            fails += chk("ONLINE: 0 node TrieuChung mang _nguon import Tây y", n == 0, str(n))
            n = run_read(driver,
                         "MATCH (n:TrieuChungTayY) WHERE 'TrieuChung' IN labels(n) "
                         "RETURN count(n) AS c")[0]["c"]
            fails += chk("ONLINE: 0 node mang cả 2 label triệu chứng", n == 0, str(n))
            markers = run_read(driver, """
                MATCH (t:TrieuChung)
                WHERE toLower(t.name) IN ['crack kêu khi hít vào', 'yan peng hui', 'úc trác']
                RETURN count(t) AS c""")[0]["c"]
            fails += chk("ONLINE: marker dịch máy không có trong vocab TrieuChung",
                         markers == 0, str(markers))
            counts = run_read(driver, """
                MATCH (b:BenhTayY)
                RETURN count(b) AS n,
                       sum(CASE WHEN b.disease_id =~ 'TAYY-\\d{5}' THEN 1 ELSE 0 END) AS ok
            """)[0]
            if counts["n"]:
                fails += chk("ONLINE: mọi BenhTayY có disease_id đúng dạng",
                             counts["n"] == counts["ok"], str(counts))
                bad_bridge = run_read(driver, """
                    MATCH (:TrieuChungTayY)-[r:TƯƠNG_ĐƯƠNG]->(:TrieuChung)
                    WHERE r.phuong_phap IS NULL RETURN count(r) AS c""")[0]["c"]
                fails += chk("ONLINE: mọi cạnh TƯƠNG_ĐƯƠNG có phuong_phap", bad_bridge == 0)
            else:
                print("[SKIP] Chưa import BenhTayY — phần đếm khớp chạy sau --apply")
        finally:
            driver.close()
    else:
        print("[SKIP] Thiếu env Neo4j — bỏ phần online")

    print("\n" + ("✅ TẤT CẢ PASS" if fails == 0 else f"❌ {fails} KIỂM TRA FAIL"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
