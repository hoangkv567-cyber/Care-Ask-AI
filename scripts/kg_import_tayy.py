#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
IMPORT BỆNH TÂY Y LÊN NEO4J + CẦU NỐI ĐÔNG–TÂY
================================================================================
Nguồn đọc: data/tayy_reference_index.json (ĐÃ lọc cột cấm + áp phân xử trùng tên
— KHÔNG đọc TayY_clean.csv trực tiếp, chỉ sha256 để kiểm sync). Chỉ import item
hiển thị (bỏ ``an: true`` — bản thua phân xử trùng, tránh QA trả dòng đôi).

Schema tạo mới (label RIÊNG — bất biến sống còn: TUYỆT ĐỐI không đụng label
TrieuChung/BenhLy vì vocab trích triệu chứng quét tất TrieuChung
(src/qa_system.py:535) và cổng an toàn disease_gates khớp chuỗi con tên BenhLy):

    (:BenhTayY {disease_id KEY, name, name_norm, icd10_code, trang_thai, khoa,
                mo_ta, ten_may, co_chat_luong, chk, _nguon})
    (:TrieuChungTayY {name KEY chuẩn hóa, _nguon})
    (BenhTayY)-[:CÓ_TRIỆU_CHỨNG {nguon}]->(TrieuChungTayY)
    (TrieuChungTayY)-[:TƯƠNG_ĐƯƠNG {phuong_phap, nguon}]->(TrieuChung)   # CHỈ MATCH phía TCM
    (BenhLy)-[:TƯƠNG_ỨNG_TÂY_Y {nguon, trang_thai}]->(BenhTayY)          # xác nhận (tên ngoặc + mapping json)
    (BenhLy)-[:LIÊN_QUAN_TRIỆU_CHỨNG {score, so_trieu_chung_chung, nguon}]->(BenhTayY)  # máy đề xuất top-3

Cầu triệu chứng: exact luôn giữ; substring 2 chiều >=4 ký tự NHƯNG cap fanout
(mặc định 5) — đo thật: mảnh dịch máy 'hoặc' tạo hub 641 cạnh nếu thả trần.
3 loại cạnh liên kết (TƯƠNG_ĐƯƠNG / TƯƠNG_ỨNG_TÂY_Y / LIÊN_QUAN_TRIỆU_CHỨNG)
được XÓA-TẠO-LẠI TOÀN BỘ mỗi lần apply/sync (dữ liệu dẫn xuất, ~4-5k cạnh, rẻ)
— không bao giờ stale. Node + cạnh CÓ_TRIỆU_CHỨNG diff theo ``chk`` per-item.

An toàn: MẶC ĐỊNH DRY-RUN (kế hoạch + forecast giới hạn Aura + worklist tên
miss). Mọi node/cạnh gắn _nguon/nguon = "tayy_import_v1" -> --undo gỡ sạch.

Chạy từ thư mục gốc dự án:
    python scripts/kg_import_tayy.py                    # dry-run
    python scripts/kg_import_tayy.py --apply            # ghi (UNWIND batch 1000)
    python scripts/kg_import_tayy.py --undo             # gỡ toàn bộ
    python scripts/kg_import_tayy.py --export-links     # -> data/dongtay_links.json
================================================================================
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

NGUON = "tayy_import_v1"
DEFAULT_INDEX = os.path.join(_ROOT, "data", "tayy_reference_index.json")
DEFAULT_CLEAN_CSV = os.path.join(_ROOT, "data", "TayY_clean.csv")
DEFAULT_MAPPING = os.path.join(_ROOT, "data", "disease_tcm_names.json")
DEFAULT_LINKS_OUT = os.path.join(_ROOT, "data", "dongtay_links.json")

AURA_NODE_LIMIT = 200_000
AURA_REL_LIMIT = 400_000

# Mảnh nối/định tính dịch máy — cấm làm cầu substring (hub rác: 'hoặc' 641 cạnh).
BRIDGE_DENYLIST = frozenset(("hoặc", "và", "nặng", "nhẹ", "khác", "không"))

LINK_TYPES = ("TƯƠNG_ĐƯƠNG", "TƯƠNG_ỨNG_TÂY_Y", "LIÊN_QUAN_TRIỆU_CHỨNG")


# ------------------------------------------------------------------ chuẩn hóa --
def norm_text(s: str) -> str:
    """NFKC + collapse whitespace + strip (giữ hoa/thường cho tên hiển thị)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "")).strip()


def norm_key(s: str) -> str:
    """Khóa chuẩn hóa (TrieuChungTayY.name, so khớp): norm_text + casefold."""
    return norm_text(s).casefold()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def item_chk(item: dict) -> str:
    """Dấu vân per-item để --sync diff: đổi bất kỳ trường import nào là chk đổi."""
    payload = json.dumps(
        [
            item.get("ten") or "", item.get("icd") or "", item.get("tt") or "",
            item.get("khoa") or "", item.get("mo_ta") or "",
            bool(item.get("ten_may")), sorted(item.get("co") or []),
            sorted(norm_key(t) for t in (item.get("tc") or []) if norm_key(t)),
        ],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_index(path: str, force: bool = False) -> list[dict]:
    """Nạp index, gate schema + sha nguồn; trả về CHỈ item hiển thị (bỏ an=true)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != "tayy-reference-index-v1":
        raise SystemExit(f"[LỖI] schema_version lạ: {data.get('schema_version')!r}")
    expected = str(data.get("source_sha256") or "")
    if expected and os.path.exists(DEFAULT_CLEAN_CSV) and not force:
        actual = sha256_file(DEFAULT_CLEAN_CSV)
        if actual != expected:
            raise SystemExit(
                "[LỖI] Index lệch TayY_clean.csv (sha256 khác). Chạy:\n"
                "  python scripts/build_tayy_reference_index.py --write\n"
                "hoặc thêm --force nếu chủ đích dùng index cũ."
            )
    items = []
    for it in data.get("items", []):
        if it.get("an"):
            continue                      # bản thua phân xử trùng — không lên graph
        it = dict(it)
        it["ten"] = norm_text(it.get("ten") or "")
        it["_tc_keys"] = sorted({k for t in (it.get("tc") or []) if (k := norm_key(t))})
        it["chk"] = item_chk(it)
        items.append(it)
    return items


# --------------------------------------------------------------- cầu triệu chứng --
def build_symptom_bridge(
    tayy_syms: set[str], tcm_names_lower: list[str], max_fanout: int
) -> list[dict]:
    """[{tayy, tcm, phuong_phap}] — tcm là tên TCM lowercase NGUYÊN VĂN từ graph
    (Cypher khớp lại bằng toLower(t.name)). Exact luôn giữ; substring 2 chiều
    >=4 ký tự, symptom nào quá max_fanout cạnh substring thì bỏ toàn bộ substring
    của nó (giữ exact) — chặn hub rác."""
    tcm_set = set(tcm_names_lower)
    out: list[dict] = []
    for sym in sorted(tayy_syms):
        if sym in tcm_set:
            out.append({"tayy": sym, "tcm": sym, "pp": "exact"})
        if sym in BRIDGE_DENYLIST or len(sym) < 4:
            continue
        subs = [
            t for t in tcm_names_lower
            if t != sym and len(t) >= 4 and (sym in t or t in sym)
        ]
        if 0 < len(subs) <= max_fanout:
            out.extend({"tayy": sym, "tcm": t, "pp": "substring"} for t in subs)
    return out


# ------------------------------------------------------------------ cầu bệnh --
def parse_paren_name(benh_ly: str) -> str | None:
    """'Tâm quý (suy tim)' -> 'suy tim'; tên không ngoặc đuôi -> None."""
    m = re.search(r"\(([^()]+)\)\s*$", norm_text(benh_ly))
    return norm_text(m.group(1)) if m else None


def _word_boundary_contains(needle_key: str, hay_key: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle_key)}(?!\w)", hay_key) is not None


def build_disease_links(
    benhly_names: list[str],
    items: list[dict],
    mapping: dict,
    max_targets: int,
) -> tuple[list[dict], list[str]]:
    """Cạnh TƯƠNG_ỨNG_TÂY_Y từ 2 nguồn: tên ngoặc BenhLy + disease_tcm_names.json.
    Khớp exact (norm_key) trước -> xac_nhan; contains word-boundary >=3 ký tự,
    ưu tiên tên đích NGẮN nhất, cap max_targets -> can_xac_nhan.
    Trả (links, worklist_miss)."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_key[norm_key(it["ten"])].append(it)

    # (benh_ly, western_name, nguon, trang_thai_goc)
    wanted: list[tuple[str, str, str, str]] = []
    for name in benhly_names:
        paren = parse_paren_name(name)
        if paren:
            wanted.append((name, paren, "ten_ngoac", "xac_nhan"))
    for group, trang_thai in (("cao", "xac_nhan"), ("can_xac_nhan", "can_xac_nhan")):
        for western, tcm_name in (mapping.get(group) or {}).items():
            if isinstance(tcm_name, str) and tcm_name.strip():
                wanted.append((norm_text(tcm_name), norm_text(western), "mapping_json", trang_thai))

    links: dict[tuple[str, str], dict] = {}
    misses: list[str] = []
    benhly_set = {norm_key(n): n for n in benhly_names}
    for benh_ly, western, nguon, trang_thai_goc in wanted:
        # mapping_json: BenhLy đích phải tồn tại trong graph (tên đã đổi theo quy ước)
        canonical_bl = benhly_set.get(norm_key(benh_ly))
        if canonical_bl is None:
            misses.append(f"(BenhLy vắng) {benh_ly!r} <- {western!r} [{nguon}]")
            continue
        wkey = norm_key(western)
        targets: list[tuple[dict, str]] = []
        if wkey in by_key:
            targets = [(it, trang_thai_goc) for it in by_key[wkey]]
        elif len(wkey) >= 3:
            hits = [
                it for it in items
                if _word_boundary_contains(wkey, norm_key(it["ten"]))
            ]
            hits.sort(key=lambda it: (len(it["ten"]), it["disease_id"] if "disease_id" in it else it["id"]))
            targets = [(it, "can_xac_nhan") for it in hits[:max_targets]]
        if not targets:
            misses.append(f"(TayY vắng) {western!r} <- {canonical_bl!r} [{nguon}]")
            continue
        for it, trang_thai in targets:
            key = (canonical_bl, it["id"])
            prev = links.get(key)
            if prev:
                # trộn nguồn: trạng thái lấy mức cao hơn (xac_nhan > can_xac_nhan)
                if prev["trang_thai"] != "xac_nhan" and trang_thai == "xac_nhan":
                    prev["trang_thai"] = "xac_nhan"
                if nguon not in prev["nguon"]:
                    prev["nguon"] = f"{prev['nguon']}+{nguon}"
            else:
                links[key] = {
                    "benh_ly": canonical_bl, "id": it["id"],
                    "nguon": nguon, "trang_thai": trang_thai,
                }
    return list(links.values()), misses


def build_related_links(
    items: list[dict],
    bridge: list[dict],
    benhly_symptoms: dict[str, set[str]],
    confirmed: set[tuple[str, str]],
    top_k: int = 3,
    min_shared: int = 2,
) -> list[dict]:
    """Cạnh máy-đề-xuất LIÊN_QUAN_TRIỆU_CHỨNG: top-k bệnh Tây y / BenhLy theo số
    triệu chứng TCM (đã bắc cầu) dùng chung; ngưỡng >= min_shared; bỏ cặp đã có
    link xác nhận."""
    tcm_to_tayy_syms: dict[str, set[str]] = defaultdict(set)
    for row in bridge:
        tcm_to_tayy_syms[row["tcm"]].add(row["tayy"])

    tayy_sym_to_items: dict[str, list[str]] = defaultdict(list)
    for it in items:
        for k in it["_tc_keys"]:
            tayy_sym_to_items[k].append(it["id"])

    out: list[dict] = []
    for benh_ly, tcm_syms in benhly_symptoms.items():
        shared_by_item: Counter = Counter()
        for tcm in tcm_syms:
            matched_items: set[str] = set()
            for tayy_sym in tcm_to_tayy_syms.get(tcm, ()):
                matched_items.update(tayy_sym_to_items.get(tayy_sym, ()))
            for item_id in matched_items:
                shared_by_item[item_id] += 1        # đếm THEO triệu chứng TCM
        ranked = [
            (item_id, shared) for item_id, shared in shared_by_item.items()
            if shared >= min_shared and (benh_ly, item_id) not in confirmed
        ]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        denom = max(len(tcm_syms), 1)
        for item_id, shared in ranked[:top_k]:
            out.append({
                "benh_ly": benh_ly, "id": item_id,
                "so_trieu_chung_chung": shared,
                "score": round(shared / denom, 3),
            })
    return out


# ------------------------------------------------------------------ graph I/O --
def fetch_graph_state(driver) -> dict:
    from kg_maintenance import run_read
    state = {}
    state["tcm_symptom_names"] = [
        r["n"] for r in run_read(driver, "MATCH (t:TrieuChung) RETURN DISTINCT toLower(t.name) AS n")
    ]
    state["benhly_names"] = [
        r["n"] for r in run_read(driver, "MATCH (b:BenhLy) RETURN b.name AS n")
    ]
    rows = run_read(driver, """
        MATCH (b:BenhLy)-[:CHIA_THÀNH]->(:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
        WHERE toLower(coalesce(r.benh_ly, '')) = toLower(b.name)
        RETURN b.name AS benh, collect(DISTINCT toLower(t.name)) AS syms
    """)
    state["benhly_symptoms"] = {r["benh"]: set(r["syms"]) for r in rows}
    state["existing"] = {
        r["id"]: r["chk"]
        for r in run_read(driver, "MATCH (b:BenhTayY) RETURN b.disease_id AS id, b.chk AS chk")
    }
    counts = run_read(driver, "MATCH (n) RETURN count(n) AS n")[0]["n"]
    rels = run_read(driver, "MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
    state["total_nodes"], state["total_rels"] = counts, rels
    return state


def plan_diff(items: list[dict], existing: dict[str, str]) -> dict:
    ids = {it["id"] for it in items}
    new = [it for it in items if it["id"] not in existing]
    changed = [it for it in items if it["id"] in existing and existing[it["id"]] != it["chk"]]
    gone = sorted(set(existing) - ids)
    return {"new": new, "changed": changed, "gone": gone}


def _batches(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def apply_import(driver, items, bridge, disease_links, related_links, diff, batch_size):
    from kg_maintenance import run_write
    run_write(driver, "CREATE CONSTRAINT benh_tayy_id IF NOT EXISTS "
                      "FOR (b:BenhTayY) REQUIRE b.disease_id IS UNIQUE")
    run_write(driver, "CREATE CONSTRAINT trieuchung_tayy_name IF NOT EXISTS "
                      "FOR (t:TrieuChungTayY) REQUIRE t.name IS UNIQUE")

    upsert = diff["new"] + diff["changed"]
    node_rows = [
        {
            "id": it["id"], "ten": it["ten"], "ten_norm": norm_key(it["ten"]),
            "icd": it.get("icd") or "", "tt": it.get("tt") or "",
            "khoa": it.get("khoa") or "", "mo_ta": it.get("mo_ta") or "",
            "ten_may": bool(it.get("ten_may")),
            "co": ";".join(it.get("co") or []), "chk": it["chk"],
        }
        for it in upsert
    ]
    for batch in _batches(node_rows, batch_size):
        run_write(driver, """
            UNWIND $rows AS row
            MERGE (b:BenhTayY {disease_id: row.id})
              ON CREATE SET b._nguon = $src
            SET b.name = row.ten, b.name_norm = row.ten_norm, b.icd10_code = row.icd,
                b.trang_thai = row.tt, b.khoa = row.khoa, b.mo_ta = row.mo_ta,
                b.ten_may = row.ten_may, b.co_chat_luong = row.co, b.chk = row.chk
        """, rows=batch, src=NGUON)
    print(f"  BenhTayY upsert: {len(node_rows)}")

    sym_rows = [{"name": k} for k in sorted({k for it in upsert for k in it["_tc_keys"]})]
    for batch in _batches(sym_rows, batch_size):
        run_write(driver, """
            UNWIND $rows AS row
            MERGE (t:TrieuChungTayY {name: row.name}) ON CREATE SET t._nguon = $src
        """, rows=batch, src=NGUON)
    print(f"  TrieuChungTayY upsert: {len(sym_rows)}")

    edge_rows = [
        {"id": it["id"], "tc": k} for it in upsert for k in it["_tc_keys"]
    ]
    for batch in _batches(edge_rows, batch_size):
        run_write(driver, """
            UNWIND $rows AS row
            MATCH (b:BenhTayY {disease_id: row.id})
            MATCH (t:TrieuChungTayY {name: row.tc})
            MERGE (b)-[r:CÓ_TRIỆU_CHỨNG]->(t) ON CREATE SET r.nguon = $src
        """, rows=batch, src=NGUON)
    print(f"  CÓ_TRIỆU_CHỨNG upsert: {len(edge_rows)}")

    # dọn cạnh triệu chứng cũ của item ĐỔI chk + item biến mất + symptom mồ côi
    for it in diff["changed"]:
        run_write(driver, """
            MATCH (b:BenhTayY {disease_id: $id})-[r:CÓ_TRIỆU_CHỨNG]->(t:TrieuChungTayY)
            WHERE NOT t.name IN $tcs DELETE r
        """, id=it["id"], tcs=it["_tc_keys"])
    if diff["gone"]:
        run_write(driver, """
            MATCH (b:BenhTayY) WHERE b.disease_id IN $ids DETACH DELETE b
        """, ids=diff["gone"])
        print(f"  BenhTayY gỡ (không còn trong index): {len(diff['gone'])}")

    # 3 loại cạnh liên kết: dữ liệu dẫn xuất -> xóa-tạo-lại toàn bộ, không bao giờ stale
    for link_type in LINK_TYPES:
        run_write(driver, f"MATCH ()-[r:{link_type}]-() WHERE r.nguon = $src DELETE r", src=NGUON)

    for batch in _batches(bridge, batch_size):
        run_write(driver, """
            UNWIND $rows AS row
            MATCH (s:TrieuChungTayY {name: row.tayy})
            MATCH (t:TrieuChung) WHERE toLower(t.name) = row.tcm
            MERGE (s)-[r:TƯƠNG_ĐƯƠNG]->(t)
              ON CREATE SET r.nguon = $src
            SET r.phuong_phap = row.pp
        """, rows=batch, src=NGUON)
    print(f"  TƯƠNG_ĐƯƠNG tạo lại: {len(bridge)} cặp")

    for batch in _batches(disease_links, batch_size):
        run_write(driver, """
            UNWIND $rows AS row
            MATCH (bl:BenhLy {name: row.benh_ly})
            MATCH (bt:BenhTayY {disease_id: row.id})
            MERGE (bl)-[r:TƯƠNG_ỨNG_TÂY_Y]->(bt)
              ON CREATE SET r.nguon = $src
            SET r.trang_thai = row.trang_thai, r.nguon_khop = row.nguon
        """, rows=batch, src=NGUON)
    print(f"  TƯƠNG_ỨNG_TÂY_Y tạo lại: {len(disease_links)} cạnh")

    for batch in _batches(related_links, batch_size):
        run_write(driver, """
            UNWIND $rows AS row
            MATCH (bl:BenhLy {name: row.benh_ly})
            MATCH (bt:BenhTayY {disease_id: row.id})
            MERGE (bl)-[r:LIÊN_QUAN_TRIỆU_CHỨNG]->(bt)
              ON CREATE SET r.nguon = $src
            SET r.score = row.score, r.so_trieu_chung_chung = row.so_trieu_chung_chung,
                r.nhan = 'may_de_xuat'
        """, rows=batch, src=NGUON)
    print(f"  LIÊN_QUAN_TRIỆU_CHỨNG tạo lại: {len(related_links)} cạnh")

    run_write(driver, """
        MATCH (t:TrieuChungTayY) WHERE t._nguon = $src AND NOT (t)--() DELETE t
    """, src=NGUON)


def undo(driver):
    from kg_maintenance import run_write
    print("Gỡ mọi cạnh nguon =", NGUON)
    run_write(driver, "MATCH ()-[r]-() WHERE r.nguon = $src DELETE r", src=NGUON)
    print("Gỡ mọi node BenhTayY/TrieuChungTayY _nguon =", NGUON)
    run_write(driver, "MATCH (b:BenhTayY) WHERE b._nguon = $src DETACH DELETE b", src=NGUON)
    run_write(driver, "MATCH (t:TrieuChungTayY) WHERE t._nguon = $src DETACH DELETE t", src=NGUON)
    print("Xong.")


def export_links(driver, out_path: str):
    from kg_maintenance import run_read
    links: dict[str, list[dict]] = defaultdict(list)
    for r in run_read(driver, """
        MATCH (bl:BenhLy)-[r:TƯƠNG_ỨNG_TÂY_Y]->(bt:BenhTayY)
        RETURN bt.disease_id AS id, bl.name AS benh_ly, r.nguon_khop AS nguon,
               r.trang_thai AS trang_thai
    """):
        links[r["id"]].append(
            {"benh_ly": r["benh_ly"], "nguon": r["nguon"], "trang_thai": r["trang_thai"]}
        )
    for r in run_read(driver, """
        MATCH (bl:BenhLy)-[r:LIÊN_QUAN_TRIỆU_CHỨNG]->(bt:BenhTayY)
        RETURN bt.disease_id AS id, bl.name AS benh_ly, r.score AS score
    """):
        links[r["id"]].append(
            {"benh_ly": r["benh_ly"], "nguon": "may_de_xuat",
             "trang_thai": "may_de_xuat", "score": r["score"]}
        )
    payload = {
        "schema_version": "dongtay-links-v1",
        "nguon": NGUON,
        "links": {k: sorted(v, key=lambda x: (x["trang_thai"] != "xac_nhan", x["benh_ly"]))
                  for k, v in sorted(links.items())},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"[OK] {out_path}: {len(links)} bệnh Tây y có liên kết Đông y")


# ----------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="Import bệnh Tây y lên Neo4j. Mặc định DRY-RUN.")
    ap.add_argument("--apply", action="store_true", help="ghi thật (mặc định dry-run)")
    ap.add_argument("--undo", action="store_true", help="gỡ toàn bộ theo _nguon")
    ap.add_argument("--export-links", action="store_true", help="xuất data/dongtay_links.json rồi thoát")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--out", default=DEFAULT_LINKS_OUT)
    ap.add_argument("--mapping", default=DEFAULT_MAPPING)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--bridge-max-fanout", type=int, default=5)
    ap.add_argument("--disease-link-max", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="bỏ qua kiểm sha256 index↔CSV")
    args = ap.parse_args()

    from kg_maintenance import get_driver
    driver = get_driver()
    try:
        if args.undo:
            undo(driver)
            return 0
        if args.export_links:
            export_links(driver, args.out)
            return 0

        items = load_index(args.index, force=args.force)
        mapping = {}
        if os.path.exists(args.mapping):
            with open(args.mapping, encoding="utf-8") as fh:
                mapping = json.load(fh)

        print(f"Item hiển thị trong index : {len(items)}")
        state = fetch_graph_state(driver)
        diff = plan_diff(items, state["existing"])
        tayy_syms = {k for it in items for k in it["_tc_keys"]}
        bridge = build_symptom_bridge(tayy_syms, state["tcm_symptom_names"], args.bridge_max_fanout)
        disease_links, misses = build_disease_links(
            state["benhly_names"], items, mapping, args.disease_link_max
        )
        confirmed = {(l["benh_ly"], l["id"]) for l in disease_links}
        related = build_related_links(items, bridge, state["benhly_symptoms"], confirmed)

        n_sym = len(tayy_syms)
        n_mention = sum(len(it["_tc_keys"]) for it in items)
        add_nodes = len(items) + n_sym - len(state["existing"])
        add_rels = n_mention + len(bridge) + len(disease_links) + len(related)
        print(f"Kế hoạch: node mới/đổi/gỡ  : {len(diff['new'])}/{len(diff['changed'])}/{len(diff['gone'])}")
        print(f"TrieuChungTayY distinct    : {n_sym}")
        print(f"CÓ_TRIỆU_CHỨNG mention     : {n_mention}")
        print(f"Cầu TƯƠNG_ĐƯƠNG            : {len(bridge)} "
              f"(exact {sum(1 for b in bridge if b['pp'] == 'exact')})")
        print(f"TƯƠNG_ỨNG_TÂY_Y xác nhận   : {len(disease_links)}")
        print(f"LIÊN_QUAN máy đề xuất      : {len(related)}")
        print(f"Forecast Aura              : ~{state['total_nodes'] + max(add_nodes, 0):,} node "
              f"/ {AURA_NODE_LIMIT:,} | ~{state['total_rels'] + add_rels:,} rel / {AURA_REL_LIMIT:,}")
        if misses:
            print(f"\nWORKLIST — {len(misses)} tên chưa khớp được (bổ sung tay sau):")
            for m in misses:
                print(f"  [MISS] {m}")

        if not (diff["new"] or diff["changed"] or diff["gone"]):
            print("\nNode/cạnh triệu chứng đã đồng bộ với index (link vẫn được tạo lại khi --apply).")
        if not args.apply:
            print("\n(DRY-RUN — thêm --apply để ghi; mọi thứ gắn nguon để --undo)")
            return 0

        print("\n>>> APPLY:")
        apply_import(driver, items, bridge, disease_links, related, diff, args.batch_size)
        print("\nXong. Chạy lại không --apply để xác nhận idempotent; --export-links để xuất link cho app.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
