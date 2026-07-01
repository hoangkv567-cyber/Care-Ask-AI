#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
CÔNG CỤ BẢO TRÌ KNOWLEDGE GRAPH TCM (Neo4j)
================================================================================
Kết nối qua .env (dùng src.config_loader — KHÔNG hardcode secret).
MẶC ĐỊNH DRY-RUN cho mọi thao tác ghi: chỉ khi thêm cờ --apply mới thực thi.

CÁCH DÙNG (chạy từ THƯ MỤC GỐC dự án):
    # 1) Chẩn đoán trạng thái (CHỈ ĐỌC, luôn chạy cái này TRƯỚC):
    python scripts/kg_maintenance.py diagnose

    # 2) Gộp node HoiChung trùng tên (dry-run xem trước -> rồi --apply):
    python scripts/kg_maintenance.py dedupe-hoichung
    python scripts/kg_maintenance.py dedupe-hoichung --apply

    # 3) Rà node HoiChung "bẩn" (rác NLP). Mặc định chỉ LIỆT KÊ + gắn cờ:
    python scripts/kg_maintenance.py clean-dirty              # dry-run: liệt kê
    python scripts/kg_maintenance.py clean-dirty --apply      # gắn cờ h._flagged_dirty=true (đảo được)
    python scripts/kg_maintenance.py clean-dirty --apply --delete   # XÓA hẳn (nguy hiểm)

    # 4) Thêm quan hệ phối ngũ (Thập bát phản / Thập cửu úy cổ điển):
    python scripts/kg_maintenance.py add-phoi-ngu             # dry-run: cặp nào sẽ tạo/bỏ qua
    python scripts/kg_maintenance.py add-phoi-ngu --apply

⚠️  BACKUP TRƯỚC KHI --apply:
    - Neo4j Aura: Console -> Snapshot / Backup.
    - Hoặc export: CALL apoc.export.cypher.all("backup.cypher", {}) (nếu có APOC + quyền ghi file).

LƯU Ý MODEL: HoiChung dùng chung theo `name`; ngữ cảnh bệnh nằm trên thuộc tính quan hệ
(r.benh_ly, p.benh_ly, p.hoi_chung). Vì vậy dedupe theo `name` là AN TOÀN với query hiện tại.
KHÔNG tách HoiChung theo (name, benh_ly) ở đây vì sẽ phá các query `HoiChung {name: $syn}` của app.
================================================================================
"""

import os
import re
import sys
import argparse

# Ép stdout/stderr sang UTF-8 để không crash UnicodeEncodeError trên console Windows (cp1258/cp1252)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cho phép import src.config_loader khi chạy từ gốc dự án
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neo4j import GraphDatabase          # noqa: E402
from src.config_loader import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Kết nối
# ---------------------------------------------------------------------------
def get_driver():
    cfg = load_config()
    neo = cfg.get("neo4j", {})
    uri = neo.get("uri") or os.getenv("NEO4J_URI")
    user = neo.get("user") or os.getenv("NEO4J_USER")
    pwd = neo.get("password") or os.getenv("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        print("LỖI: Thiếu NEO4J_URI/USER/PASSWORD. Hãy tạo .env (xem .env.example).")
        sys.exit(1)
    return GraphDatabase.driver(uri, auth=(user, pwd))


def run_read(driver, cypher, **params):
    with driver.session() as s:
        return [dict(r) for r in s.execute_read(lambda tx: list(tx.run(cypher, **params)))]


def run_write(driver, cypher, **params):
    with driver.session() as s:
        return s.execute_write(lambda tx: tx.run(cypher, **params).consume())


def has_apoc(driver):
    try:
        run_read(driver, "RETURN apoc.version() AS v")
        return True
    except Exception:
        return False


def _hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1) DIAGNOSE (chỉ đọc)
# ---------------------------------------------------------------------------
# Heuristic "bẩn" cho tên HoiChung (đã tinh chỉnh theo dữ liệu thật):
# - Node bẩn THẬT = tên lẫn tên bài thuốc/giai đoạn/vị thuốc trong ngoặc đơn, quá dài,
#   chứa chữ số (Bài 1/Nghiệm phương 2), hoặc chứa từ mô tả cơ chế (do/nếu/kèm/gây...).
# - KHÔNG dùng dấu phẩy làm dấu hiệu bẩn nữa vì hội chứng GHÉP hợp lệ cũng có phẩy
#   (vd "Can khí uất kết, tỳ thất kiện vận") -> tránh bắt nhầm.
_DIRTY_HOICHUNG_WHERE = r"""
    h.name IS NULL
    OR trim(h.name) = ''
    OR size(h.name) > 40
    OR h.name =~ '.*[0-9].*'
    OR h.name =~ '.*\(.*\).*'
    OR toLower(h.name) =~ '.*( hoặc | là | gây | do | nếu ).*'
"""
# Ghi chú: KHÔNG coi " kèm " / " và " / dấu phẩy là dấu hiệu bẩn — đó là từ nối của các
# hội chứng GHÉP hợp lệ (vd "Can Thận Âm Hư kèm Đờm Trệ", "Khí huyết đều hư, kinh mạch ứ trệ").


def diagnose(driver, args):
    _hr("TỔNG QUAN SỐ LƯỢNG NODE")
    for label in ["BenhLy", "HoiChung", "TrieuChung", "BaiThuoc", "ViThuoc"]:
        # Nhãn cố định (whitelist), không phải input người dùng -> nội suy an toàn
        rows = run_read(driver, f"MATCH (n:{label}) RETURN count(n) AS c")
        print(f"  {label:<11}: {rows[0]['c'] if rows else 0}")

    _hr("HoiChung TRÙNG TÊN (cùng name, >1 node) — ứng viên gộp")
    dups = run_read(driver, """
        MATCH (h:HoiChung)
        WITH h.name AS name, count(*) AS c
        WHERE c > 1
        RETURN name, c ORDER BY c DESC, name LIMIT 40
    """)
    total_dup_groups = run_read(driver, """
        MATCH (h:HoiChung) WITH h.name AS name, count(*) AS c WHERE c > 1
        RETURN count(*) AS groups, sum(c - 1) AS redundant_nodes
    """)
    if dups:
        for d in dups:
            print(f"  {d['c']:>3}x  {d['name']}")
        if total_dup_groups:
            g = total_dup_groups[0]
            print(f"  ... Tổng: {g['groups']} nhóm trùng, dư ~{g['redundant_nodes']} node thừa.")
    else:
        print("  (Không có HoiChung trùng tên — tốt)")

    _hr("HoiChung 'BẨN' (tên nghi rác NLP) — ứng viên làm sạch")
    dirty = run_read(driver, f"""
        MATCH (h:HoiChung) WHERE {_DIRTY_HOICHUNG_WHERE}
        RETURN h.name AS name LIMIT 40
    """)
    dirty_count = run_read(driver, f"""
        MATCH (h:HoiChung) WHERE {_DIRTY_HOICHUNG_WHERE}
        RETURN count(*) AS c
    """)
    dc = dirty_count[0]['c'] if dirty_count else 0
    print(f"  Số node nghi bẩn: {dc}")
    for d in dirty[:40]:
        print(f"    - {repr(d['name'])}")

    _hr("HoiChung MỒ CÔI (không có cạnh nào) — rác treo")
    orphans = run_read(driver, """
        MATCH (h:HoiChung) WHERE NOT (h)--() RETURN count(*) AS c
    """)
    print(f"  Số HoiChung không có bất kỳ quan hệ nào: {orphans[0]['c'] if orphans else 0}")

    _hr("CẠNH CÓ_BIỂU_HIỆN thiếu benh_ly (NULL) — nguồn khớp tràn")
    rel_null = run_read(driver, """
        MATCH (:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(:TrieuChung)
        RETURN
          count(r) AS tong,
          sum(CASE WHEN r.benh_ly IS NULL THEN 1 ELSE 0 END) AS null_benhly
    """)
    if rel_null:
        r = rel_null[0]
        pct = (100.0 * r['null_benhly'] / r['tong']) if r['tong'] else 0
        print(f"  CÓ_BIỂU_HIỆN: {r['tong']} cạnh, trong đó {r['null_benhly']} thiếu benh_ly ({pct:.1f}%)")

    _hr("BaiThuoc thiếu benh_ly / hoi_chung — khó truy hồi nhất quán")
    bt = run_read(driver, """
        MATCH (p:BaiThuoc)
        RETURN
          count(p) AS tong,
          sum(CASE WHEN p.benh_ly  IS NULL OR trim(coalesce(p.benh_ly,''))='' THEN 1 ELSE 0 END) AS thieu_benhly,
          sum(CASE WHEN p.hoi_chung IS NULL OR trim(coalesce(p.hoi_chung,''))='' THEN 1 ELSE 0 END) AS thieu_hoichung
    """)
    if bt:
        b = bt[0]
        print(f"  BaiThuoc: {b['tong']} node | thiếu benh_ly: {b['thieu_benhly']} | thiếu hoi_chung: {b['thieu_hoichung']}")

    _hr("HoiChung KHÔNG có bài thuốc điều trị — pháp trị sẽ rỗng")
    no_tx = run_read(driver, """
        MATCH (h:HoiChung)
        WHERE NOT (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(:BaiThuoc)
        RETURN count(DISTINCT h.name) AS c
    """)
    print(f"  Số hội chứng (theo name) chưa có ĐƯỢC_ĐIỀU_TRỊ_BẰNG: {no_tx[0]['c'] if no_tx else 0}")

    _hr("BaiThuoc.hoi_chung KHÔNG khớp HoiChung.name — bài thuốc sẽ 'biến mất' khỏi truy vấn")
    # App dùng điều kiện p.hoi_chung = h.name để truy hồi bài thuốc. Sau khi normalize-names,
    # số này phải = 0 (script tự đồng bộ). Nếu >0 trước khi dọn -> đã có sai lệch sẵn.
    mismatch = run_read(driver, """
        MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
        WHERE p.hoi_chung <> h.name
        RETURN count(p) AS c
    """)
    print(f"  Số BaiThuoc có hoi_chung lệch tên HoiChung nối tới: {mismatch[0]['c'] if mismatch else 0}")

    print("\n>>> Xong chẩn đoán (chỉ đọc). Dùng số liệu trên để quyết định bước --apply.\n")


# ---------------------------------------------------------------------------
# 2) DEDUPE HoiChung theo name
# ---------------------------------------------------------------------------
def dedupe_hoichung(driver, args):
    _hr("GỘP HoiChung TRÙNG TÊN (theo name)")
    groups = run_read(driver, """
        MATCH (h:HoiChung)
        WITH h.name AS name, collect(h) AS nodes, count(*) AS c
        WHERE c > 1
        RETURN name, c ORDER BY c DESC, name
    """)
    if not groups:
        print("  Không có node trùng tên. Không cần làm gì.")
        return
    print(f"  Có {len(groups)} nhóm trùng tên. Vài nhóm đầu:")
    for g in groups[:20]:
        print(f"    {g['c']:>3}x  {g['name']}")

    if not args.apply:
        print("\n  [DRY-RUN] Chưa thay đổi gì. Thêm --apply để gộp (giữ 1 node/tên, dồn hết quan hệ vào).")
        return

    if not has_apoc(driver):
        print("\n  LỖI: Cần APOC để gộp an toàn (apoc.refactor.mergeNodes).")
        print("  Neo4j Aura có sẵn APOC Core. Nếu self-host, cài plugin APOC rồi chạy lại.")
        print("  (Không tự gộp bằng Cypher thuần để tránh mất/nhân bản quan hệ.)")
        return

    print("\n  Đang gộp bằng apoc.refactor.mergeNodes (properties=combine, mergeRels=true)...")
    summary = run_write(driver, """
        MATCH (h:HoiChung)
        WITH h.name AS name, collect(h) AS nodes
        WHERE size(nodes) > 1
        CALL apoc.refactor.mergeNodes(nodes, {properties:'combine', mergeRels:true}) YIELD node
        RETURN count(*) AS merged
    """)
    print(f"  Đã gộp xong {len(groups)} nhóm. (nodes deleted: {summary.counters.nodes_deleted}, "
          f"rels deleted: {summary.counters.relationships_deleted})")
    print("  Gợi ý: chạy lại `diagnose` để xác nhận số node HoiChung đã giảm.")


# ---------------------------------------------------------------------------
# 3) CLEAN DIRTY HoiChung
# ---------------------------------------------------------------------------
def clean_dirty(driver, args):
    _hr("RÀ HoiChung 'BẨN' (tên nghi rác NLP)")
    dirty = run_read(driver, f"""
        MATCH (h:HoiChung) WHERE {_DIRTY_HOICHUNG_WHERE}
        OPTIONAL MATCH (h)--()
        WITH h, count(*) AS deg
        RETURN h.name AS name, deg ORDER BY deg ASC, name LIMIT 200
    """)
    if not dirty:
        print("  Không phát hiện node bẩn theo heuristic. (Có thể chỉnh heuristic trong _DIRTY_HOICHUNG_WHERE.)")
        return
    print(f"  Phát hiện {len(dirty)} node nghi bẩn (hiển thị tối đa 200), kèm bậc (số quan hệ):")
    for d in dirty:
        print(f"    deg={d['deg']:>3}  {repr(d['name'])}")

    if not args.apply:
        print("\n  [DRY-RUN] Chưa đổi gì.")
        print("  --apply           -> gắn cờ h._flagged_dirty=true (ĐẢO ĐƯỢC, để bạn review thủ công)")
        print("  --apply --delete  -> DETACH DELETE các node này (NGUY HIỂM, không đảo được)")
        return

    if args.delete:
        print("\n  ⚠️  XÓA HẲN các node bẩn (DETACH DELETE)...")
        summary = run_write(driver, f"""
            MATCH (h:HoiChung) WHERE {_DIRTY_HOICHUNG_WHERE}
            DETACH DELETE h
        """)
        print(f"  Đã xóa {summary.counters.nodes_deleted} node (và {summary.counters.relationships_deleted} quan hệ).")
    else:
        print("\n  Gắn cờ h._flagged_dirty=true (không xóa) để bạn review...")
        summary = run_write(driver, f"""
            MATCH (h:HoiChung) WHERE {_DIRTY_HOICHUNG_WHERE}
            SET h._flagged_dirty = true
        """)
        print(f"  Đã gắn cờ {summary.counters.properties_set} node.")
        print("  Xem lại: MATCH (h:HoiChung) WHERE h._flagged_dirty RETURN h.name")
        print("  Bỏ cờ  : MATCH (h:HoiChung) WHERE h._flagged_dirty REMOVE h._flagged_dirty")


# ---------------------------------------------------------------------------
# 3b) NORMALIZE-NAMES — chuẩn hoá tên HoiChung bẩn (bóc bài thuốc/giai đoạn khỏi tên)
# ---------------------------------------------------------------------------
# Tiền tố KHÔNG PHẢI hội chứng (giai đoạn/nguyên nhân/biến chứng...): khi tên bắt đầu bằng
# các cụm này, tên hội chứng THẬT nằm trong ngoặc đơn đầu tiên
# (vd "Giai đoạn mới phát (Phong nhiệt phạm phế)", "Di chứng (Ứ tắc kinh lạc não)").
# QUAN TRỌNG: nếu giữ tiền tố, nhiều hội chứng khác nhau sẽ gộp nhầm vào 1 node (vd 3 node
# "Di chứng (...)" -> "Di chứng") => phải bóc phần trong ngoặc.
_STAGE_PREFIXES = [
    "giai đoạn", "thể cấp", "thể mãn", "thể mạn", "mạn tính", "cấp tính",
    "phòng tái phát", "kinh nghiệm", "phòng ", "biến chứng", "di chứng",
    "do ", "sang chấn", "thời kỳ", "đợt ", "hoá mủ", "hóa mủ",
]


def _propose_clean_syndrome_name(name):
    """Đề xuất tên hội chứng SẠCH từ tên bẩn. Trả None nếu không tự làm sạch được (cần sửa tay).

    Quy tắc:
      - "X do Y" (không ngoặc)         -> lấy phần sau ' do ' (vd 'Cổ trướng do Âm hư huyết ứ' -> 'Âm hư huyết ứ')
      - tiền tố là giai đoạn/thể bệnh  -> tên thật nằm trong ngoặc đầu tiên
      - còn lại (tiền tố là hội chứng) -> bóc toàn bộ ngoặc, giữ tiền tố
    """
    if not name:
        return None
    original = name.strip()
    s = original

    # Động từ pháp trị (nếu đề xuất bắt đầu bằng các từ này -> là PHÁP TRỊ, không phải hội chứng)
    _TREAT_VERBS = ("bổ ", "kiện ", "thanh ", "ôn ", "hoạt ", "khu ", "trừ ", "tư ", "sơ ", "dưỡng ", "ích ", "lợi ", "nhuận ")

    def _reject(cand):
        # Loại đề xuất vẫn còn "bẩn"/không phải tên hội chứng -> để sửa tay
        if not cand or len(cand) < 3 or cand == original or len(cand) > 40:
            return True
        cl = cand.lower()
        if re.search(r"[0-9]", cand):
            return True
        if re.search(r"( hoặc | nếu | kèm )", " " + cl + " "):
            return True
        if cl.endswith(" do") or cl.endswith(" của") or cl.endswith(" và"):  # cụt/dangling
            return True
        if cl.startswith(_TREAT_VERBS):  # là pháp trị, không phải hội chứng
            return True
        return False

    def _finalize(cand):
        cand = re.sub(r"\s+", " ", (cand or "")).strip(" -.;,")
        if _reject(cand):
            return None
        return cand[0].upper() + cand[1:]  # viết hoa chữ đầu cho đúng kiểu tên hội chứng

    # Trường hợp "X do Y" không ngoặc
    if "(" not in s and " do " in (" " + s.lower() + " "):
        return _finalize(s.rsplit(" do ", 1)[-1])

    if "(" not in s:
        return None  # dài/rác nhưng không có cấu trúc rõ -> cần sửa tay

    parens = re.findall(r"\(([^()]*)\)", s)     # nội dung từng ngoặc đơn
    prefix = s.split("(", 1)[0].strip(" -")      # phần trước ngoặc đầu tiên
    prefix_l = prefix.lower()

    # "X - Y (Z)": vị trí hội chứng thật quá mơ hồ (có khi ở Y, có khi ở Z) -> để sửa tay
    if " - " in prefix:
        return None

    is_generic_prefix = prefix_l == "chung" or any(prefix_l.startswith(p) for p in _STAGE_PREFIXES)
    if is_generic_prefix and parens:
        return _finalize(parens[0])              # tên thật nằm trong ngoặc
    return _finalize(prefix)                       # bóc ngoặc, giữ tiền tố


def normalize_names(driver, args):
    _hr("CHUẨN HOÁ TÊN HoiChung BẨN (bóc bài thuốc/giai đoạn khỏi tên hội chứng)")
    dirty = run_read(driver, f"""
        MATCH (h:HoiChung) WHERE {_DIRTY_HOICHUNG_WHERE}
        RETURN h.name AS name ORDER BY name
    """)
    all_names = set(
        r["name"] for r in run_read(driver, "MATCH (h:HoiChung) RETURN h.name AS name") if r.get("name")
    )

    renames, manual = [], []
    for d in dirty:
        old = d["name"]
        new = _propose_clean_syndrome_name(old)
        if not new:
            manual.append(old)
            continue
        renames.append((old, new, new in all_names and new != old))

    print(f"  Tự đề xuất được: {len(renames)}  |  Cần sửa tay: {len(manual)}\n")
    for old, new, col in renames:
        tag = "MERGE→node đã có" if col else "đổi tên"
        print(f"  [{tag}]")
        print(f"      cũ : {old!r}")
        print(f"      mới: {new!r}")
    if manual:
        print("\n  --- CẦN SỬA TAY (không tự làm sạch an toàn được) ---")
        for m in manual:
            print(f"    • {m!r}")

    if not args.apply:
        print("\n  [DRY-RUN] Chưa đổi gì. HÃY REVIEW KỸ danh sách trên rồi thêm --apply.")
        print("  (Trùng tên -> node bẩn được MERGE vào node sạch cùng tên bằng APOC, gộp hết quan hệ.)")
        print("  Sau khi đổi tên, script TỰ đồng bộ thuộc tính BaiThuoc.hoi_chung theo tên mới")
        print("  (app truy hồi bài thuốc bằng p.hoi_chung = h.name -> bắt buộc khớp).")
        return

    if not has_apoc(driver):
        print("\n  LỖI: cần APOC (apoc.refactor.mergeNodes) để xử lý trùng tên an toàn.")
        print("  Neo4j Aura có sẵn APOC Core. Cài APOC rồi chạy lại.")
        return

    consumed = 0
    with driver.session() as s:
        for old, new, _col in renames:
            # MERGE node đích theo tên mới (tạo nếu chưa có), rồi gộp node bẩn vào -> đúng cho cả
            # trường hợp trùng sẵn LẪN nhiều node bẩn cùng quy về một tên trong cùng lần chạy.
            res = s.execute_write(lambda tx: tx.run("""
                MATCH (dirty:HoiChung {name:$old})
                MERGE (target:HoiChung {name:$new})
                WITH dirty, target WHERE elementId(dirty) <> elementId(target)
                CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
                RETURN node
            """, old=old, new=new).consume())
            consumed += res.counters.nodes_deleted
    print(f"\n  Xong: đã xử lý {consumed} node bẩn (đổi tên/gộp).")

    # [QUAN TRỌNG] Đồng bộ thuộc tính denormalized BaiThuoc.hoi_chung theo tên HoiChung mới.
    # App truy hồi bài thuốc bằng điều kiện `p.hoi_chung = h.name` (qa_system:320/329/476,
    # neo4j_client:31). Nếu đổi tên hội chứng mà không cập nhật p.hoi_chung, bài thuốc sẽ
    # "biến mất" khỏi mọi truy vấn. Query này đồng bộ toàn bộ (cả sai lệch có sẵn từ trước).
    sync = run_write(driver, """
        MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
        WHERE p.hoi_chung <> h.name
        SET p.hoi_chung = h.name
    """)
    print(f"  Đồng bộ BaiThuoc.hoi_chung theo tên mới: {sync.counters.properties_set} thuộc tính cập nhật.")
    print("  Chạy lại `diagnose` + `clean-dirty` để xác nhận số node bẩn đã giảm mạnh.")


# ---------------------------------------------------------------------------
# 3c) NORMALIZE-SOURCE — chuẩn hoá thuộc tính nguồn DongY.hội_chứng
# ---------------------------------------------------------------------------
def normalize_source(driver, args):
    _hr("CHUẨN HOÁ NGUỒN DongY.hội_chứng (để re-import bước 1-4 KHÔNG tái nhiễm tên bẩn)")
    try:
        rows = run_read(driver, "MATCH (d:DongY) WHERE d.`hội_chứng` IS NOT NULL "
                                "RETURN DISTINCT trim(d.`hội_chứng`) AS hc ORDER BY hc")
    except Exception as e:
        print(f"  Không đọc được DongY.hội_chứng: {e}")
        return
    if not rows:
        print("  Không thấy node DongY nào (có thể staging đã xoá sau import). Bỏ qua.")
        return

    renames = []
    for r in rows:
        old = r["hc"]
        new = _propose_clean_syndrome_name(old)
        if new and new != old:
            renames.append((old, new))

    print(f"  Tổng giá trị hội_chứng khác nhau: {len(rows)} | sẽ chuẩn hoá: {len(renames)}\n")
    for old, new in renames[:80]:
        print(f"    {old!r}\n      → {new!r}")
    if len(renames) > 80:
        print(f"    ... và {len(renames) - 80} giá trị khác")

    if not args.apply:
        print("\n  [DRY-RUN] Chưa đổi gì. --apply để cập nhật d.hội_chứng.")
        print("  (Đây là để RE-IMPORT về sau ra tên sạch; graph hiện tại đã sạch nên KHÔNG cần re-import ngay.)")
        return

    n = 0
    with driver.session() as s:
        for old, new in renames:
            res = s.execute_write(lambda tx: tx.run(
                "MATCH (d:DongY) WHERE trim(d.`hội_chứng`) = $old SET d.`hội_chứng` = $new",
                old=old, new=new).consume())
            n += res.counters.properties_set
    print(f"\n  Đã cập nhật {n} node DongY. Nguồn giờ khớp graph đã dọn -> re-import an toàn.")


# ---------------------------------------------------------------------------
# 4) PHỐI NGŨ — Thập bát phản (Tương phản) & Thập cửu úy (Tương úy)
# ---------------------------------------------------------------------------
# Dữ liệu KINH ĐIỂN. Quan hệ chỉ được tạo khi CẢ HAI vị thuốc đã tồn tại trong graph
# (khớp name không phân biệt hoa/thường). Tên có thể khác dị bản trong DB -> script sẽ
# BÁO CÁO cặp bị bỏ qua để bạn bổ sung alias nếu cần.

# Thập bát phản (18 phản) — quan hệ ĐỐI XỨNG (tạo 2 chiều): (a)-[:TƯƠNG_PHẢN]->(b) và ngược lại
THAP_BAT_PHAN = [
    ("Cam thảo", "Cam toại"),
    ("Cam thảo", "Đại kích"),
    ("Cam thảo", "Hải tảo"),
    ("Cam thảo", "Nguyên hoa"),
    ("Ô đầu", "Bối mẫu"),
    ("Ô đầu", "Qua lâu"),
    ("Ô đầu", "Bán hạ"),
    ("Ô đầu", "Bạch liễm"),
    ("Ô đầu", "Bạch cập"),
    ("Lê lô", "Nhân sâm"),
    ("Lê lô", "Sa sâm"),
    ("Lê lô", "Đan sâm"),
    ("Lê lô", "Huyền sâm"),
    ("Lê lô", "Khổ sâm"),
    ("Lê lô", "Tế tân"),
    ("Lê lô", "Bạch thược"),
]

# Thập cửu úy (19 úy) — quan hệ CÓ HƯỚNG: (a)-[:TƯƠNG_ÚY]->(b) nghĩa "a úy b"
THAP_CUU_UY = [
    ("Lưu hoàng", "Phác tiêu"),
    ("Thủy ngân", "Phê sương"),
    ("Lang độc", "Mật đà tăng"),
    ("Ba đậu", "Khiên ngưu"),
    ("Đinh hương", "Uất kim"),
    ("Xuyên ô", "Tê giác"),
    ("Thảo ô", "Tê giác"),
    ("Nha tiêu", "Tam lăng"),
    ("Quan quế", "Xích thạch chi"),
    ("Nhân sâm", "Ngũ linh chi"),
]


# Alias tên vị thuốc: 1 tên kinh điển ứng với nhiều node trong graph.
# "Ô đầu" là mẫu căn của Xuyên ô / Thảo ô / Phụ tử -> tạo phản với mọi biến thể CÓ trong graph.
VITHUOC_ALIAS = {
    "Ô đầu": ["Ô đầu", "Xuyên ô", "Thảo ô", "Phụ tử"],
    "Bán hạ": ["Bán hạ", "Bán hạ chế"],
    "Bối mẫu": ["Bối mẫu", "Xuyên bối mẫu", "Chiết bối mẫu", "Thổ bối mẫu"],
    "Cam thảo": ["Cam thảo", "Chích cam thảo"],
    "Qua lâu": ["Qua lâu", "Qua lâu nhân", "Qua lâu bì"],
}


def _existing_vithuoc(driver):
    rows = run_read(driver, "MATCH (v:ViThuoc) RETURN v.name AS name")
    return {r["name"].strip().lower(): r["name"] for r in rows if r.get("name")}


def add_phoi_ngu(driver, args):
    _hr("THÊM QUAN HỆ PHỐI NGŨ (Thập bát phản / Thập cửu úy)")
    existing = _existing_vithuoc(driver)
    print(f"  Graph hiện có {len(existing)} vị thuốc.")

    def resolve_all(term):
        """Trả về mọi tên node ViThuoc có trong graph ứng với 1 tên kinh điển (qua alias)."""
        out = []
        for c in VITHUOC_ALIAS.get(term, [term]):
            node = existing.get(c.strip().lower())
            if node and node not in out:
                out.append(node)
        return out

    def _missing(a, b, as_, bs_):
        miss = []
        if not as_:
            miss.append(a)
        if not bs_:
            miss.append(b)
        return "thiếu: " + ", ".join(miss)

    plan_phan, skip_phan, seen_phan = [], [], set()
    for a, b in THAP_BAT_PHAN:
        as_, bs_ = resolve_all(a), resolve_all(b)
        if as_ and bs_:
            for x in as_:
                for y in bs_:
                    key = frozenset((x, y))  # phản là đối xứng -> khử trùng theo cặp không hướng
                    if x != y and key not in seen_phan:
                        seen_phan.add(key)
                        plan_phan.append((x, y))
        else:
            skip_phan.append((a, b, _missing(a, b, as_, bs_)))

    plan_uy, skip_uy, seen_uy = [], [], set()
    for a, b in THAP_CUU_UY:
        as_, bs_ = resolve_all(a), resolve_all(b)
        if as_ and bs_:
            for x in as_:
                for y in bs_:
                    if x != y and (x, y) not in seen_uy:  # úy có hướng -> khử trùng theo cặp có hướng
                        seen_uy.add((x, y))
                        plan_uy.append((x, y))
        else:
            skip_uy.append((a, b, _missing(a, b, as_, bs_)))

    print(f"\n  TƯƠNG_PHẢN: {len(plan_phan)} cặp tạo được, {len(skip_phan)} cặp bỏ qua (thiếu vị thuốc)")
    for a, b in plan_phan:
        print(f"    ✓ {a}  ⟷  {b}")
    for a, b, why in skip_phan:
        print(f"    ✗ {a}  ⟷  {b}   ({why})")

    print(f"\n  TƯƠNG_ÚY: {len(plan_uy)} cặp tạo được, {len(skip_uy)} cặp bỏ qua")
    for a, b in plan_uy:
        print(f"    ✓ {a}  →  {b}")
    for a, b, why in skip_uy:
        print(f"    ✗ {a}  →  {b}   ({why})")

    if not args.apply:
        print("\n  [DRY-RUN] Chưa tạo quan hệ nào. Thêm --apply để MERGE các cặp tạo được ở trên.")
        return

    created = 0
    with driver.session() as s:
        # TƯƠNG_PHẢN: đối xứng -> tạo 2 chiều
        for a, b in plan_phan:
            res = s.execute_write(lambda tx: tx.run("""
                MATCH (x:ViThuoc {name:$a}), (y:ViThuoc {name:$b})
                MERGE (x)-[r1:TƯƠNG_PHẢN]->(y)
                MERGE (y)-[r2:TƯƠNG_PHẢN]->(x)
                SET r1.nguon='Thập bát phản', r2.nguon='Thập bát phản'
            """, a=a, b=b).consume())
            created += res.counters.relationships_created
        # TƯƠNG_ÚY: có hướng
        for a, b in plan_uy:
            res = s.execute_write(lambda tx: tx.run("""
                MATCH (x:ViThuoc {name:$a}), (y:ViThuoc {name:$b})
                MERGE (x)-[r:TƯƠNG_ÚY]->(y)
                SET r.nguon='Thập cửu úy'
            """, a=a, b=b).consume())
            created += res.counters.relationships_created
    print(f"\n  Đã tạo {created} quan hệ phối ngũ mới (MERGE nên chạy lại không nhân bản).")
    print("  Kiểm tra: MATCH (a:ViThuoc)-[r:TƯƠNG_PHẢN|TƯƠNG_ÚY]->(b) RETURN a.name, type(r), b.name")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Bảo trì Knowledge Graph TCM (Neo4j). Mặc định DRY-RUN.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("diagnose", help="Chẩn đoán trạng thái graph (CHỈ ĐỌC)")

    sp = sub.add_parser("dedupe-hoichung", help="Gộp HoiChung trùng tên")
    sp.add_argument("--apply", action="store_true", help="Thực thi (mặc định dry-run)")

    sp = sub.add_parser("clean-dirty", help="Rà/gắn cờ/xóa HoiChung bẩn")
    sp.add_argument("--apply", action="store_true", help="Thực thi (gắn cờ)")
    sp.add_argument("--delete", action="store_true", help="Kèm --apply: XÓA hẳn thay vì gắn cờ")

    sp = sub.add_parser("normalize-names", help="Chuẩn hoá tên HoiChung bẩn (bóc bài thuốc/giai đoạn)")
    sp.add_argument("--apply", action="store_true", help="Thực thi (mặc định dry-run)")

    sp = sub.add_parser("normalize-source", help="Chuẩn hoá nguồn DongY.hội_chứng (để re-import không tái nhiễm)")
    sp.add_argument("--apply", action="store_true", help="Thực thi (mặc định dry-run)")

    sp = sub.add_parser("add-phoi-ngu", help="Thêm quan hệ Tương phản/Tương úy cổ điển")
    sp.add_argument("--apply", action="store_true", help="Thực thi (mặc định dry-run)")

    args = p.parse_args()
    driver = get_driver()
    try:
        {
            "diagnose": diagnose,
            "dedupe-hoichung": dedupe_hoichung,
            "clean-dirty": clean_dirty,
            "normalize-names": normalize_names,
            "normalize-source": normalize_source,
            "add-phoi-ngu": add_phoi_ngu,
        }[args.cmd](driver, args)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
