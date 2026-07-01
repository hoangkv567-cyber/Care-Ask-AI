#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
LÀM GIÀU SYMPTOM LIST cho các HoiChung CỐT LÕI (KG enrichment)
================================================================================
Bổ sung triệu chứng KINH ĐIỂN (chuẩn Đông y) cho các hội chứng cốt lõi còn THƯA,
để bước grounded-scoring chọn được hội chứng ĐẶC HIỆU (vd Tỳ khí hư thay vì Khí hư chung).

Kết nối qua .env. MẶC ĐỊNH DRY-RUN. Cạnh thêm được đánh dấu `nguon:'enrich'`, KHÔNG gắn benh_ly
=> chỉ giúp chấm điểm hội chứng, KHÔNG ảnh hưởng truy hồi bài thuốc (các query đó lọc r.benh_ly=b.name).

⚠️ REVIEW TRƯỚC KHI --apply: Bảng SYNDROME_SYMPTOMS dưới đây là tri thức Đông y do tôi soạn theo
sách vở — BẠN (người có chuyên môn) HÃY RÀ LẠI, thêm/bớt cho đúng trước khi chạy --apply.

Cách dùng (từ thư mục gốc dự án):
    python scripts/kg_enrich_symptoms.py            # xem trước (syndrome nào có, triệu chứng nào mới)
    python scripts/kg_enrich_symptoms.py --apply    # tạo cạnh CÓ_BIỂU_HIỆN
    python scripts/kg_enrich_symptoms.py --undo     # gỡ toàn bộ cạnh nguon='enrich' (đảo được)
================================================================================
"""
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from neo4j import GraphDatabase          # noqa: E402
from src.config_loader import load_config  # noqa: E402


# ==========================================================================
# BẢNG TRI THỨC — hội chứng cốt lõi -> triệu chứng kinh điển đặc hiệu.
# HÃY RÀ SOÁT/CHỈNH theo chuyên môn Đông y của bạn trước khi --apply.
# Chỉ nên thêm triệu chứng ĐẶC HIỆU (đủ để phân biệt), tránh triệu chứng quá chung.
# ==========================================================================
SYNDROME_SYMPTOMS = {
    "Tỳ khí hư": ["mệt mỏi", "ăn kém", "đại tiện lỏng", "đầy bụng", "rìa lưỡi có hằn răng",
                  "lưỡi bệu", "sắc mặt vàng", "đoản khí", "tinh thần uể oải"],
    "Tỳ dương hư": ["sợ lạnh", "tay chân lạnh", "đại tiện lỏng", "đau bụng ưa ấm", "mệt mỏi",
                    "lưỡi nhợt", "rìa lưỡi có hằn răng"],
    "Vị khí hư": ["ăn kém", "đầy bụng", "mệt mỏi", "ợ hơi", "buồn nôn"],
    "Thận âm hư": ["đau lưng mỏi gối", "ù tai", "ra mồ hôi trộm", "ngũ tâm phiền nhiệt",
                   "chóng mặt", "họng khô", "lưỡi đỏ"],
    "Thận dương hư": ["sợ lạnh", "tay chân lạnh", "đau lưng lạnh", "tiểu đêm", "mỏi gối", "lưỡi nhợt"],
    "Khí huyết hư": ["mệt mỏi", "sắc mặt nhợt", "chóng mặt", "hoa mắt", "hồi hộp", "hay quên", "lưỡi nhợt"],
    "Tâm huyết hư": ["hồi hộp", "mất ngủ", "hay quên", "sắc mặt nhợt", "chóng mặt", "lưỡi nhợt"],
    "Can huyết hư": ["hoa mắt", "chóng mặt", "tê tay chân", "sắc mặt nhợt", "móng tay nhợt"],
    "Can khí uất kết": ["ngực sườn đầy tức", "hay thở dài", "cáu gắt", "tinh thần uất ức", "ợ hơi"],
    "Phế khí hư": ["ho", "đoản khí", "mệt mỏi", "tự ra mồ hôi", "tiếng nói nhỏ"],
    "Phế âm hư": ["ho khan", "ít đờm", "họng khô", "ra mồ hôi trộm", "lưỡi đỏ", "gò má đỏ"],
    "Đàm thấp": ["nặng đầu", "rêu lưỡi dày nhớt", "người béo", "chóng mặt", "buồn nôn", "tức ngực"],
    "Thấp nhiệt": ["miệng đắng", "rêu lưỡi vàng nhớt", "tiểu vàng", "người nặng nề"],
    "Huyết ứ": ["đau cố định", "lưỡi tím", "lưỡi có điểm ứ huyết", "da sạm", "môi tím"],
    "Âm hư": ["ra mồ hôi trộm", "ngũ tâm phiền nhiệt", "gò má đỏ", "họng khô", "lưỡi đỏ", "chóng mặt"],
}


def get_driver():
    cfg = load_config()
    neo = cfg.get("neo4j", {})
    uri = neo.get("uri") or os.getenv("NEO4J_URI")
    user = neo.get("user") or os.getenv("NEO4J_USER")
    pwd = neo.get("password") or os.getenv("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        print("LỖI: thiếu NEO4J_URI/USER/PASSWORD trong .env")
        sys.exit(1)
    return GraphDatabase.driver(uri, auth=(user, pwd))


def main():
    p = argparse.ArgumentParser(description="Làm giàu symptom list cho HoiChung cốt lõi. Mặc định DRY-RUN.")
    p.add_argument("--apply", action="store_true", help="Thực thi tạo cạnh (mặc định chỉ xem trước)")
    p.add_argument("--undo", action="store_true", help="Gỡ toàn bộ cạnh CÓ_BIỂU_HIỆN nguon='enrich'")
    args = p.parse_args()
    driver = get_driver()
    try:
        if args.undo:
            with driver.session() as s:
                res = s.execute_write(lambda tx: tx.run(
                    "MATCH (:HoiChung)-[r:CÓ_BIỂU_HIỆN {nguon:'enrich'}]->(:TrieuChung) DELETE r"
                ).consume())
            print(f"Đã gỡ {res.counters.relationships_deleted} cạnh enrich.")
            return

        # Kiểm tra: syndrome nào tồn tại trong graph, triệu chứng nào đã có / sẽ thêm mới
        with driver.session() as s:
            existing_syns = set(r["name"] for r in s.execute_read(
                lambda tx: list(tx.run("MATCH (h:HoiChung) RETURN h.name AS name"))) if r.get("name"))

        total_new = 0
        plan = []  # (syndrome, [symptoms])
        print("=" * 70)
        print("KẾ HOẠCH LÀM GIÀU (triệu chứng sẽ thêm cho mỗi hội chứng)")
        print("=" * 70)
        for syn, symptoms in SYNDROME_SYMPTOMS.items():
            if syn not in existing_syns:
                print(f"\n  ✗ '{syn}' — KHÔNG có node HoiChung này trong graph (bỏ qua)")
                continue
            print(f"\n  ✓ '{syn}':")
            for sym in symptoms:
                print(f"      + {sym}")
            plan.append((syn, symptoms))
            total_new += len(symptoms)
        print(f"\n  => {len(plan)} hội chứng, tổng {total_new} cạnh (MERGE nên chạy lại không nhân đôi).")

        if not args.apply:
            print("\n  [DRY-RUN] Chưa tạo gì. HÃY RÀ SOÁT bảng SYNDROME_SYMPTOMS (chỉnh trong file nếu cần),")
            print("  rồi chạy lại với --apply. Gỡ toàn bộ bằng --undo.")
            return

        created = 0
        with driver.session() as s:
            for syn, symptoms in plan:
                for sym in symptoms:
                    res = s.execute_write(lambda tx: tx.run("""
                        MATCH (h:HoiChung {name:$syn})
                        MERGE (t:TrieuChung {name:$sym})
                        MERGE (h)-[r:CÓ_BIỂU_HIỆN {nguon:'enrich'}]->(t)
                        RETURN r
                    """, syn=syn, sym=sym).consume())
                    created += res.counters.relationships_created
        print(f"\n  Đã tạo {created} cạnh CÓ_BIỂU_HIỆN mới (nguon='enrich').")
        print("  Kiểm chứng: chạy lại `python scripts/test_grounded_scoring.py` xem Tỳ khí hư đã lên top chưa.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
