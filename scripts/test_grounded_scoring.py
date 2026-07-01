#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test GROUNDED SYNDROME-SCORING trên Neo4j thật (không cần Ollama).
Chấm điểm mỗi HoiChung = số triệu chứng bệnh nhân khớp với triệu chứng của nó (CÓ_BIỂU_HIỆN).
Chạy để KIỂM TRA ranking có hợp lý không TRƯỚC khi tin dùng trong app.

Cách dùng (từ thư mục gốc dự án):
    python scripts/test_grounded_scoring.py
    python scripts/test_grounded_scoring.py "mệt mỏi, đại tiện lỏng, hằn răng, mặt nhợt"
"""
import os
import sys

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


def word_boundary(term: str) -> str:
    # Giống hệt qa_system._word_boundary_pattern: khớp từ độc lập, bắt cả node triệu chứng ghép
    return r'.*(^|[^\p{L}])\Q' + term.lower() + r'\E($|[^\p{L}]).*'


def main():
    if len(sys.argv) > 1:
        terms = [t.strip() for t in sys.argv[1].split(",") if t.strip()]
    else:
        # Ca "Tỳ khí hư" user hay test
        terms = ["mệt mỏi", "đại tiện lỏng", "tinh thần uể oải", "rìa lưỡi có hằn răng",
                 "mặt nhợt nhạt", "lưỡi đỏ", "rêu trắng dày", "vùng đỏ trên mặt"]

    cfg = load_config()
    neo = cfg.get("neo4j", {})
    uri = neo.get("uri") or os.getenv("NEO4J_URI")
    user = neo.get("user") or os.getenv("NEO4J_USER")
    pwd = neo.get("password") or os.getenv("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        print("LỖI: thiếu NEO4J_URI/USER/PASSWORD trong .env")
        sys.exit(1)

    patterns = [word_boundary(t) for t in terms]
    # Điểm cân bằng = matched + idf_sum (coverage chính + đặc hiệu phá hoà). Loại node bẩn (số/ngoặc/cờ).
    cypher = """
    UNWIND $patterns AS pat
    MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
    WHERE toLower(t.name) =~ pat
      AND NOT h.name =~ '.*[0-9(].*'
      AND NOT coalesce(h._flagged_dirty, false)
      AND NOT toLower(h.name) STARTS WITH 'thể '
    WITH pat, collect(DISTINCT h.name) AS syns
    WITH pat, syns, toFloat(size(syns)) AS df
    UNWIND syns AS syndrome
    WITH syndrome, sum(1.0/df) AS idf, count(DISTINCT pat) AS matched
    RETURN syndrome, (matched + idf) AS score, matched
    ORDER BY score DESC, syndrome LIMIT 20
    """
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session() as s:
            rows = s.execute_read(lambda tx: list(tx.run(cypher, patterns=patterns)))
    finally:
        driver.close()

    print(f"Triệu chứng test ({len(terms)}): {', '.join(terms)}\n")
    print("Xếp hạng hội chứng theo điểm IDF (đặc hiệu cao->thấp) | matched = số triệu chứng khớp:")
    if not rows:
        print("  (Không hội chứng nào khớp — kiểm tra lại tên triệu chứng có trùng node TrieuChung không)")
    for r in rows:
        print(f"  điểm {r['score']:.2f}  (khớp {r['matched']})  |  {r['syndrome']}")
    print("\n>>> Kỳ vọng: hội chứng ĐẶC HIỆU (khớp nhiều triệu chứng hiếm) ở TOP, không phải hội chứng chung chung.")


if __name__ == "__main__":
    main()
