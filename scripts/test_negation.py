#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_negation.py — Khóa hành vi CHỐNG PHỦ ĐỊNH trong trích triệu chứng văn bản.

Bối cảnh: _extract_symptoms_by_matching (qa_system) khớp longest-match thuần, nên 'không sốt, ho
khan' từng trích ra 'sốt' rồi cộng điểm oan cho hội chứng nhiệt. Nay các triệu chứng nằm trong tầm
phủ định ('không/chưa/chẳng' đứng trước, cùng mệnh đề) bị loại — NHƯNG các triệu chứng mà bản thân
TÊN đã chứa 'không' ('miệng nhạt không khát', 'tay chân không ấm') phải được GIỮ nguyên.

Chạy:  python scripts/test_negation.py    (Exit 0 nếu PASS — không cần Neo4j, dùng vocab giả lập)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.qa_system import TCMQA
from src.fusion_pipeline import TCMFusionPipeline as F


# Từ vựng triệu chứng giả lập: có cả triệu chứng thường VÀ triệu chứng chứa 'không' trong tên.
VOCAB = [
    "sốt", "ho khan", "ho", "mệt mỏi", "sợ lạnh", "khát", "đau đầu", "chóng mặt",
    "ớn lạnh", "buồn nôn", "táo bón",
    "miệng nhạt không khát", "tay chân không ấm", "đại tiện không thông", "không muốn ăn",
]


class _FakeNeo4j:
    def get_all_symptoms(self):
        return list(VOCAB)


# (input, tập kỳ vọng CÓ, tập kỳ vọng KHÔNG có)
CASES = [
    ("không sốt, ho khan",              {"ho khan"},                 {"sốt"}),
    ("sốt cao, không ho",               {"sốt"},                     {"ho", "ho khan"}),
    ("mệt mỏi nhưng không sợ lạnh",     {"mệt mỏi"},                 {"sợ lạnh"}),
    ("không sốt không ho khan",         set(),                       {"sốt", "ho khan"}),
    # Triệu chứng có 'không' trong TÊN -> phải GIỮ, và không được phủ định hàng xóm:
    ("miệng nhạt không khát, mệt mỏi",  {"miệng nhạt không khát", "mệt mỏi"}, {"khát"}),
    ("tay chân không ấm, sợ lạnh",      {"tay chân không ấm", "sợ lạnh"},     set()),
    ("bụng đầy, không muốn ăn",         {"không muốn ăn"},           set()),
    # 'không những ... mà còn ...' = nhấn mạnh CÓ, không phủ định:
    ("không những mệt mỏi mà còn sốt",  {"mệt mỏi", "sốt"},          set()),
    # Không có phủ định -> giữ nguyên (sanity):
    ("đau đầu, chóng mặt",              {"đau đầu", "chóng mặt"},    set()),
    # Phủ định đứng SAU triệu chứng -> không loại triệu chứng đứng trước:
    ("khát nước nhưng không nhiều",     {"khát"},                    set()),
]


# Cổng hàn-nhiệt (_kw_hit_clean): (text, keyword, kỳ vọng có khớp?)
GATE_CASES = [
    ("không sốt, ho khan",            "sốt",       False),   # 'sốt' bị phủ định -> KHÔNG là dấu nhiệt
    ("sốt cao ho khan",               "sốt",       True),
    ("người mệt, không sợ lạnh",      "sợ lạnh",   False),   # phủ định -> không phải dấu hàn
    ("sợ lạnh, tay chân lạnh",        "sợ lạnh",   True),
    ("mệt mỏi nhưng không khát nước", "khát nước", False),
    ("khát nước nhiều",               "khát nước", True),
    # 'không những ... mà còn sốt' = nhấn mạnh CÓ -> vẫn tính là dấu nhiệt:
    ("không những mệt mà còn sốt",    "sốt",       True),
    # cụm tên chứa 'không' đứng trước dấu hàn khác dấu phẩy -> dấu hàn sau phẩy vẫn tính:
    ("miệng nhạt không khát, sợ lạnh","sợ lạnh",   True),
]


def run():
    q = TCMQA.__new__(TCMQA)          # bỏ __init__ (không kết nối Neo4j/LLM)
    q.neo4j_client = _FakeNeo4j()

    all_ok = True
    for text, must_have, must_not in CASES:
        got = set(x.lower() for x in q._extract_symptoms_by_matching(text.lower()))
        fails = []
        for good in must_have:
            if good not in got:
                fails.append(f"THIẾU '{good}'")
        for bad in must_not:
            if bad in got:
                fails.append(f"CÓ '{bad}' (đáng lẽ bị phủ định/không trích)")
        status = "PASS" if not fails else "FAIL"
        if fails:
            all_ok = False
        print(f"[{status}] {text!r}")
        print(f"        trích được: {sorted(got)}")
        for fl in fails:
            print(f"        ✗ {fl}")

    print("── Cổng hàn-nhiệt (_kw_hit_clean) sau khi che phủ định ──")
    for text, kw, expect in GATE_CASES:
        got = F._kw_hit_clean(text.lower(), [kw])
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            all_ok = False
        print(f"  [{status}] _kw_hit_clean({text!r}, [{kw!r}]) = {got} (kỳ vọng {expect})")

    print("\n" + ("✅ TẤT CẢ PASS" if all_ok else "❌ CÓ CA FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
