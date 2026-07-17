#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_symptom_concepts.py — Test tầng [KHÁI NIỆM TRIỆU CHỨNG] (Cách 3) của _find_matching_diseases.

Bối cảnh: người bệnh khai 'ỉa chảy' còn CSV ghi 'đi ngoài nhiều lần' -> Cách 1 (chuỗi con) và Cách 2
(mọi-từ-trong-một-phân-đoạn) đều trượt, khiến bệnh ĐÚNG tụt ratio và thua tie-break IDF trước bệnh
lạc đề. data/symptom_concepts.json bắc cầu đồng nghĩa; test này khóa cả LỢI lẫn HẠI của nó.

Chạy:  python scripts/test_symptom_concepts.py   (exit != 0 nếu fail). Không cần Neo4j/LLM.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as F

_o = None


def pipe():
    global _o
    if _o is None:
        _o = F.__new__(F)
        F._load_csv_data(_o)
    return _o


# (text, khái niệm kỳ vọng) — khóa ranh giới của khái niệm 'tiêu chảy'
CONCEPT_CASES = [
    ("ỉa chảy", {"tiêu chảy"}),
    ("đại tiện lỏng", {"tiêu chảy"}),
    ("đi ngoài nhiều lần", {"tiêu chảy"}),
    ("tiêu chảy kéo dài", {"tiêu chảy"}),
    ("phân lỏng nát", {"tiêu chảy"}),
    ("táo hoặc ỉa chảy", {"tiêu chảy"}),          # field mô tả pha lỏng xen kẽ vẫn là bằng chứng
    # ⚠ Ranh giới PHẢI giữ — gộp nhầm là khớp oan bệnh khác:
    ("sôi bụng", set()),                          # trường minh, KHÔNG phải tiêu chảy
    ("đau bụng", set()),
    ("đại tiện táo", set()),                      # táo bón là cực NGƯỢC
    ("đi ngoài ra máu", set()),                   # huyết tiện/trĩ — veto
    ("đại tiện ra máu", set()),
    ("đi ngoài mót rặn", set()),                  # lỵ tật — veto
    ("tiêu chảy ra máu", set()),                  # có trigger NHƯNG veto thắng -> lỵ, không phải tả thường
    # ⚠ Trigger phải NEO vào phân/đại tiện. Tính từ độ đặc trần tả cả đờm/bạch đới/chất nôn:
    # từng có bug thật do nạp 'lỏng loãng'/'lỏng nhão' làm trigger.
    ("ho đờm trắng lỏng loãng", set()),
    ("bạch đới lỏng loãng", set()),
    ("khạc đờm lỏng nhão", set()),
    ("nôn ra nước lỏng loãng", set()),
    # ... nhưng CÙNG tính từ đó khi ĐÃ neo vào đại tiện thì vẫn phải nhận ra:
    ("đại tiện lỏng nhão", {"tiêu chảy"}),
    ("đại tiện lỏng loãng", {"tiêu chảy"}),
    ("đại tiện táo hoặc lỏng nát", {"tiêu chảy"}),
]


def test_concepts():
    o = pipe()
    npass = nfail = 0
    print("== (1) _concepts_of_text — ranh giới khái niệm ==")
    for text, expect in CONCEPT_CASES:
        got = set(o._concepts_of_text(text))
        ok = (got == expect)
        print(f"[{'PASS' if ok else 'FAIL'}] {text!r} -> {sorted(got) or '∅'} (kỳ vọng {sorted(expect) or '∅'})")
        npass += ok
        nfail += (not ok)
    return npass, nfail


def test_bridge_helps():
    """Bắc cầu phải LÀM ĐƯỢC VIỆC — và test phải chứng minh chính TẦNG NÀY làm, không phải thứ khác.

    ⚠ Bài học: bản test đầu chỉ assert 'Viêm đại tràng là bệnh danh #1' cho ca thật, nhưng nó PASS cả
    khi tắt hẳn tầng khái niệm — vì dòng KB 'Viêm đại tràng × Tỳ khí hư' (thêm cùng đợt) tự khớp chữ
    đủ để thắng. Test đó khóa nhầm thứ. Nay neo vào dòng 794 (Thấp nhiệt) và LOẠI dòng Tỳ khí hư khỏi
    corpus, để đường duy nhất đạt ratio 0.60 là bắc cầu 'ỉa chảy' -> field 'đi ngoài nhiều lần'
    (khớp chữ chỉ cho 0.40). Tắt concepts -> assertion này PHẢI sập."""
    o = pipe()
    npass = nfail = 0
    print("\n== (2) Bắc cầu CỨU ca tiêu hóa — cô lập đúng tầng ==")
    _saved = o.csv_rows
    try:
        o.csv_rows = [r for r in _saved
                      if not (r.get("benh_ly", "").strip() == "Viêm đại tràng"
                              and r.get("hoi_chung", "").strip() == "Tỳ khí hư")]
        sym = ["ỉa chảy", "đau bụng", "nóng rát hậu môn"]
        cands = [c for c in o._find_matching_diseases(sym, ", ".join(sym))
                 if c["benh_ly"] == "Viêm đại tràng"]
        got = cands[0]["ratio"] if cands else 0.0
        # khớp chữ thuần = 2/5 = 0.40 (lời khai 'ỉa chảy' không chạm được field 'đi ngoài nhiều lần');
        # có bắc cầu = 3/5 = 0.60. Ngưỡng 0.55 tách bạch hai đường.
        ok = got >= 0.55
        print(f"[{'PASS' if ok else 'FAIL'}] 'ỉa chảy' bắc cầu tới field 'đi ngoài nhiều lần': "
              f"Viêm đại tràng × Thấp nhiệt ratio={got:.3f} (cần >=0.55; khớp chữ thuần chỉ 0.40)")
        npass += ok
        nfail += (not ok)
    finally:
        o.csv_rows = _saved

    # Ca THẬT đầy đủ: bệnh chính phải đúng (dù đường nào tới cũng được — KB row hoặc bắc cầu).
    sym = ["đại tiện xong đỡ đau", "đỏ nóng rát hậu môn", "ra mồ hôi nhiều", "đại tiện lỏng",
           "sôi bụng", "đau bụng", "mệt mỏi", "ỉa chảy", "mặt nhợt nhạt", "quầng đen dưới mắt",
           "rêu trắng mỏng"]
    names = [c["benh_ly"] for c in o._find_matching_diseases(sym, ", ".join(sym))]
    top = names[0] if names else None
    for label, ok in [
        ("ca thật: Viêm đại tràng là bệnh danh #1", top == "Viêm đại tràng"),
        ("ca thật: Viêm đại tràng xếp TRÊN Xuất hãn dị thường",
         "Viêm đại tràng" in names and ("Xuất hãn dị thường" not in names
                                        or names.index("Viêm đại tràng") < names.index("Xuất hãn dị thường"))),
    ]:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}  (top: {names[:3]})")
        npass += ok
        nfail += (not ok)
    return npass, nfail


def test_bridge_does_not_harm():
    """Ca mồ hôi THUẦN (không triệu chứng tiêu hóa) KHÔNG được vì bắc cầu mà lệch sang bệnh ruột."""
    o = pipe()
    npass = nfail = 0
    print("\n== (3) Bắc cầu KHÔNG phá ca mồ hôi thuần ==")
    sym = ["tự ra mồ hôi nhiều", "mệt mỏi đoản khí", "ăn kém", "dễ bị cảm mạo", "lưỡi đỏ nhạt"]
    cands = o._find_matching_diseases(sym, ", ".join(sym))
    names = [c["benh_ly"] for c in cands]
    for label, ok in [
        ("Xuất hãn dị thường vẫn còn trong ứng viên", "Xuất hãn dị thường" in names),
        ("KHÔNG bệnh ruột nào chen lên #1",
         bool(names) and names[0] not in ("Viêm đại tràng", "Tiết tả", "Tiết tả tính", "Nhi tiết tả")),
    ]:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}  (top: {names[:3]})")
        npass += ok
        nfail += (not ok)
    return npass, nfail


def test_count_once():
    """[ĐẾM 1 LẦN/KHÁI NIỆM] Một lời khai không được thổi ratio qua NHIỀU field cùng nghĩa một dòng.
    Dòng 'Viêm đại tràng × Tỳ thận dương hư' có CẢ 'tiêu chảy kéo dài' lẫn 'phân lỏng nát' — nếu đếm
    cả hai thì ca 'Nội thương phát nhiệt × Dương hư phát nhiệt' bị soán top-1 (đã tái hiện thật)."""
    o = pipe()
    npass = nfail = 0
    print("\n== (4) Đếm 1 lần/khái niệm — chống thổi ratio ==")
    sym = ["Sốt mà lại muốn mặc thêm áo ấm", "sợ lạnh", "chân tay lạnh", "đầu váng hay quên",
           "lưng mỏi gối lạnh", "đại tiện lỏng", "có hằn răng"]
    cands = o._find_matching_diseases(sym, ", ".join(sym))
    names = [c["benh_ly"] for c in cands]
    top = names[0] if names else None
    ok = (top == "Nội thương phát nhiệt")
    print(f"[{'PASS' if ok else 'FAIL'}] bệnh thật giữ được #1, không bị Viêm đại tràng soán "
          f"(top: {names[:3]})")
    npass += ok
    nfail += (not ok)
    return npass, nfail


def main():
    tp = tf = 0
    for fn in (test_concepts, test_bridge_helps, test_bridge_does_not_harm, test_count_once):
        p, f = fn()
        tp += p
        tf += f
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
