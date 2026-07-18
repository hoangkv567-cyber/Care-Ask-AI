#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_duonghu_cold_gate.py — Khóa hai lỗi đã gây "core ngoại lai + Mục 5 trắng".

BỐI CẢNH (ca thật): "tiếng nấc nông và yếu, tay chân mát, đoản khí, mệt mỏi, ăn ít".
Bộ chấm điểm xếp ĐÚNG 'Tỳ thận dương hư' hạng 1 (6.576, khớp 5 triệu chứng), nhưng [CỔNG DƯƠNG HƯ]
coi lời khai "không có dấu hàn" vì danh sách dấu hàn thiếu 'tay chân mát' -> hạ bậc hội chứng đúng
-> core rơi sang 'Chính hư ứ kết' (thể của bệnh KHÁC: Tích tụ) -> Mục 5 TRẮNG + Mục 4 bịa thực tích.

(1) 'mát' neo BỘ PHẬN phải được tính là dấu hàn; 'mát' trong ngữ cảnh KHÁT/UỐNG là dấu NHIỆT,
    TUYỆT ĐỐI không được tính (đảo cực hàn-nhiệt là lớp lỗi nguy hiểm nhất của dự án).
(2) Cổng grounding phải re-rank được core ngoại lai về thể thuộc bệnh trong cửa sổ.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_duonghu_cold_gate.py   (exit != 0 nếu fail)
Không cần Neo4j/LLM.
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


# Dùng ĐÚNG danh sách thật trong mã nguồn (hằng số lớp) — không chép tay, để test không bị lệch
# với code khi ai đó sửa một bên.
COLD_KWS = list(F._DUONGHU_COLD_KWS)


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_source_list_intact():
    """Nhóm 'mát' phải CÒN trong hằng số thật — nếu bị gỡ, ca nấc hư hàn sẽ tái lỗi."""
    print("== (1) Nhóm 'mát' còn trong _DUONGHU_COLD_KWS ==")
    kws = set(F._DUONGHU_COLD_KWS)
    n = f = 0
    for kw in ("tay chân mát", "chân tay mát", "tứ chi mát", "chi mát", "da mát", "mát lạnh"):
        ok = kw in kws
        n += _chk(f"còn '{kw}'", ok)
        f += (not ok)
    # 'mát' TRẦN phải KHÔNG có (nó là dấu NHIỆT trong 'thích uống nước mát')
    ok = "mát" not in kws
    n += _chk("KHÔNG có 'mát' trần (chống đảo cực hàn-nhiệt)", ok)
    f += (not ok)
    # các dấu hàn gốc phải còn nguyên
    for kw in ("sợ lạnh", "tay chân lạnh", "tiểu đêm", "ngũ canh"):
        ok = kw in kws
        n += _chk(f"giữ dấu hàn gốc '{kw}'", ok)
        f += (not ok)
    return n, f


def test_cold_detection():
    o = pipe()
    print("\n== (2) Nhận dạng dấu hàn ==")
    n = f = 0
    for txt in ["tiếng nấc nông và yếu, tay chân mát, đoản khí, mệt mỏi, ăn ít",
                "người da mát sợ lạnh hoặc các khớp đau nhức",
                "tay chân thân mình mát lạnh",
                "chân tay mát, mệt mỏi"]:
        ok = o._kw_hit_clean(txt, COLD_KWS)
        n += _chk(f"HÀN: {txt[:44]}...", ok)
        f += (not ok)
    # ⚠ 'mát' nghĩa NHIỆT (bệnh nhân THÍCH mát) — không được tính là hàn
    for txt in ["khát thích uống nước mát",
                "miệng khô họng khát thích uống nước mát",
                "thích chườm mát, sốt cao",
                "người nóng, thích uống nước mát"]:
        ok = not o._kw_hit_clean(txt, COLD_KWS)
        n += _chk(f"KHÔNG phải hàn: {txt[:40]}...", ok)
        f += (not ok)
    return n, f


def test_reground_core():
    """Cổng grounding: core của bệnh KHÁC phải bị thay bằng thể thuộc bệnh trong cửa sổ."""
    print("\n== (3) Cổng grounding core ngoại lai ==")
    n = f = 0
    window = [{"benh_ly": "Ách nghịch",
               "hoi_chung_all": ["Tỳ thận dương hư", "Vị hàn", "Vị nhiệt", "Vị âm hư"]}]
    # 'Chính hư ứ kết' thuộc bệnh 'Tích tụ' -> KHÔNG grounded ở Ách nghịch
    ok = not F._core_grounded_in_window("Chính hư ứ kết", window)
    n += _chk("'Chính hư ứ kết' KHÔNG grounded ở Ách nghịch", ok)
    f += (not ok)
    new, reason = F._reground_core("Chính hư ứ kết",
                                   ["Chính hư ứ kết", "Tỳ thận dương hư", "Tỳ khí hư"], window)
    ok = new == "Tỳ thận dương hư" and reason
    n += _chk("re-rank sang 'Tỳ thận dương hư'", ok, repr(new))
    f += (not ok)
    # NO-OP khi core vốn grounded (kể cả biến thể tạng)
    for core in ("Tỳ thận dương hư", "Vị hàn"):
        got, r = F._reground_core(core, [core], window)
        ok = got == core and r is None
        n += _chk(f"NO-OP khi core đã grounded: {core}", ok)
        f += (not ok)
    # Không ứng viên nào grounded -> GIỮ core cũ (thà 'chưa có bài' còn hơn đổi bừa)
    got, r = F._reground_core("Chính hư ứ kết", ["Chính hư ứ kết", "Khí huyết hư"], window)
    ok = got == "Chính hư ứ kết"
    n += _chk("giữ core cũ khi không có ứng viên grounded", ok)
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_source_list_intact, test_cold_detection, test_reground_core):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
