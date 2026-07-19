#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_thirst_polarity.py — Khóa TÍNH CHẤT KHÁT làm dấu phân cực hàn/nhiệt.

VÌ SAO: form cũ chỉ có "Khát, thích uống" -> quy về "khát nước" (đánh dấu NHIỆT). Dữ liệu không đủ
phân cực nên ở ca thật (nữ 34t DƯƠNG HƯ) LLM TỰ BỊA tính chất uống để khớp chẩn đoán — ba lần chạy
ra ba kiểu khác nhau ("không thể uống nhiều nước", "uống nước nhiều vẫn không giải được", "khát mà
không uống được đủ"), tất cả đều TRÁI lời khai "thích uống". Bịa dấu chẩn đoán, không phải bịa văn.

Y LÝ (đây là lý do bài test này tồn tại):
    渴喜冷飲  khát thích uống LẠNH      -> NHIỆT
    渴喜熱飲  khát thích uống ẤM/NÓNG   -> HÀN
    渴不欲飲  khát mà KHÔNG muốn uống   -> thấp/đàm/ứ huyết/dương hư — KHÔNG phải nhiệt

BẪY ĐÃ ĐO VÀ PHẢI KHÓA: chuỗi mới KHÔNG chứa chuỗi con 'khát nước', nên nếu chỉ sửa form mà không
dạy các cổng từ khóa mới thì bệnh nhân NHIỆT thật khai đúng "thích uống lạnh" sẽ MẤT SẠCH dấu nhiệt
— tức sửa form một mình làm hệ TỆ HƠN trước.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_thirst_polarity.py   (exit != 0 nếu fail)
Không cần Neo4j/LLM.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import logging  # noqa: E402
logging.disable(logging.INFO)

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402
from src.interview import compose_interview_text, _INTERVIEW_MAP  # noqa: E402

SRC = open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
HTML = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()

# Đọc danh sách dấu nhiệt mạnh từ NGUỒN THẬT, không chép tay.
STRONG_HEAT = [s.strip().strip('"\'')
               for s in re.search(r'_strong_heat_kws = \[(.*?)\]', SRC, re.S).group(1)
               .replace("\n", " ").split(",")
               if s.strip() and "kws" not in s]
COLD = [k for k in F._THERMAL_COLD_STRONG if k != "tiểu trong"]

_o = None


def pipe():
    global _o
    if _o is None:
        _o = F.__new__(F)
        F._load_csv_data(_o)
    return _o


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_polarity():
    """Mỗi lựa chọn form phải đi xuyên 3 tầng (form -> interview -> cổng) ra ĐÚNG cực."""
    print("== (1) Phân cực từng lựa chọn ==")
    o = pipe()
    n = f = 0
    for code, want_heat, want_cold in [
        ("khat_lanh", True, False),         # 渴喜冷飲 -> nhiệt
        ("khat_am", False, True),           # 渴喜熱飲 -> hàn
        ("khat_khong_uong", False, False),  # 渴不欲飲 -> KHÔNG phải nhiệt (cốt lõi của ca thật)
        ("khat_nuoc", True, False),         # mơ hồ, giữ hành vi cũ
        ("mieng_nhat", False, False),
    ]:
        txt = compose_interview_text(answers={"khat": code})
        got_h = o._kw_hit_clean(txt, STRONG_HEAT)
        got_c = o._kw_hit_clean(txt, COLD)
        ok = (got_h == want_heat) and (got_c == want_cold)
        n += _chk(f"{code:<16} -> {txt!r:<34} nhiệt={got_h} hàn={got_c}", ok)
        f += (not ok)
    return n, f


def test_no_polarity_inversion():
    """Lớp lỗi nguy hiểm nhất dự án: dấu HÀN không bao giờ được tính là NHIỆT và ngược lại."""
    print("\n== (2) Không đảo cực ==")
    o = pipe()
    n = f = 0
    ok = not o._kw_hit_clean("khát, thích uống nóng", STRONG_HEAT)
    n += _chk("'thích uống nóng' (HÀN) KHÔNG bị tính là nhiệt", ok)
    f += (not ok)
    ok = not o._kw_hit_clean("khát, thích uống nước lạnh", COLD)
    n += _chk("'thích uống nước lạnh' (NHIỆT) KHÔNG bị tính là hàn", ok)
    f += (not ok)
    # Phủ định phải được tôn trọng — nếu không, 'không thích uống lạnh' thành dấu nhiệt giả.
    for s, kws, lab in [("không thích uống nước lạnh", STRONG_HEAT, "nhiệt"),
                        ("không thích uống nóng", COLD, "hàn")]:
        ok = not o._kw_hit_clean(s, kws)
        n += _chk(f"phủ định: {s!r} KHÔNG là dấu {lab}", ok)
        f += (not ok)
    return n, f


def test_gate_composition():
    """Cổng _thirst_only_on_cold (vòng 1) phải chặn khát MƠ HỒ nhưng KHÔNG chặn nhiệt ĐẶC HIỆU."""
    print("\n== (3) Ghép với cổng khát-đơn-độc ==")
    o = pipe()

    def gate(s):
        hs = o._kw_hit_clean(s, STRONG_HEAT)
        hn = o._kw_hit_clean(s, [k for k in STRONG_HEAT if k != "khát nước"])
        return bool(hs and not hn and o._kw_hit_clean(s, COLD)
                    and not o._kw_hit_clean(s, list(F._THERMAL_NO_SWAP_SIGNS)))

    base = "tiểu tiện trong dài, nước tiểu trong, tiểu đêm, rêu trắng mỏng, lưỡi bệu"
    n = f = 0
    ok = gate(base + ", khát nước")
    n += _chk("khát MƠ HỒ trên nền hàn -> vẫn CHẶN trục Nhiệt", ok)
    f += (not ok)
    ok = not gate(base + ", khát, thích uống nước lạnh")
    n += _chk("khát thích uống LẠNH -> KHÔNG chặn (nhiệt đặc hiệu, có thể là hàn-nhiệt thác tạp)", ok)
    f += (not ok)
    return n, f


def test_wiring():
    """Ba tầng phải khớp nhau: thiếu tầng nào là mất dấu im lặng."""
    print("\n== (4) Nối đủ ba tầng ==")
    n = f = 0
    codes = set(_INTERVIEW_MAP["khat"].keys())
    for label, ok in [
        ("interview.py có đủ 3 mã mới",
         {"khat_lanh", "khat_am", "khat_khong_uong"} <= codes),
        ("giữ tương thích ngược mã cũ", {"khat_nuoc", "mieng_nhat"} <= codes),
        ("index.html có đủ 3 lựa chọn mới",
         all(f'value="{c}"' in HTML for c in ("khat_lanh", "khat_am", "khat_khong_uong"))),
        # Không có cụm này thì lựa chọn 'uống lạnh' mất sạch dấu nhiệt — đúng cái bẫy đã đo.
        ("_strong_heat_kws nhận 'uống nước lạnh'", '"uống nước lạnh"' in SRC),
        ("_THERMAL_COLD_STRONG nhận 'thích uống nóng'",
         "thích uống nóng" in " ".join(F._THERMAL_COLD_STRONG)),
        # cold_kws gác 5 quyết định — thêm từ vào đó bật 'Hàn Nhiệt Thác Tạp' giả (vòng 2 đã cấm).
        ("KHÔNG nhét cụm hàn vào cold_kws",
         not re.search(r'cold_kws = \[[^\]]*thích uống nóng', SRC, re.S)),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    # Mọi mã trong form phải có ánh xạ, và ngược lại — lệch là rơi lựa chọn im lặng.
    html_codes = set(re.findall(r'<option value="(khat_[a-z_]*)"', HTML))
    ok = html_codes <= codes
    n += _chk("mọi lựa chọn HTML đều có ánh xạ", ok, "" if ok else str(html_codes - codes))
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_polarity, test_no_polarity_inversion, test_gate_composition, test_wiring):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
