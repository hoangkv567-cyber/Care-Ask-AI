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

def _grab_list(start_pat):
    """Bóc danh sách khai báo inline từ NGUỒN THẬT (không chép tay, không lệch khi ai đó sửa mã).

    BẮT BUỘC bóc chú thích TRƯỚC khi tách theo dấu phẩy: chú thích trong các danh sách này có
    chứa dấu phẩy và cả chuỗi trong ngoặc kép, parse thô sẽ nạp rác vào danh sách rồi cho ra kết
    quả sai (đã dính đúng bẫy này một lần khi đo tay).
    """
    lines = SRC.split("\n")
    i = next(i for i, l in enumerate(lines) if re.search(start_pat, l))
    buf = []
    for l in lines[i:]:
        buf.append(re.sub(r'#.*$', '', l))
        if "]" in buf[-1]:
            break
    s = " ".join(buf)
    s = s[s.index("[") + 1:s.rindex("]")]
    return [x.strip().strip('"\'') for x in s.split(",") if x.strip().strip('"\'')]


STRONG_HEAT = _grab_list(r'_strong_heat_kws = \[')     # gác việc dựng trục 'Nhiệt' ở Bát Cương
HEAT_KWS = _grab_list(r'^\s+heat_kws = \[')            # gác has_heat_pulse_indicator (7 nơi dùng)
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


def test_parity_all_heat_lists():
    """BẤT BIẾN ĐƠN ĐIỆU: lời khai ĐẶC HIỆU HƠN không bao giờ được nhận diện YẾU HƠN lời khai mơ hồ.

    Lỗ đã xảy ra thật: bản vá form đầu tiên chỉ nạp 'uống nước lạnh' vào _strong_heat_kws mà bỏ sót
    heat_kws (một trong NĂM danh sách dấu nhiệt) -> 'khát nước' (mơ hồ) = True nhưng
    'khát, thích uống nước lạnh' (渴喜冷飲, đặc hiệu nhất) = False. Nguyên tắc rút ra: NẠP TỪ KHÓA
    VÀO MÃ TRƯỚC, MỞ LỰA CHỌN FORM SAU.
    """
    print("\n== (5) Parity trên MỌI danh sách dấu nhiệt ==")
    o = pipe()
    vague = compose_interview_text(answers={"khat": "khat_nuoc"})
    specific = compose_interview_text(answers={"khat": "khat_lanh"})
    n = f = 0
    for lab, kws in [("_strong_heat_kws", STRONG_HEAT), ("heat_kws", HEAT_KWS)]:
        ok = o._kw_hit_clean(specific, kws) >= o._kw_hit_clean(vague, kws)
        n += _chk(f"{lab}: khai đặc hiệu >= khai mơ hồ", ok,
                  "" if ok else f"đặc hiệu={o._kw_hit_clean(specific, kws)} mơ hồ={o._kw_hit_clean(vague, kws)}")
        f += (not ok)
    # Không lựa chọn nào được vừa bật hàn vừa bật nhiệt
    for code in _INTERVIEW_MAP["khat"]:
        t = compose_interview_text(answers={"khat": code})
        if not t:
            continue
        ok = not (o._kw_hit_clean(t, STRONG_HEAT) and o._kw_hit_clean(t, COLD))
        n += _chk(f"{code} không vừa hàn vừa nhiệt", ok)
        f += (not ok)
    return n, f


def test_amhu_demote_gate():
    """'khát' TRẦN trong danh sách dấu nhiệt âm-hư khớp bằng `in` THÔ -> không hiểu phủ định.

    Hệ quả đo được: 'miệng nhạt KHÔNG khát' (lời khai PHỦ ĐỊNH khát, dấu HÀN kinh điển) bị đếm
    thành dấu NHIỆT, vô hiệu cổng hạ bậc âm-hư. Từ khi form cho khai 渴喜熱飲 thì
    'khát, thích uống nóng' (dấu HÀN) cũng dính.
    """
    print("\n== (6) Cổng hạ bậc âm-hư không bị 'khát' trần vô hiệu ==")
    n = f = 0
    for s, want in [
        ("miệng nhạt không khát, sợ lạnh, tay chân lạnh", False),
        ("khát, thích uống nóng, sợ lạnh", False),
        ("khát nhưng không muốn uống, tay chân lạnh", False),
        ("khát, thích uống nước lạnh, sốt", True),
        ("lưỡi bóng không rêu, lưỡi nhạt", True),      # dấu âm-hư thật phải SỐNG
    ]:
        got = any(k in s for k in F._AMHU_HEAT_SIGNS)
        ok = got == want
        n += _chk(f"dấu nhiệt âm-hư={got} cho {s[:42]!r}", ok)
        f += (not ok)
    ok = "khát" not in F._AMHU_HEAT_SIGNS and "khát" not in F._NHIET_HEAT_SIGNS
    n += _chk("'khát' TRẦN đã bị gỡ khỏi cả hai danh sách", ok)
    f += (not ok)
    ok = "khát" not in F._MIXED_HEAT_SIGNS
    n += _chk("'khát' TRẦN đã bị gỡ khỏi _MIXED_HEAT_SIGNS (banner)", ok)
    f += (not ok)
    return n, f


def test_prompt_rule():
    """Vá tầng NHẬP thôi chưa đủ — phải có luật cấm LLM bịa HÀNH VI UỐNG ở tầng SINH VĂN."""
    print("\n== (7) Luật 21 trong prompt Mục 3 ==")
    n = f = 0
    for label, ok in [
        ("có luật 21 chốt chặn tính chất khát", "21. CHỐT CHẶN TÍNH CHẤT KHÁT" in SRC),
        ("cấm đích danh các cụm LLM đã bịa (3 lần chạy)",
         all(x in SRC for x in ("không uống được", "uống không giải khát", "uống nhiều vẫn không đỡ"))),
        ("vẫn CHO PHÉP biện luận cơ chế dương bất khí hóa",
         "dương bất khí hóa" in SRC and "tân dịch bất thượng thừa" in SRC),
        ("luật 16(a) không còn dùng 'khát' trần làm dấu nhiệt",
         "lưỡi đỏ/ít rêu/khô, khát," not in SRC),
    ]:
        n += _chk(label, ok)
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
    for fn in (test_polarity, test_no_polarity_inversion, test_gate_composition,
               test_parity_all_heat_lists, test_amhu_demote_gate, test_prompt_rule, test_wiring):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
