#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_thac_tap_axis.py — Khóa ba lỗi lộ ra khi bệnh nhân khai 渴喜冷飲 trên nền dương hư.

CA THẬT (nữ 34t, chọn "Khát, thích uống nước LẠNH"): lời khai có SÁU dấu hàn
(tiểu tiện trong dài, nước tiểu trong, tiểu đêm, rêu trắng mỏng, lưỡi bệu, core 'Thận dương hư')
đối MỘT dấu nhiệt. Hệ ra "Lý - NHIỆT - Hư" — đảo cực — kèm Mục 4 bịa "HƯ NHIỆT".

BA LỖI, ĐỘC LẬP NHAU:

(1) _han_corr QUÁ HẸP (6 từ khóa) — bỏ sót toàn bộ dấu hàn TIẾT NIỆU/THIỆT CHẨN. Nhánh đối chiếu
    Hàn/Nhiệt thấy chỉ có căn cứ Nhiệt -> XÓA SẠCH 'Hàn'.
    Vì sao trước đây không lộ: cổng _thirst_only_on_cold chặn 'khát nước' TRẦN dựng Nhiệt, nên
    nhánh này KHÔNG BAO GIỜ chạy. Khi form cho khai một dấu nhiệt THẬT thì nó chạy và lỗ lộ ra.

(2) _annotate_deficiency_heat kích hoạt cho core DƯƠNG hư, bịa y lý "âm huyết/chính khí hư tổn
    không chế ước được dương" để hợp thức hóa nhãn Nhiệt sai. Hư nhiệt kinh điển sinh từ ÂM hư;
    dương hư mà phát nhiệt chỉ có ở đới dương/cách dương (chứng NGUY KỊCH, cơ chế khác hẳn).

(3) NHẦM TRỤC: 'thác tạp' chỉ xuất hiện ở nhãn 'Hàn Nhiệt Thác Tạp' (trục HÀN/NHIỆT) nhưng bị
    tính là bằng chứng THỰC -> Mục 4 khẳng định "còn tồn tại yếu tố Thực (tà khí/đàm thấp ứ trệ)"
    trong khi Bát Cương KHÔNG hề có Thực. Trục hư/thực dùng nhãn riêng 'Bản Hư Tiêu Thực'.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_thac_tap_axis.py   (exit != 0 nếu fail)
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

SRC = open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
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


CA = ("thích uống nước lạnh, tiểu tiện trong dài, tinh thần uể oải, nước tiểu trong, "
      "rêu trắng mỏng, khát nước, lưỡi bệu, tiểu đêm, đau lưng, mỏi gối")
# Bản sao TẤT ĐỊNH của _han_corr — dùng chính hằng thật.
HAN_KWS = (["sợ lạnh", "úy hàn", "rét run", "tay chân lạnh", "chân tay lạnh", "lưng lạnh"]
           + [k for k in F._THERMAL_COLD_STRONG if k != "tiểu trong"])


def test_han_corr():
    print("== (1) Căn cứ HÀN phải nhận dấu tiết niệu/thiệt chẩn ==")
    o = pipe()
    n = f = 0
    ok = o._kw_hit_clean(CA, HAN_KWS)
    n += _chk("ca thật (6 dấu hàn) -> CÓ căn cứ Hàn", ok)
    f += (not ok)
    for s, want in [
        ("nước tiểu trong, tiểu đêm", True),
        ("tiểu tiện trong dài", True),
        ("khát, thích uống nóng", True),
        # BẪY GIỚI TỪ: 'trong' làm giới từ, KHÔNG phải nước tiểu trong
        ("đi tiểu trong ngày 10 lần, tiểu buốt rắt, tiểu vàng", False),
        ("sốt cao, khát nhiều, rêu vàng", False),
    ]:
        got = o._kw_hit_clean(s, HAN_KWS)
        ok = got == want
        n += _chk(f"căn cứ Hàn={got} cho {s[:40]!r}", ok)
        f += (not ok)
    # Neo vào ĐÚNG khối _han_corr: hằng _THERMAL_COLD_STRONG còn được dùng ở cổng khát-đơn-độc
    # nữa, nên tìm chuỗi đó trên TOÀN FILE sẽ vẫn xanh dù _han_corr bị thu hẹp lại (test yếu).
    _m = re.search(r'_han_corr = .*?(?=\n\s*_nhiet_corr)', SRC, re.S)
    ok = bool(_m) and "_THERMAL_COLD_STRONG" in _m.group(0)
    n += _chk("_han_corr dùng hằng _THERMAL_COLD_STRONG (không chép tay)", ok)
    f += (not ok)
    ok = bool(_m) and "tiểu trong" in _m.group(0)
    n += _chk("_han_corr loại 'tiểu trong' trần (bẫy giới từ)", ok)
    f += (not ok)
    # CẤM neo theo TÊN hội chứng: _YANG_DEFICIENCY_RE khớp cả 'Âm dương lưỡng hư'
    ok = bool(F._YANG_DEFICIENCY_RE.search("âm dương lưỡng hư"))
    n += _chk("(chứng minh vì sao cấm) _YANG_DEFICIENCY_RE KHỚP 'Âm dương lưỡng hư'", ok)
    f += (not ok)
    return n, f


def test_muc4_polarity_gate():
    """Chú giải hư nhiệt CHỈ dành cho thể có phần ÂM hư."""
    print("\n== (2) Cổng cực ở chú giải hư nhiệt ==")
    o = pipe()
    md = "### 4. Phân tích Cơ chế Ngọn\n- Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy.\n"
    n = f = 0
    for core, want in [
        ("Thận dương hư", False), ("Tỳ thận dương hư", False),
        ("Tâm dương hư", False),              # \bâm\b KHÔNG dính 'âm' trong 'tâm'
        ("Tâm thận dương hư", False), ("Dương khí hư", False), ("Hư hàn", False),
        ("Can thận âm hư", True), ("Thận âm hư hỏa vượng", True),
        ("Âm dương lưỡng hư", True),          # có phần âm -> hư nhiệt hợp lý
        ("Khí âm hư", True), ("Phế âm hư", True),
    ]:
        got = o._annotate_deficiency_heat(md, "Lý - Nhiệt - Hư", core, CA) != md
        ok = got == want
        n += _chk(f"{core:<22} chú giải hư nhiệt={got}", ok)
        f += (not ok)
    return n, f


def test_thac_tap_axis():
    """'Hàn Nhiệt Thác Tạp' thuộc trục HÀN/NHIỆT — không được suy ra THỰC."""
    print("\n== (3) Không nhầm trục hàn-nhiệt sang hư-thực ==")
    n = f = 0
    labels = set(re.findall(r'all_bat_cuong\.add\("([^"]*)"\)', SRC))
    tt = {l for l in labels if "thác tạp" in l.lower()}
    ok = tt == {"Hàn Nhiệt Thác Tạp"}
    n += _chk("chỉ MỘT nhãn chứa 'thác tạp', thuộc trục hàn/nhiệt", ok, str(tt))
    f += (not ok)
    ok = "Bản Hư Tiêu Thực" in labels
    n += _chk("trục hư/thực có nhãn RIÊNG 'Bản Hư Tiêu Thực'", ok)
    f += (not ok)
    # Mutation-guard: không chỗ nào được suy Thực từ 'thác tạp'
    # BẮT BUỘC bóc chú thích CUỐI DÒNG trước khi khớp: chính câu giải thích "KHÔNG tính 'thác tạp'"
    # nằm ngay sau dấu # trên cùng dòng gán, parse thô sẽ báo lỗi giả (đã dính bẫy này 3 lần).
    bad = [ln.strip() for ln in SRC.split("\n")
           if re.search(r'(?:_has_thuc|hint_has_thuc)\s*=.*thác tạp', re.sub(r'#.*$', '', ln))]
    ok = not bad
    n += _chk("KHÔNG suy 'Thực' từ 'thác tạp'", ok, str(bad[:1]))
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_han_corr, test_muc4_polarity_gate, test_thac_tap_axis):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
