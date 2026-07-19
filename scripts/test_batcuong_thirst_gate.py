#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_batcuong_thirst_gate.py — Khóa cổng [KHÁT KHÔNG ĐỦ DỰNG NHIỆT] ở Bát Cương.

CA THẬT (nữ 34t): "tiểu tiện trong dài, tinh thần uể oải, nước tiểu trong, khát nước, tiểu đêm,
đau lưng, mỏi gối, rêu trắng mỏng, lưỡi bệu" — core 'Tỳ thận dương hư' (dương hư = nội hàn).
Chữ 'khát nước' ĐƠN ĐỘC nằm trong _strong_heat_kws -> dựng trục 'Nhiệt' -> xuống bộ đối chiếu
Hàn/Nhiệt thì _han_corr=False (tên core không chứa 'hàn', bệnh nhân không khai sợ lạnh) nên tag
'Hàn' ĐÚNG (lấy từ Neo4j) BỊ XÓA -> nhãn 'Lý - NHIỆT - Hư' TRÁI CỰC chính hội chứng vừa chốt,
rồi Mục 4 tự đẻ lý lẽ "hư nhiệt" để hợp thức hóa. ĐẢO CỰC = lớp lỗi nguy hiểm nhất của dự án.

Cổng là GIAO của 3 điều kiện, KHÔNG neo vào tên core (tránh vòng tự-hợp-thức):
    khát là dấu nhiệt mạnh DUY NHẤT  ∧  có dấu HÀN định tính  ∧  0 dấu khóa nhiệt/âm-hư.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_batcuong_thirst_gate.py   (exit != 0 nếu fail)
Không cần Neo4j/LLM.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402

SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "fusion_pipeline.py")
SRC = open(SRC_PATH, encoding="utf-8").read()

# Danh sách dấu nhiệt mạnh nằm INLINE trong thân hàm -> đọc lại từ NGUỒN THẬT thay vì chép tay,
# để test không lệch khi ai đó sửa một bên. Cắt theo tên biến + dấu ngoặc vuông cân bằng.
_m = re.search(r"_strong_heat_kws = \[(.*?)\]", SRC, re.S)
STRONG_HEAT = [s.strip().strip('"\'') for s in _m.group(1).replace("\n", " ").split(",") if s.strip()]
COLD_UNAMBIG = [k for k in F._THERMAL_COLD_STRONG if k != "tiểu trong"]

_o = F.__new__(F)


def _gate(sym):
    """Bản sao TẤT ĐỊNH của cổng — dùng CHÍNH các hằng/hàm thật. True = chặn dựng Nhiệt."""
    has_strong = _o._kw_hit_clean(sym, STRONG_HEAT)
    nonthirst = _o._kw_hit_clean(sym, [k for k in STRONG_HEAT if k != "khát nước"])
    return bool(has_strong and not nonthirst
                and _o._kw_hit_clean(sym, COLD_UNAMBIG)
                and not _o._kw_hit_clean(sym, list(F._THERMAL_NO_SWAP_SIGNS)))


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


CA_THAT = ("tiểu tiện trong dài, tinh thần uể oải, nước tiểu trong, khát nước, tiểu đêm, "
           "đau lưng, mỏi gối, rêu trắng mỏng, lưỡi bệu")


def test_gate_behaviour():
    """Chỉ ca hư-hàn mới bị chặn; MỌI ca nhiệt thật phải đi qua."""
    print("== (1) Hành vi cổng ==")
    n = f = 0
    cases = [
        # (nhãn, lời khai, có chặn Nhiệt không)
        ("CA THẬT nữ 34t (dương hư)", CA_THAT, True),
        ("khát + tay chân lạnh + sợ lạnh", "khát nước, tay chân lạnh, sợ lạnh, mệt mỏi", True),
        # ⚠ Dưới đây đều PHẢI đi qua — chặn nhầm = mù nhiệt
        ("Vị nhiệt thực (rêu vàng + lưỡi đỏ)", "khát nước thích uống lạnh, rêu vàng, lưỡi đỏ", False),
        ("Can dương thượng kháng + tiểu trong", "đau đầu căng, mắt đỏ, khát nước, nước tiểu trong", False),
        ("Nhiệt độc sang dương + tiểu trong", "mụn đỏ sưng đau, khát nước, nước tiểu trong", False),
        ("Thấp nhiệt lâm chứng (bẫy giới từ)",
         "đi tiểu trong ngày 10 lần, tiểu buốt rắt, tiểu vàng, khát nước", False),
        ("Âm hư hỏa vượng (dấu khóa âm-hư)",
         "khát nước, ngũ tâm phiền nhiệt, mồ hôi trộm, gò má đỏ, tay chân lạnh", False),
        ("Tiêu khát Thận âm hư", "khát nước uống nhiều, tiểu nhiều, lưỡi đỏ ít rêu", False),
        ("Dương hư phát nhiệt (có sốt)", "sốt về chiều, sợ lạnh, tay chân lạnh, khát nước", False),
        ("Dương hư kiêm thấp nhiệt", "tay chân lạnh, khát nước, tiểu vàng, rêu vàng nhớt", False),
        ("Không khát, không hàn -> cổng im", "đau đầu, chóng mặt, mất ngủ", False),
    ]
    for label, sym, exp in cases:
        ok = _gate(sym) == exp
        n += _chk(f"{'CHẶN' if exp else 'CHO QUA'}: {label}", ok)
        f += (not ok)
    return n, f


def test_gate_present_in_source():
    """Mutation-guard: gỡ cổng khỏi mã nguồn thì test này FAIL (bản sao ở trên vẫn xanh)."""
    print("\n== (2) Cổng còn trong mã nguồn ==")
    n = f = 0
    for label, ok in [
        ("có biến _thirst_only_on_cold", "_thirst_only_on_cold" in SRC),
        ("cổng được nối vào nhánh add('Nhiệt')",
         bool(re.search(r"and not _thirst_only_on_cold\s*\)\s*:\s*\n\s*all_bat_cuong\.add\(\"Nhiệt\"\)", SRC))),
        ("có cảnh báo log khi cổng chặn (hồi quy không im lặng)", "[KHÁT KHÔNG ĐỦ NHIỆT]" in SRC),
        ("dùng hằng _THERMAL_COLD_STRONG (không chép tay)", "_THERMAL_COLD_STRONG if k != " in SRC),
        ("dùng hằng _THERMAL_NO_SWAP_SIGNS", "_THERMAL_NO_SWAP_SIGNS)" in SRC),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    return n, f


def test_invariants():
    """Các chốt cứng — vi phạm là mở đường cho lớp lỗi cũ quay lại."""
    print("\n== (3) Bất biến ==")
    n = f = 0
    # 'khát nước' PHẢI còn là dấu nhiệt: nó là chứng chủ của Bạch hổ thang / Tiêu khát / Vị nhiệt,
    # và còn nuôi has_heat_pulse_indicator dùng ở 7 chỗ khác. Chỉ được hạ bậc TẠI NHÁNH add('Nhiệt').
    ok = "khát nước" in STRONG_HEAT
    n += _chk("'khát nước' VẪN nằm trong _strong_heat_kws (cấm gỡ)", ok)
    f += (not ok)
    # 'tiểu trong' trần bị loại khỏi tập dò: 'trong' làm giới từ ("đi tiểu trong ngày") dính oan.
    ok = "tiểu trong" not in COLD_UNAMBIG and "nước tiểu trong" in COLD_UNAMBIG
    n += _chk("'tiểu trong' trần bị loại, 'nước tiểu trong' được giữ", ok)
    f += (not ok)
    # Không keyword nào vừa là hàn vừa là nhiệt.
    clash = set(k.lower() for k in COLD_UNAMBIG) & set(k.lower() for k in STRONG_HEAT)
    n += _chk("không keyword nào vừa hàn vừa nhiệt", not clash, str(clash) if clash else "")
    f += (not not clash)
    # Cổng KHÔNG được neo vào tên hội chứng cốt lõi (vòng tự-hợp-thức: nhãn tự biện minh cho nhãn).
    _blk = SRC[SRC.find("_cold_strong_unambig ="):SRC.find("all_bat_cuong.add(\"Nhiệt\")")]
    ok = "final_primary" not in _blk
    n += _chk("cổng không đọc tên core (chống vòng tự-hợp-thức)", ok)
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_gate_behaviour, test_gate_present_in_source, test_invariants):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
