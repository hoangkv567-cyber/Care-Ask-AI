#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_cc_locator_gate.py — Khóa việc nạp CHỦ CHỨNG ĐỊNH VỊ GIẢI PHẪU (_cc_is_locator_mark).

VÌ SAO (audit ổn định 40 ca): bốn bệnh vùng MẮT và VÚ đều bị gọi thành 'Tý chứng / Kiên tý /
NHA THỐNG'. Tái hiện bằng lời khai THẬT (không phải input sinh máy):
    "đau mắt, sưng đỏ mắt"                                  -> Tý chứng, Kiên tý, Nha thống
    "vú sưng đau, có cục cứng ở vú, đau tăng khi sờ, sốt nhẹ" -> Nha thống, Phong thấp tâm, ...
Ca thứ hai là viêm tuyến vú kinh điển mà ra ĐAU RĂNG. Bộ khớp ăn theo từ chung 'sưng'/'đau'
(có mặt ở rất nhiều bệnh), còn DANH TỪ BỘ PHẬN thì bị bỏ qua vì cột chủ_chứng chỉ được nạp cho
dấu ĐẶC THÙ GIỚI.

CÁCH VÁ (và cách ĐÃ LOẠI): ngưỡng df THÔ nạp mọi chủ chứng ít gặp -> kéo theo rác ('tai nạn',
'đập đầu', 'kích động') và trả giá ĐO ĐƯỢC: gold 22/28 -> 21/28, M2 754 -> 746. Thu hẹp vào chủ
chứng chứa DANH TỪ BỘ PHẬN + df thấp thì hồi quy = 0 mà vẫn sửa được cả hai ca.

BẪY PHẢI CHẶN (nhóm 2) — chứa danh từ bộ phận nhưng KHÔNG định vị bệnh ở đó:
    'hoa mắt'(df=45) = chóng mặt        'ù tai'(df=48) = ù tai toàn thân
    'vàng mắt' = HOÀNG ĐẢN (bệnh GAN)   'mắt húp' = phù      'tai nạn' = 'tai' không phải cái tai
Nếu nạp nhầm nhóm này, ca chóng mặt/vàng da sẽ bị kéo sang bệnh MẮT — đúng lớp lỗi định vị mà
bản vá này sinh ra để chống.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_cc_locator_gate.py   (exit != 0 nếu fail)
Cần CSV (không cần Neo4j/LLM).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402

_o = None


def pipe():
    global _o
    if _o is None:
        _o = F.__new__(F)
        F._load_csv_data(_o)
        _o._build_chu_chung_index()
    return _o


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_nhan_dau_dinh_vi():
    print("== (1) NHẬN dấu định vị giải phẫu ==")
    n = f = 0
    for kw in ("đau mắt", "sưng mắt", "cộm mắt", "nhức mắt", "đau vú", "sưng vú",
               "cục ở vú", "áp xe vú", "nghẹt mũi", "chảy máu mũi", "sổ mũi"):
        ok = F._cc_is_locator_mark(kw)
        n += _chk(f"nhận '{kw}'", ok)
        f += (not ok)
    return n, f


def test_chan_ten_bo_phan_tran():
    """TÊN BỘ PHẬN TRẦN không được nạp — đây là hồi quy ĐÃ ĐO, không phải lo xa.

    Bản vá đầu chỉ đòi 'đa-từ + có danh từ bộ phận' nên nạp cả 'hậu môn' (df=7). Hậu quả: ca thật
    trong test_symptom_concepts (viêm đại tràng, khai 'ỉa chảy...') bật khỏi #1, nhường cho
    'Hậu môn lậu (rò hậu môn), Trĩ, Hậu môn nứt kẽ' — vì MỌI bệnh vùng ruột-hậu môn đều nhắc
    'hậu môn'. Tên bộ phận trần không phân biệt được bệnh TRONG vùng; lời than có định vị thì có.
    """
    print("\n== (1b) CHẶN tên bộ phận trần (thiếu từ triệu chứng) ==")
    n = f = 0
    for kw in ("hậu môn", "tinh hoàn", "tuyến vú", "cổ họng", "núm vú"):
        ok = not F._cc_is_locator_mark(kw)
        n += _chk(f"chặn tên bộ phận trần '{kw}'", ok)
        f += (not ok)
    return n, f


def test_chan_bay():
    print("\n== (2) CHẶN bẫy: có danh từ bộ phận nhưng KHÔNG định vị bệnh ==")
    n = f = 0
    for kw, why in (("hoa mắt", "= chóng mặt"), ("ù tai", "= ù tai, không phải bệnh tai"),
                    ("vàng mắt", "= hoàng đản, bệnh GAN"), ("mắt vàng", "= hoàng đản"),
                    ("mắt húp", "= phù"), ("tai nạn", "'tai' không phải cái tai"),
                    ("tai biến", "'tai' không phải cái tai"), ("quáng gà", "không đặc hiệu vùng"),
                    ("mờ mắt", "không đặc hiệu vùng")):
        ok = not F._cc_is_locator_mark(kw)
        n += _chk(f"chặn '{kw}' ({why})", ok)
        f += (not ok)
    # đơn-âm không bao giờ được nạp (khớp bừa)
    for kw in ("mắt", "vú", "tai"):
        ok = not F._cc_is_locator_mark(kw)
        n += _chk(f"chặn đơn-âm '{kw}'", ok)
        f += (not ok)
    return n, f


def test_admit_set_thuc_te():
    """Kiểm trên TẬP NẠP THẬT dựng từ KB — không chép tay, để test lệch code thì fail."""
    print("\n== (3) Tập nạp thật từ KB ==")
    o = pipe()
    # chu_chung phải KHÁC rỗng, nếu không hàm thoát sớm và chưa kịp dựng _cc_admit_set
    o._cc_match_ratio({"chu_chung": "đau mắt"}, [], "")   # ép dựng _cc_admit_set
    admit = o._cc_admit_set
    n = f = 0
    for kw in ("đau mắt", "đau vú", "sưng vú", "cộm mắt"):
        ok = kw in admit
        n += _chk(f"CÓ trong tập nạp: '{kw}'", ok)
        f += (not ok)
    # 'mắt đỏ' df=18: quá phổ biến -> phải nằm NGOÀI (nếu lọt, ca chóng mặt/sốt sẽ kéo sang bệnh mắt)
    for kw in ("mắt đỏ", "hoa mắt", "ù tai", "mệt mỏi", "chóng mặt"):
        ok = kw not in admit
        n += _chk(f"NGOÀI tập nạp: '{kw}'", ok)
        f += (not ok)
    return n, f


def test_cc_match_ratio():
    """Dòng KB vùng mắt: khai 'đau mắt' -> nạp; khai 'hoa mắt/chóng mặt' -> KHÔNG nạp."""
    print("\n== (4) _cc_match_ratio trên dòng KB vùng mắt ==")
    o = pipe()
    row = {"chu_chung": "mắt đỏ, đỏ mắt, mắt sưng, sưng mắt, đau mắt, mắt đau, nhức mắt, cộm mắt"}
    n = f = 0
    r1, hit1 = o._cc_match_ratio(row, ["đau mắt"], "đau mắt, sưng đỏ mắt")
    ok = r1 > 0 and hit1
    n += _chk("khai 'đau mắt' -> NẠP ứng viên vùng mắt", ok, f"ratio={r1:.2f} hit={hit1!r}")
    f += (not ok)
    r2, hit2 = o._cc_match_ratio(row, ["hoa mắt", "chóng mặt"], "hoa mắt, chóng mặt, ù tai, mệt mỏi")
    ok = r2 == 0
    n += _chk("khai 'hoa mắt/chóng mặt' -> KHÔNG nạp bệnh mắt", ok, f"ratio={r2:.2f} hit={hit2!r}")
    f += (not ok)
    r3, _ = o._cc_match_ratio(row, ["vàng da", "vàng mắt"], "vàng da, vàng mắt, nước tiểu vàng sẫm")
    ok = r3 == 0
    n += _chk("khai 'vàng mắt' (hoàng đản) -> KHÔNG nạp bệnh mắt", ok, f"ratio={r3:.2f}")
    f += (not ok)
    # dòng KB vùng VÚ
    row_vu = {"chu_chung": "vú, tuyến vú, đau vú, sưng vú, cục ở vú, núm vú, tắc sữa, áp xe vú"}
    r4, hit4 = o._cc_match_ratio(row_vu, ["đau vú"], "đau vú, sưng vú, có cục ở vú, sốt nhẹ")
    ok = r4 > 0
    n += _chk("khai 'đau vú/sưng vú' -> NẠP ứng viên vùng vú", ok, f"ratio={r4:.2f} hit={hit4!r}")
    f += (not ok)
    # GIỚI HẠN ĐÃ BIẾT (ghi lại để người sau không tưởng là bug mới): so khớp ở tầng NÀY là chuỗi
    # con, nên lời khai ĐẢO TRẬT TỰ ('vú sưng đau') KHÔNG khớp mục 'đau vú'. Ca thật vẫn ra đúng
    # bệnh vùng vú (đã kiểm app: 'vú sưng đau, có cục cứng ở vú, đau tăng khi sờ, sốt nhẹ' ->
    # 'Nhũ lao', trước bản vá là 'Nha thống') nhờ tầng chuẩn hoá/tách triệu chứng phía trên nạp
    # được biến thể. Chốt hành vi hiện tại để nếu sau này ai đó nới tầng này thì biết mình đang đổi gì.
    r5, _ = o._cc_match_ratio(row_vu, ["vú sưng đau"], "vú sưng đau, có cục cứng ở vú")
    ok = r5 == 0
    n += _chk("(giới hạn) đảo trật tự 'vú sưng đau' không khớp ở tầng chuỗi con", ok, f"ratio={r5:.2f}")
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_nhan_dau_dinh_vi, test_chan_ten_bo_phan_tran, test_chan_bay, test_admit_set_thuc_te, test_cc_match_ratio):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
