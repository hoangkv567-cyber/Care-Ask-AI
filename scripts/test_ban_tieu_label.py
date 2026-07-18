#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_ban_tieu_label.py — Khóa nhãn BẢN/TIÊU ở Mục 5 (_branch_role_label).

VÌ SAO: chữ "Tiêu" bị dùng cho HAI nghĩa khác nhau.
  - Mục 4  : "Tiêu Thực"  = trục BỆNH CƠ (có tà thực hay không).
  - Mục 5  : "Tiêu"       = 标本, nghĩa "nhánh / bệnh kèm theo" — KHÔNG nói gì về tà thực.

CA THẬT (audit ổn định, bệnh "Âm hành đàm hạch"):
  Mục 4: "Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy."     <- ĐÚNG
  Mục 5: "Trị Bệnh Di tinh — *Tiêu – nhánh/kèm theo* (Hội chứng Thận hư không bền)"
'Thận hư không bền' là HƯ THUẦN, không có tà thực -> thầy thuốc đọc thấy hai mục CHỌI NHAU.

VÁ: chỉ in chữ "Tiêu" khi hội chứng kèm theo THẬT SỰ mang tà thực; hư thuần -> "Kèm theo".
Đây là sửa CHỮ, không đổi bài thuốc và không đổi việc dòng đó nằm ở branch_lines.

RÀNG BUỘC KHÔNG ĐƯỢC PHÁ (nhóm 3): _sync_muc4_with_muc5_tieu ĐỌC LẠI nhãn 'Tiêu' từ markdown để
phát hiện Mục 4 chối Tiêu Thực oan. Nếu đổi nhãu quá tay (bỏ luôn chữ 'Tiêu' cho ca tà thực THẬT)
thì cổng đồng bộ đó sẽ mù -> tái lỗi "Không có Tiêu Thực" trong khi đang kê bài trừ đàm/thanh nhiệt.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_ban_tieu_label.py   (exit != 0 nếu fail)
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

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402

# Hội chứng HƯ THUẦN — không tà thực -> KHÔNG được gọi là "Tiêu"
HU_THUAN = ["Thận hư không bền", "Khí huyết lưỡng hư", "Thận khí hư", "Tỳ thận dương hư",
            "Tâm tỳ lưỡng hư", "Phế khí hư", "Can huyết hư", "Thận âm hư"]
# Hội chứng CÓ tà thực -> PHẢI giữ nhãn "Tiêu"
CO_THUC = ["Đàm thấp", "Thấp nhiệt", "Huyết ứ", "Can khí uất kết", "Âm hư táo nhiệt",
           "Đàm hạch", "Hàn ngưng khí trệ", "Vị nhiệt"]


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_hu_thuan_khong_goi_tieu():
    print("== (1) Kèm theo HƯ THUẦN -> KHÔNG in chữ 'Tiêu' ==")
    n = f = 0
    for syn in HU_THUAN:
        lb = F._branch_role_label(syn)
        ok = "Tiêu" not in lb and "Kèm theo" in lb
        n += _chk(f"{syn}", ok, lb)
        f += (not ok)
    return n, f


def test_co_thuc_van_goi_tieu():
    print("\n== (2) Kèm theo CÓ TÀ THỰC -> vẫn là 'Tiêu' ==")
    n = f = 0
    for syn in CO_THUC:
        lb = F._branch_role_label(syn)
        ok = lb.startswith("Tiêu")
        n += _chk(f"{syn}", ok, lb)
        f += (not ok)
    # biến thể nhãn cho nhánh fallback thể-KB phải cùng quy tắc
    for syn, want_tieu in (("Đàm thấp", True), ("Thận hư không bền", False)):
        lb = F._branch_role_label(syn, kb_variant=True)
        ok = lb.startswith("Tiêu") == want_tieu
        n += _chk(f"[thể KB] {syn}", ok, lb)
        f += (not ok)
    return n, f


def test_sync_muc4_van_bat_duoc_tieu_thuc():
    """Đổi nhãn KHÔNG được làm mù cổng đồng bộ Mục 4↔5 ở ca tà thực THẬT."""
    print("\n== (3) Cổng đồng bộ Mục 4↔5 vẫn bắt được Tiêu Thực thật ==")
    o = F.__new__(F)
    n = f = 0

    def _md(role, syn):
        return ("**Thuộc chứng:** Lý - Hư (tổng cương: thiên Âm)\n\n"
                "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)\n"
                "- Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy.\n\n"
                "### 5. Bài thuốc\n"
                f"- Trị Bệnh **X** — *{role}* (Hội chứng {syn}) → Dùng bài **Y**\n")

    # (a) kèm theo CÓ tà thực, nhãn 'Tiêu' -> phải NÂNG Bát Cương + viết lại Mục 4
    md_a = _md("Tiêu – nhánh/kèm theo", "Đàm thấp")
    out_a = o._sync_muc4_with_muc5_tieu(md_a, "ho đờm nhiều, ngực tức")
    ok = out_a != md_a and "không có tiêu thực" not in out_a.lower()
    n += _chk("tà thực + nhãn 'Tiêu' -> ĐỒNG BỘ (Mục 4 hết chối)", ok)
    f += (not ok)
    ok = "Bản Hư Tiêu Thực" in out_a or "Thực" in out_a.split("###")[0]
    n += _chk("Bát Cương được nâng lên có Thực", ok)
    f += (not ok)

    # (b) kèm theo HƯ THUẦN, nhãn mới 'Kèm theo' -> KHÔNG được đụng vào Mục 4
    md_b = _md("Kèm theo – bệnh phối hợp", "Thận hư không bền")
    out_b = o._sync_muc4_with_muc5_tieu(md_b, "di tinh, mộng tinh")
    ok = out_b == md_b
    n += _chk("hư thuần + nhãn 'Kèm theo' -> NO-OP (Mục 4 giữ nguyên)", ok)
    f += (not ok)

    # (c) BẢO HIỂM: kể cả khi ai đó lỡ gắn 'Tiêu' cho hư thuần, cổng vẫn không nâng oan
    md_c = _md("Tiêu – nhánh/kèm theo", "Thận hư không bền")
    out_c = o._sync_muc4_with_muc5_tieu(md_c, "di tinh, mộng tinh")
    ok = out_c == md_c
    n += _chk("hư thuần dù mang nhãn 'Tiêu' -> vẫn NO-OP (không nâng oan)", ok)
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_hu_thuan_khong_goi_tieu, test_co_thuc_van_goi_tieu,
               test_sync_muc4_van_bat_duoc_tieu_thuc):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
