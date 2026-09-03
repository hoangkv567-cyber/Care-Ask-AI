#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_herb_tail_notes.py — Test _strip_herb_tail_notes + _dedupe_herbs:
cắt ghi chú đuôi (bài thay thế / gia giảm) khỏi vị_thuốc mà KHÔNG phá ngoặc bào chế / alias / vị base.
Ca thử lấy từ CSV thật + các bẫy do agent phân loại nêu.  python scripts/test_herb_tail_notes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def herbs(s):
    return [x.strip() for x in F._dedupe_herbs(s).split(",") if x.strip()]


def main():
    cases = []
    def chk(desc, cond):
        cases.append((desc, cond))

    # (c) BÀI THAY THẾ cuối field — ca thật Đái tháo nhạt: cắt blob, giữ 12 vị base
    dtn = ("Nhân sâm, Sinh địa, Thiên hoa phấn, Ngọc trúc, Thiên môn, Mạch môn, Địa cốt bì, Đan sâm, "
           "Đan bì, Thạch cao, Tri mẫu, Sinh Cam thảo. (Hoặc dùng Mạch Môn Đông Thang: Hoàng cầm, "
           "Mạch môn, Cát căn, Tri mẫu, Trúc diệp, Ô mai, Lô căn, Thiên hoa phấn, Sa sâm)")
    h = herbs(dtn)
    chk("Đái tháo nhạt: cắt bài thay thế (không 'Hoàng cầm'/'Sa sâm'/'(Hoặc'/'gia')",
        "Hoàng cầm" not in " ".join(h) and "Sa sâm" not in " ".join(h) and "hoặc" not in " ".join(h).lower())
    chk("Đái tháo nhạt: giữ vị base (Nhân sâm, Sinh địa, Sinh Cam thảo)",
        any("Nhân sâm" in x for x in h) and any("Sinh địa" in x for x in h) and any("Cam thảo" in x for x in h))
    chk("Đái tháo nhạt: ~12 vị base (không 16-17)", 10 <= len(h) <= 13)

    # (d) GIA GIẢM điều kiện — ca thật Ách nghịch: cắt, giữ tới 'Trích thảo'
    ach = ("Đinh hương, Thị đế, Đảng sâm, Sinh khương, Trích thảo. (Nếu có đờm trệ, ợ hơi có mùi, "
           "gia: Trần bì, Hậu phác, Chỉ thực)")
    h2 = herbs(ach)
    chk("Ách nghịch: cắt gia giảm (không 'Trần bì'/'gia'/'Nếu'/'đờm trệ')",
        all(t not in " ".join(h2).lower() for t in ("trần bì", "gia:", "nếu", "đờm trệ", "ợ hơi")))
    chk("Ách nghịch: giữ vị base tới Trích thảo", any("Trích thảo" in x for x in h2) and any("Đinh hương" in x for x in h2))

    # (a) BÀO CHẾ — GIỮ nguyên (KHÔNG cắt)
    baoche = "Cam thảo (chích), Tri mẫu (tẩm muối), Biển đậu (sao), Bạch truật (thổ sao)"
    h3 = herbs(baoche)
    chk("Bào chế: giữ đủ 4 vị", len(h3) == 4)

    # (b) THAY VỊ inline — rút về vị chính, GIỮ vị sau ngoặc
    b1 = herbs("Nhân sâm (hoặc Đảng sâm), A giao, Mạch môn")
    chk("Thay-vị inline: 'Nhân sâm (hoặc Đảng sâm), A giao, Mạch môn' -> giữ A giao + Mạch môn",
        any("A giao" in x for x in b1) and any("Mạch môn" in x for x in b1) and "đảng sâm" not in " ".join(b1).lower())
    b2 = herbs("Sinh khương (hoặc Đảng sâm, Thái tử sâm), Cam thảo")
    chk("Thay-vị inline có dấu phẩy: giữ 'Cam thảo', bỏ '(hoặc Đảng sâm, Thái tử sâm)'",
        any("Cam thảo" in x for x in b2) and "thái tử" not in " ".join(b2).lower())

    # (c) '(kết hợp bột ...)' đứng riêng sau dấu phẩy -> bỏ
    kh = herbs("Hoàng cầm, Cam thảo, Cát cánh, (kết hợp bột Tiêu dao tán)")
    chk("(kết hợp bột X): bỏ note, giữ 3 vị", len(kh) == 3 and "kết hợp" not in " ".join(kh).lower())

    # TRAP 1: bào chế DÍNH LIỀN tên vị (không ngoặc) -> KHÔNG cắt
    t1 = herbs("Hoàng bá sao đen, Sinh khương, Can khương, Đại hoàng thán")
    chk("Trap bào chế dính liền: giữ đủ 4 token", len(t1) == 4)

    # TRAP 2: '(gia thêm)' / '(có thể)' đứng riêng (không list) -> KHÔNG cắt cả cụm
    t2 = herbs("Sa tiền (gia thêm), Cam thảo, Thạch cao (có thể)")
    chk("Trap '(gia thêm)'/'(có thể)': giữ đủ 3 vị", len(t2) == 3)

    # TRAP 3: alias '(Đương quy)' -> giữ
    t3 = herbs("Qui xuyên (Đương quy), Hoài sơn (Sơn dược), Đơn bì (Mẫu đơn bì)")
    chk("Trap alias: giữ đủ 3 vị", len(t3) == 3)

    ok = 0
    for d, c in cases:
        print(f"  [{'PASS' if c else 'FAIL'}] {d}")
        ok += bool(c)
    print(f"\n{ok}/{len(cases)} PASS")
    if ok < len(cases):
        print("\n[debug] Đái tháo nhạt ->", herbs(dtn))
        print("[debug] Ách nghịch ->", herbs(ach))
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
