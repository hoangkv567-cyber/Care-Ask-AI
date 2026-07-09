#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_interview.py — Kiểm module VẤN CHẨN CÓ CẤU TRÚC (src/interview.py).

Bảo đảm: (1) các lựa chọn thập vấn + nhân khẩu được quy về đúng cụm text chuẩn; (2) text sinh ra
thực sự kích hoạt đúng cổng hàn-nhiệt của pipeline (vd chọn 'sợ lạnh' -> cổng hàn nhận diện, chọn
'khát nước' -> cổng nhiệt nhận diện) — tức phần Vấn chẩn có cấu trúc gánh được việc phân định hàn
nhiệt mà lẽ ra cần mạch chẩn.

Chạy:  python scripts/test_interview.py    (Exit 0 nếu PASS — không cần Neo4j)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.interview import compose_interview_text, merge_symptoms
from src.fusion_pipeline import TCMFusionPipeline as F


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def run():
    ok = True

    print("── compose_interview_text: quy về từ khóa chuẩn ──")
    t = compose_interview_text(age=45, sex="nam", onset="man",
                               answers={"han_nhiet": "so_lanh", "dai_tien": "long", "ngu": "kho_ngu"})
    ok &= _check(f"gồm '45 tuổi': {t!r}", "45 tuổi" in t)
    ok &= _check("gồm 'nam giới'", "nam giới" in t)
    ok &= _check("gồm 'lâu ngày' (mạn tính -> kích hoạt Hư mạn)", "lâu ngày" in t)
    ok &= _check("gồm 'sợ lạnh'", "sợ lạnh" in t)
    ok &= _check("gồm 'đại tiện lỏng'", "đại tiện lỏng" in t)
    ok &= _check("gồm 'mất ngủ'", "mất ngủ" in t)

    print("── Bỏ qua lựa chọn 'bình thường' / mã lạ / tuổi sai ──")
    t2 = compose_interview_text(age=999, sex="", onset="",
                                answers={"han_nhiet": "binh_thuong", "khat": "xyz"})
    ok &= _check(f"rỗng khi toàn 'bình thường'/lạ: {t2!r}", t2 == "")

    print("── merge_symptoms: ghép sạch, không dấu phẩy mồ côi ──")
    ok &= _check("ghép có phẩy", merge_symptoms("ho khan", "sợ lạnh") == "ho khan, sợ lạnh")
    ok &= _check("free rỗng -> chỉ interview", merge_symptoms("", "sợ lạnh") == "sợ lạnh")
    ok &= _check("interview rỗng -> chỉ free", merge_symptoms("ho khan", "") == "ho khan")
    ok &= _check("free có phẩy cuối -> không mồ côi", merge_symptoms("ho khan,", "sợ lạnh") == "ho khan, sợ lạnh")

    print("── Text sinh ra kích hoạt ĐÚNG cổng hàn-nhiệt của pipeline ──")
    cold_text = compose_interview_text(answers={"han_nhiet": "so_lanh"}).lower()
    ok &= _check("'sợ lạnh' -> cổng HÀN bật", F._kw_hit_clean(cold_text, ["sợ lạnh", "úy hàn", "rét run"]))
    heat_text = compose_interview_text(answers={"khat": "khat_nuoc", "tieu_tien": "vang"}).lower()
    ok &= _check("'khát nước' -> cổng NHIỆT bật", F._kw_hit_clean(heat_text, ["khát nước", "sốt", "nước tiểu vàng"]))
    # 'miệng nhạt không khát' (hàn/thấp) KHÔNG được cổng nhiệt 'khát nước' bắt nhầm
    bland = compose_interview_text(answers={"khat": "mieng_nhat"}).lower()
    ok &= _check("'miệng nhạt không khát' KHÔNG bật cổng nhiệt 'khát nước'",
                 not F._kw_hit_clean(bland, ["khát nước"]))

    print("\n" + ("✅ TẤT CẢ PASS" if ok else "❌ CÓ CA FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
