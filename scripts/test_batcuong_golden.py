#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_batcuong_golden.py — Harness hồi quy cho tầng BÁT CƯƠNG (chạy ĐẦU RA THẬT của pipeline).

VÌ SAO: suốt 9 vòng vá, eval_gold và eval_disease_matching BẤT BIẾN hoàn toàn (M1 76.3% / M4 4/4)
— tức 41 file test KHÔNG phủ tầng Bát Cương. Ba hồi quy THẬT đã xảy ra:
   #1 'khát nước' TRẦN dựng trục Nhiệt trên nền dương hư          -> ĐẢO CỰC
   #2 cổng lọc prose cắt cả mệnh đề, xóa mất biện luận hợp lệ     -> (tầng prose, harness này KHÔNG bắt)
   #3 mở lựa chọn 渴喜冷飲 phá tiền đề bản vá #1, tag 'Hàn' bị xóa -> ĐẢO CỰC
Cả ba chỉ được phát hiện khi người dùng dán kết quả ra. Harness này đóng lỗ đó cho #1 và #3.

KHÁC với các test hiện có: những test kia kiểm BẢN SAO của cổng (chép logic vào file test), nên
gỡ sạch cổng trong nguồn mà test vẫn xanh — đã đo đúng như vậy hai lần. Harness này chạy
run_diagnosis THẬT và so đầu ra.

CHẠY OFFLINE: Bát Cương chốt TRƯỚC lượt gọi LLM sinh prose -> chặn LLM bằng ĐÚNG MỘT seam thì
Mục 1/2 vẫn nguyên giá trị. Đo: ~4s/ca, 0 lượt gọi LLM (live: 15-83s/ca, 3 lượt).
Đã kiểm offline == live trên cả 3 ca hồi quy, và tất định qua 3 lần chạy.
CẦN Neo4j. KHÔNG cần mạng/API key.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_batcuong_golden.py [--limit N] [-v]
Sinh lại golden:  python scripts/build_batcuong_golden.py   (đọc tay trước khi commit!)

⚠ KHI TEST ĐỎ: phải xác định là HỒI QUY hay CẢI THIỆN CÓ CHỦ ĐÍCH *trước khi* sinh lại golden.
Sinh lại một cách phản xạ = đóng băng luôn cái sai, và harness thành vô dụng.
"""
import argparse
import io
import json
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
logging.disable(logging.WARNING)

from scripts.build_batcuong_golden import extract, make_pipeline, GOLDEN  # noqa: E402

# Ba trục Bát Cương — dùng để CHỈ RA trục nào lệch, giúp đọc lỗi nhanh (so khớp vẫn là chuỗi đầy đủ).
_AXES = {
    "Biểu/Lý": ("biểu - lý đồng bệnh", "biểu", "lý"),
    "Hàn/Nhiệt": ("hàn nhiệt thác tạp", "hàn", "nhiệt"),
    "Hư/Thực": ("bản hư tiêu thực", "hư", "thực"),
}


def axis_of(bc: str, axis: str) -> str:
    """Giá trị của MỘT trục trong nhãn. Duyệt nhãn ghép TRƯỚC (dài hơn) để không khớp nhầm."""
    parts = [p.strip() for p in (bc or "").split(" - ")]
    parts = [re.sub(r'\s*\(.*$', '', p) for p in parts]
    for val in _AXES[axis]:
        if any(p == val for p in parts):
            return val
    return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(GOLDEN):
        print("THIẾU data/batcuong_golden.json — chạy scripts/build_batcuong_golden.py trước.")
        sys.exit(2)
    data = json.load(io.open(GOLDEN, encoding="utf-8"))
    cases = data["cases"][:args.limit] if args.limit else data["cases"]

    p = make_pipeline()
    print(f"Chạy {len(cases)} ca golden OFFLINE (0 lượt gọi LLM)...\n")
    fails, errs = [], []
    for i, c in enumerate(cases, 1):
        try:
            r = p.run_diagnosis(user_symptoms=c["raw"])
            got = extract((r.get("diagnosis_result") or {}).get("answer") or "")
        except Exception as e:
            errs.append((c["id"], f"{type(e).__name__}: {e}"))
            continue
        exp = c["expect"]
        diff = {k: (exp[k], got[k]) for k in ("bat_cuong", "core", "benh_danh") if exp[k] != got[k]}
        if diff:
            fails.append((c, diff))
            print(f"  [ĐỎ ] {c['id']}")
            for k, (e, g) in diff.items():
                print(f"        {k}: kỳ vọng {e!r}")
                print(f"        {' ' * len(k)}  nhận   {g!r}")
                if k == "bat_cuong":
                    for ax in _AXES:
                        ve, vg = axis_of(e, ax), axis_of(g, ax)
                        if ve != vg:
                            print(f"        >>> TRỤC {ax}: {ve} -> {vg}")
            if c.get("note"):
                print(f"        ghi chú: {c['note']}")
        elif args.verbose:
            print(f"  [xanh] {c['id'][:36]:<36} {got['bat_cuong'][:40]}")

    print(f"\n{len(cases) - len(fails) - len(errs)} XANH, {len(fails)} ĐỎ, {len(errs)} LỖI / {len(cases)}")
    for cid, why in errs:
        print(f"  LỖI {cid}: {why}")
    if fails:
        print("\n⚠ Trước khi chạy build_batcuong_golden.py để sinh lại: xác định từng ca đỏ là HỒI QUY "
              "hay CẢI THIỆN CÓ CHỦ ĐÍCH. Sinh lại phản xạ = đóng băng cái sai.")
    sys.exit(1 if (fails or errs) else 0)


if __name__ == "__main__":
    main()
