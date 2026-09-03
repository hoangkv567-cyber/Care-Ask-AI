#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_muc4_seam.py — Khóa MỐI NỐI giữa hai tầng cùng ghi Mục 4.

CA THẬT (Viêm đại tràng, tái hiện tất định). Mục 4 TỰ MÂU THUẪN trong cùng một đoạn:
  "Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy. ⚠️ Lưu ý: ... lời khai lại CÓ DẤU THỰC
   TRỆ (ợ chua, bụng đầy, đau bụng). Cần cân nhắc THƯƠNG THỰC / THỰC TRỆ cấp trên nền hư
   (BẢN HƯ TIÊU THỰC) — nếu đúng thì PHẢI phối thêm pháp tiêu thực đạo trệ..."

CẢ HAI CÂU ĐỀU DO MÃ SINH, tuần tự và KHÔNG BIẾT NHAU:
  (1) _sync_tieu_thuc_with_bat_cuong ép câu "Không có Tiêu Thực..."
  (2) _annotate_acute_onset_caution NỐI ĐUÔI cảnh báo, không đọc câu đứng trước.

PHÁN ĐỊNH (đã thẩm định đối kháng): hệ ĐÚNG ở NHÃN, sai ở MỐI NỐI. TUYỆT ĐỐI KHÔNG sửa bằng cách
lật has_thuc — đo được trên KB thật: tập từ khóa trệ trung tiêu ngây thơ lật OAN 72/407 dòng HƯ
(riêng "ợ chua" lật 'Tào tạp | Tỳ vị hư hàn'), và _THUC_TEMPLATES lại chứa CƠ CHẾ HƯ ở 3 key nên
lật xong Mục 4 "Tiêu Thực" sẽ viết bằng cơ chế Hư -> đổi 1 mâu thuẫn lấy 3.

RÀNG BUỘC SỐNG CÒN: chuỗi "Không có Tiêu Thực" là KHÓA GIAO THỨC NGẦM — _muc4_denies_tieu_thuc đọc
nó ở 4 nơi, trong đó có MỘT nơi chạy SAU điểm vá. Chỉ được CHÈN quanh nó, CẤM thay hẳn.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_muc4_seam.py   (exit != 0 nếu fail)
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


MD = ("### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)\n"
      "- Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy.\n")
CASE = ("bệnh mới mắc, ngực bụng đầy tức, đại tiện lỏng, tiêu lỏng, buồn nôn, đau bụng, "
        "mệt mỏi, nôn mửa, ăn kém, ợ chua, rêu trắng mỏng, lưỡi bệu")


def test_no_self_contradiction():
    """Không được vừa phủ định vừa khẳng định Tiêu Thực trong cùng đoạn."""
    print("== (1) Mục 4 không tự mâu thuẫn ==")
    out = pipe()._annotate_acute_onset_caution(MD, "Tỳ khí hư", CASE)
    n = f = 0
    fired = out != MD
    n += _chk("cảnh báo VẪN bắn (không được im lặng bỏ qua dấu thực trệ)", fired)
    f += (not fired)
    # Mâu thuẫn = câu phủ định TRẦN đứng cạnh khẳng định có thực trệ.
    bare_denial = "Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy." in out
    asserts_thuc = "thực trệ" in out.lower()
    ok = not (bare_denial and asserts_thuc)
    n += _chk("KHÔNG còn phủ định TRẦN đứng cạnh khẳng định thực trệ", ok,
              "" if ok else "vẫn mâu thuẫn")
    f += (not ok)
    # Câu mệnh lệnh 'PHẢI phối pháp tiêu thực' quá mạnh: Mục 5 không có kênh thực thi.
    ok = "phải phối thêm pháp tiêu thực" not in out.lower()
    n += _chk("bỏ mệnh lệnh 'phải phối pháp tiêu thực' (Mục 5 không thực thi được)", ok)
    f += (not ok)
    # Phải nêu ĐIỀU KIỆN kiểm chứng thay vì phán chắc
    ok = "cự án" in out.lower() or "ấn" in out.lower()
    n += _chk("nêu dấu cần hỏi thêm để chốt (cự án / thiện án)", ok)
    f += (not ok)
    return n, f


def test_protocol_key_preserved():
    """KHÓA GIAO THỨC: 4 tầng đọc chuỗi 'Không có Tiêu Thực' — một tầng chạy SAU điểm vá."""
    print("\n== (2) Giữ khóa giao thức ==")
    out = pipe()._annotate_acute_onset_caution(MD, "Tỳ khí hư", CASE)
    n = f = 0
    ok = "Không có Tiêu Thực" in out
    n += _chk("chuỗi 'Không có Tiêu Thực' còn nguyên", ok)
    f += (not ok)
    m = re.search(r'### 4\..*?\n-?\s*(.*)', out, re.S)
    ok = bool(m) and F._muc4_denies_tieu_thuc(m.group(1).lower())
    n += _chk("_muc4_denies_tieu_thuc vẫn trả True (bảo vệ tầng chạy sau)", ok)
    f += (not ok)
    # Đếm số nơi đọc khóa này — nếu ai đó thêm nơi mới thì phải biết
    ok = SRC.count("_muc4_denies_tieu_thuc(") >= 4
    n += _chk(f"khóa được đọc ở {SRC.count('_muc4_denies_tieu_thuc(')} nơi (>=4)", ok)
    f += (not ok)
    return n, f


def test_thuc_kws_untouched():
    """CẤM lật trục Hư/Thực bằng cách nạp dấu trệ trung tiêu — đo được 72/407 dòng HƯ lật oan."""
    print("\n== (3) thuc_kws KHÔNG được nạp dấu trung tiêu ==")
    m = re.search(r'thuc_kws = \[(.*?)\]', SRC, re.S)
    body = re.sub(r'#.*$', '', m.group(1), flags=re.M) if m else ""
    n = f = 0
    for kw in ["ợ chua", "bụng đầy", "đau bụng", "buồn nôn", "nôn mửa", "đầy tức", "ăn kém"]:
        ok = f'"{kw}"' not in body
        n += _chk(f"thuc_kws KHÔNG chứa {kw!r}", ok)
        f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_no_self_contradiction, test_protocol_key_preserved, test_thuc_kws_untouched):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
