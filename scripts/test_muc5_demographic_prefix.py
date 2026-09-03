#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_muc5_demographic_prefix.py — Khóa việc tách tiền tố NHÂN KHẨU HỌC ở Mục 5.

CA THẬT (nữ 34t, dương hư di niệu): than phiền trọng tâm là tiết niệu (tiểu trong dài, tiểu đêm)
+ đau lưng mỏi gối. KB CÓ dòng đúng 'Di niệu × Người lớn - Dương hư -> Bát vị hoàn (Kim quỹ thận
khí hoàn) gia giảm' — bài kinh điển. Nhưng [FALLBACK THỂ TỔNG QUÁT] tokenize nhãn THÔ nên hai chữ
'người'/'lớn' tự loại nhãn khỏi phép thử tập-con -> thể đúng rớt IM LẶNG -> hệ kê bài viêm cột
sống của bệnh KHÁC (có Thục phụ tử + Ma hoàng) cho bệnh nhân nhẹ.

BẪY CHÍNH của bản vá: 18/22 nhãn có gạch trong KB là nhãn GHÉP HAI HỘI CHỨNG
('Âm hoàng - Hàn thấp trở át' vs 'Dương hoàng - Nhiệt trọng ư thấp'). Tách mù bằng split(' - ')
sẽ ĐẢO CỰC hàn/nhiệt — nên regex phải neo ĐẦU CHUỖI vào từ vựng ĐÓNG.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_muc5_demographic_prefix.py
Không cần Neo4j/LLM.
"""
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "Medicine_clean.csv")


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def _kb():
    rows = list(csv.DictReader(io.open(CSV_PATH, encoding="utf-8-sig")))
    cols = list(rows[0].keys())
    return rows, cols[0], cols[1]      # rows, cột bệnh, cột hội chứng


def test_strip():
    print("== (1) Tách đúng 4 nhãn nhân khẩu học ==")
    n = f = 0
    for src, want in [
        ("Người lớn - Dương hư", "Dương hư"),
        ("Người lớn - Khí hư", "Khí hư"),
        ("Trẻ em - Do hàn", "Do hàn"),
        ("Trẻ em - Do nhiệt", "Do nhiệt"),
        ("NGƯỜI LỚN - Dương hư", "Dương hư"),        # không phân biệt hoa/thường
        ("Người  lớn  -  Dương hư", "Dương hư"),     # khoảng trắng thừa
        ("Trẻ em – Do hàn", "Do hàn"),               # gạch en-dash
    ]:
        got = F._strip_demographic_prefix(src)
        ok = got == want
        n += _chk(f"{src!r} -> {want!r}", ok, "" if ok else f"nhận {got!r}")
        f += (not ok)
    return n, f


def test_no_overreach():
    """Nhãn GHÉP HAI HỘI CHỨNG phải còn NGUYÊN VẸN — tách chúng là đảo cực."""
    print("\n== (2) Không tách quá tay (chống đảo cực) ==")
    n = f = 0
    for label in [
        "Âm hoàng - Hàn thấp trở át",        # HÀN
        "Dương hoàng - Nhiệt trọng ư thấp",  # NHIỆT — cặp đối cực với dòng trên
        "Cấp tính - Thấp nhiệt",             # giai đoạn cấp/mãn quyết định cực điều trị
        "Can huyết hư - Can thận âm hư",
        "Thể nhẹ - Phong hàn",
        "Nam giới - Thận hư",                # ngoài từ vựng đóng -> KHÔNG tách
        "Người lớn tuổi - Can thận hư",      # 'người lớn tuổi' != 'người lớn'
    ]:
        got = F._strip_demographic_prefix(label)
        ok = got == label
        n += _chk(f"giữ nguyên: {label}", ok, "" if ok else f"-> bị đổi thành {got!r}")
        f += (not ok)
    # Tiền tố ở GIỮA chuỗi không được đụng (regex phải neo đầu chuỗi)
    ok = F._strip_demographic_prefix("Thấp nhiệt kèm Người lớn - Dương hư") == \
        "Thấp nhiệt kèm Người lớn - Dương hư"
    n += _chk("chỉ neo ĐẦU chuỗi, không tách ở giữa", ok)
    f += (not ok)
    # Tách xong không được tụt về token đơn vô nghĩa
    ok = F._strip_demographic_prefix("Trẻ em - Do hàn") == "Do hàn"
    n += _chk("'Trẻ em - Do hàn' -> 'Do hàn' (giữ 'Do', không tụt còn 'hàn')", ok)
    f += (not ok)
    return n, f


def test_kb_blast_radius():
    """Bán kính ĐÓNG BĂNG trên KB thật: nới regex ngoài ý định thì test này FAIL."""
    print("\n== (3) Bán kính trên KB thật ==")
    n = f = 0
    rows, cB, cH = _kb()
    labels = {(r[cH] or "").strip() for r in rows if (r[cH] or "").strip()}
    dashed = {l for l in labels if " - " in l or " – " in l}
    stripped = {l for l in dashed if F._strip_demographic_prefix(l) != l}
    want = {"Người lớn - Dương hư", "Người lớn - Khí hư", "Trẻ em - Do hàn", "Trẻ em - Do nhiệt"}
    ok = stripped == want
    n += _chk("tập nhãn bị tách ĐÚNG BẰNG 4 nhãn dự kiến", ok,
              "" if ok else f"thừa/thiếu: {stripped ^ want}")
    f += (not ok)
    ok = len(dashed - stripped) == len(dashed) - 4
    n += _chk(f"{len(dashed) - 4} nhãn có gạch còn lại KHÔNG bị đụng", ok)
    f += (not ok)
    dis = {(r[cB] or "").strip() for r in rows
           if F._strip_demographic_prefix((r[cH] or "").strip()) != (r[cH] or "").strip()}
    ok = dis == {"Di niệu"}
    n += _chk("bệnh bị ảnh hưởng CHỈ gồm {'Di niệu'}", ok, "" if ok else str(dis))
    f += (not ok)
    return n, f


def test_wiring():
    """Mutation-guard: phải sửa ĐỦ 3 điểm. Thiếu 1 là lỗi chắc chắn xảy ra."""
    print("\n== (4) Nối đủ 3 điểm trong Mục 5 ==")
    src = open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
    n = f = 0
    for label, ok in [
        ("(1) _core_toks tách tiền tố (đối xứng hai phía)",
         "_strip_demographic_prefix(primary_key)" in src),
        ("(2) _hc_eff dùng cho phép thử tập-con", "_hc_eff = self._strip_demographic_prefix(_hc)" in src),
        ("(3) _gen_rows lưu nhãn SẠCH (sort key + prose)", "_gen_rows.append((_hc_eff," in src),
        ("_syndrome_is_hu xét trên nhãn sạch", "self._syndrome_is_hu(_hc_eff)" in src),
        ("regex neo đầu chuỗi + từ vựng ĐÓNG",
         r"r'^\s*(người\s+lớn|trẻ\s+em)\s*[-–]\s*(?=\S)'" in src),
        # Quét MÃ THẬT, bỏ dòng chú thích — nếu không sẽ bắt trúng chính câu cảnh báo
        # "tách mù bằng split(' - ') sẽ ĐẢO CỰC" nằm ngay trên _DEMO_PREFIX_RE.
        ("KHÔNG dùng split(' - ') mù ở tầng này",
         not any("split(' - ')" in ln for ln in src.splitlines()
                 if not ln.lstrip().startswith("#"))),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_strip, test_no_overreach, test_kb_blast_radius, test_wiring):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
