#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_herb_safety_note.py — Khóa cảnh báo AN TOÀN DƯỢC ở Mục 5 (họ Ô đầu).

VÌ SAO: cổng Phụ tử/Ô đầu cũ chỉ treo trên nhánh BẮC CẦU dương->khí (biến _dk). Bài tới qua nhánh
tập-con (_subset_ok) — như 'Bát vị hoàn (Kim quỹ thận khí hoàn)' ở ca thật nữ 34t — ra thẳng mà
KHÔNG một chữ cảnh báo, dù chứa Phụ tử. Cảnh báo phải theo VỊ THUỐC, ĐỘC LẬP nhánh khớp.

HAI VỊ GIẢ DANH bắt buộc loại trừ:
  - 'Địa phụ tử' (Kochia scoparia, hạt cây chổi xể) — thanh nhiệt lợi thấp, KHÔNG thuộc họ Ô đầu.
  - 'Ma hoàng CĂN' (RỄ) — thu sáp CHỈ hãn, NGƯỢC cực với ma hoàng (phát hãn).

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_herb_safety_note.py
Không cần Neo4j/LLM.
"""
import csv
import io
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
_o = F.__new__(F)


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def _tox(vi):
    """Vị họ Ô ĐẦU. Gọi MÃ THẬT (F._herb_safety_flags), KHÔNG chép bản sao — bản sao vẫn xanh kể cả
    khi bộ dò trong nguồn bị gỡ sạch, tức là không khóa được gì (đã đo đúng như vậy một lần)."""
    return F._herb_safety_flags(_o._dedupe_herbs(vi))["aconite"]


def _eph(vi):
    """Ma hoàng — HẠNG MỤC RIÊNG, rủi ro khác (phát hãn/kích thích tim mạch), không phải 'có độc'."""
    return F._herb_safety_flags(_o._dedupe_herbs(vi))["ephedra"]


def test_detect():
    print("== (1) Nhận đúng vị họ Ô đầu ==")
    n = f = 0
    for label, vi, want in [
        ("Bát vị hoàn (ca thật)",
         "Thục địa, Hoài sơn, Đan bì, Bạch linh, Trạch tả, Sơn thù, Nhục quế, Phụ tử, "
         "Thỏ ty tử, Tang phiêu tiêu", ["Phụ tử"]),
        ("Dương thị... (ca thật)",
         "Thục phụ tử, Hán phòng kỷ (Phòng kỷ), Qui bản chích, Thục địa, Nhục quế", ["Phụ tử"]),
        # 'ô đầu' KHÔNG phải chuỗi con của 'xuyên ô'/'thảo ô' -> mỗi vị ra đúng tên của nó
        ("Xuyên ô + Thảo ô", "Xuyên ô, Thảo ô, Cam thảo", ["Xuyên ô", "Thảo ô"]),
        ("phụ phiến (tên thương phẩm, trước đây BỎ SÓT)", "Phụ phiến, Can khương", ["Phụ phiến"]),
        ("bài không độc", "Đẳng sâm, Bạch truật, Bạch linh, Cam thảo", []),
    ]:
        got = _tox(vi)
        ok = got == want
        n += _chk(f"{label}: {want}", ok, "" if ok else f"nhận {got}")
        f += (not ok)
    return n, f


def test_false_friends():
    """Hai vị GIẢ DANH — báo động giả ở đây làm thầy thuốc mất tin vào cảnh báo."""
    print("\n== (2) Không báo động giả ==")
    n = f = 0
    for label, vi in [
        ("Địa phụ tử (Kochia, không độc)", "Địa phụ tử, Bạch tiễn bì, Khổ sâm"),
        ("Ma hoàng căn (RỄ, thu sáp chỉ hãn)", "Ma hoàng căn, Hoàng kỳ, Mẫu lệ"),
        ("cả hai cùng dòng", "Địa phụ tử, Ma hoàng căn, Cam thảo"),
    ]:
        got = _tox(vi)
        ok = got == []
        n += _chk(f"KHÔNG cảnh báo: {label}", ok, "" if ok else f"báo nhầm {got}")
        f += (not ok)
    # ...nhưng Phụ tử thật đứng cạnh vị giả danh thì VẪN phải bắt
    got = _tox("Địa phụ tử, Thục phụ tử, Cam thảo")
    ok = got == ["Phụ tử"]
    n += _chk("vẫn bắt Phụ tử thật khi đứng cạnh 'Địa phụ tử'", ok, "" if ok else str(got))
    f += (not ok)
    return n, f


def test_kb_scan():
    """Quét KB thật: liệt kê dòng chứa vị giả danh để chốt không có báo động giả nào."""
    print("\n== (3) Quét KB thật ==")
    n = f = 0
    rows = list(csv.DictReader(io.open(os.path.join(ROOT, "data", "Medicine_clean.csv"),
                                       encoding="utf-8-sig")))
    cols = list(rows[0].keys())
    cVi = cols[8] if len(cols) > 8 else cols[-1]
    fake = [r for r in rows if re.search(r'địa phụ tử|ma hoàng căn', (r[cVi] or ""), re.I)]
    bad = [r for r in fake if _tox(r[cVi] or "")
           and not re.search(r'(?<!địa )phụ tử|ô đầu|xuyên ô|thảo ô', (r[cVi] or ""), re.I)]
    n += _chk(f"{len(fake)} dòng KB chứa vị giả danh -> 0 báo động giả", not bad,
              "" if not bad else f"{len(bad)} dòng báo nhầm")
    f += (not not bad)
    real = [r for r in rows if _tox(r[cVi] or "")]
    ok = len(real) > 0
    n += _chk(f"vẫn bắt được {len(real)} dòng KB có vị họ Ô đầu thật", ok)
    f += (not ok)
    return n, f


def test_ephedra_category():
    """Ma hoàng là hạng mục RIÊNG: rủi ro phát hãn/tim mạch, KHÔNG được gọi là 'có độc'."""
    print("\n== (4) Ma hoàng — hạng mục riêng ==")
    n = f = 0
    ok = _eph("Ma hoàng chích, Cam thảo") == ["Ma hoàng"] and _tox("Ma hoàng chích, Cam thảo") == []
    n += _chk("ma hoàng vào 'ephedra', KHÔNG vào 'aconite'", ok)
    f += (not ok)
    # 'ma hoàng căn' (RỄ) ngược cực — loại trừ này giờ mới THẬT SỰ có tải
    ok = _eph("Ma hoàng căn, Hoàng kỳ") == []
    n += _chk("'ma hoàng căn' KHÔNG bị gắn cờ (loại trừ đã hết là code chết)", ok)
    f += (not ok)
    s = F._herb_safety_sentence([], ["Ma hoàng"])
    ok = "CÓ ĐỘC" not in s and "ephedrin" in s
    n += _chk("câu ma hoàng KHÔNG chứa 'CÓ ĐỘC'", ok, "" if ok else s[:90])
    f += (not ok)
    s2 = F._herb_safety_sentence(["Phụ tử"], ["Ma hoàng"])
    ok = "CÓ ĐỘC" in s2 and "ephedrin" in s2 and ". chứa" not in s2 and ". Ma" not in s2
    n += _chk("có cả hai -> một câu ghép sạch, không chữ thường sau dấu chấm", ok)
    f += (not ok)
    return n, f


def test_annotator():
    """Hàm chú giải hoist: phủ mọi nhánh in, idempotent, có cổng chỉ định cho ma hoàng."""
    print("\n== (5) Chú giải phủ toàn bộ Mục 5 ==")
    o = F.__new__(F)
    F._load_csv_data(o)
    md = ("- Trị Bệnh **X** → bài **Ma hoàng thang**\n"
          "  - *Vị thuốc:* Ma hoàng, Quế chi, Hạnh nhân, Cam thảo\n"
          "- Trị Bệnh **Y** → bài **Chân vũ thang**\n"
          "  - *Vị thuốc:* Phụ tử, Bạch truật, Bạch linh\n")
    n = f = 0
    # Cốt lõi NGOẠI CẢM BIỂU: ma hoàng LÀ phép trị -> cảnh báo ở đó là dạy sai y lý
    r1 = o._annotate_toxic_herb_lines(md, "Phong hàn phạm biểu")
    ok = "ephedrin" not in r1 and "Ô đầu" in r1
    n += _chk("biểu chứng -> im lặng ma hoàng, VẪN cảnh báo Ô đầu", ok)
    f += (not ok)
    r2 = o._annotate_toxic_herb_lines(md, "Tỳ thận dương hư")
    ok = "ephedrin" in r2
    n += _chk("nội thương -> CÓ cảnh báo ma hoàng", ok)
    f += (not ok)
    ok = o._annotate_toxic_herb_lines(r2, "Tỳ thận dương hư").count("An toàn dược") == \
        r2.count("An toàn dược") == 2
    n += _chk("IDEMPOTENT: chạy 2 lần vẫn 2 dòng cảnh báo", ok)
    f += (not ok)
    clean = "- Trị Bệnh **Z** → bài **Tứ quân**\n  - *Vị thuốc:* Đẳng sâm, Bạch truật, Cam thảo\n"
    ok = o._annotate_toxic_herb_lines(clean, "Tỳ khí hư") == clean
    n += _chk("bài không có vị lưu ý -> markdown BẤT BIẾN", ok)
    f += (not ok)
    return n, f


def test_wiring():
    print("\n== (6) Nối vào đường dựng markdown ==")
    n = f = 0
    for label, ok in [
        ("có khối _tox_note (nhánh xấp xỉ giữ ngữ cảnh riêng)", "_tox_note" in SRC),
        ("cảnh báo ĐỘC LẬP nhánh _dk", "_flags = self._herb_safety_flags(" in SRC),
        ("loại trừ vị giả danh", '_TOXIC_LOOKALIKE = ("địa phụ tử", "ma hoàng căn")' in SRC),
        ("'phụ phiến' đã vào danh sách (trước đây bỏ sót 5 dòng KB)", '"phụ phiến"' in SRC),
        ("_tox_note được nối vào dòng in ra", "{_tox_note}" in SRC),
        # CHỐNG HỒI QUY IM LẶNG: cảnh báo phải nằm ở tầng quét markdown CHỐT, không phải rải
        # inline theo từng nhánh — ai thêm nhánh in bài mới cũng tự động được phủ.
        ("có hàm chú giải hoist", "def _annotate_toxic_herb_lines" in SRC),
        ("hàm được gọi trên markdown đã chốt",
         "final_markdown = self._annotate_toxic_herb_lines(final_markdown" in SRC),
        ("gọi ĐÚNG 1 lần (không rải lại theo nhánh)",
         SRC.count("self._annotate_toxic_herb_lines(") == 1),
        ("lời khuyên gia vị ôn dương KHÔNG còn xui Phụ tử trần",
         "ưu tiên Nhục quế/Can " in SRC and "Phụ tử CÓ ĐỘC" in SRC),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_detect, test_false_friends, test_kb_scan, test_ephedra_category,
               test_annotator, test_wiring):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
