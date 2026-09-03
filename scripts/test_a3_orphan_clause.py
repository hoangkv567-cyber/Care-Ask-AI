#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_a3_orphan_clause.py — Khóa việc dọn MẢNH VỠ do luật (a3) của censor để lại.

VẤN ĐỀ: (a3) gỡ đúng span '<động từ nhân-quả> <triệu chứng bịa>' nhưng không nhìn phần dư HAI BÊN
span trong cùng mệnh đề. Ca thật (nữ 34t) in ra cho bệnh nhân:
    "...không đủ lực để điều tiết thủy dịch, về đêm."          <- trạng ngữ MỒ CÔI
    "...điều tiết thủy dịch, về đêm chân."                     <- CẮT GIỮA CỤM ('phù' có trong
                                                                  vocab, 'phù chân' thì không)

BẢN VÁ đặt TẠI ĐIỂM CẮT (_a3_cut), neo theo span mà chính (a3) vừa khớp — KHÔNG phải tầng quét lại
toàn văn bản chạy sau censor. Lý do: 'chạy sau censor' KHÔNG đồng nghĩa 'chỉ chạm văn bản censor đã
sửa' — nhánh except của khối biện luận đi thẳng tới tầng hậu xử lý mà censor chưa hề chạy.

4 CỔNG (AND, fail-closed): biên trái là ','/';' cùng dòng · phần dư 1-3 từ · phần dư KHÔNG chứa từ
y lý · phần dư KHÔNG chứa lời khai bệnh nhân.

⚠ KHÔNG có whitelist trạng ngữ: đo được whitelist xóa oan 13/14 mệnh đề CƠ CHẾ y lý — đúng loại câu
prompt đang dạy LLM viết. Danh sách ĐÓNG _A3_YLY_RE chỉ dùng để GIỮ LẠI.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_a3_orphan_clause.py   (exit != 0 nếu fail)
Không cần Neo4j/LLM.
"""
import json
import io
import os
import re
import subprocess
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


# Mệnh đề CƠ CHẾ y lý hợp lệ — TUYỆT ĐỐI không được đụng. Nhiều câu có trạng ngữ thời gian đứng
# sau dấu phẩy y hệt mảnh vỡ; đây chính là lý do cấm whitelist trạng ngữ.
MECH = [
    "Âm hư sinh nội nhiệt, về đêm dương khí nhập vào phần Âm.",
    "Vệ khí ban ngày hành ở biểu, về đêm hành ở phần Âm.",
    "Can tàng huyết, về đêm huyết quy về Can.",
    "Mạch trầm tế, vô lực.",
    "Lưỡi đỏ, ít rêu.",
    "Chất lưỡi nhợt, rêu trắng mỏng.",
    "Thận không cố nhiếp nên tiểu đêm, nước tiểu trong.",
    "Tính chất của mủ do nhiệt độc sinh ra.",
    "Sốt hâm hấp, về chiều như triều dâng.",
    "Phù hai chi dưới, buổi sáng ở mi mắt.",
    "Tỳ chủ vận hóa, Vị chủ thu nạp.",
    "Dương khí hư suy, không ôn ấm được cơ nhục.",
]
LOI_KHAI = ["", "đau lưng, mỏi gối", "sốt về chiều, mất ngủ"]


def test_a_must_fix():
    print("== (A) Hai ca mảnh vỡ PHẢI được dọn ==")
    o = pipe()
    n = f = 0
    for txt, lk, forbidden in [
        ("do dương hư nên không đủ lực để điều tiết thủy dịch, về đêm gây mất ngủ.",
         "phù chân, tiểu đêm", ", về đêm."),
        ("không đủ lực để điều tiết thủy dịch, về đêm sinh ra phù chân.",
         "mất ngủ, đau lưng", "về đêm chân"),
    ]:
        out = o._post_process_hallucinations(txt, lk)
        ok = forbidden not in out and "thủy dịch" in out
        n += _chk(f"hết mảnh vỡ {forbidden!r}", ok, "" if ok else repr(out.strip()))
        f += (not ok)
    return n, f


def test_b_mechanism_intact():
    """Nhóm lớn nhất: mệnh đề cơ chế phải BẤT BIẾN TỪNG KÝ TỰ với mọi lời khai."""
    print("\n== (B) Mệnh đề cơ chế y lý bất biến ==")
    o = pipe()
    n = f = 0
    bad = []
    for lk in LOI_KHAI:
        for m in MECH:
            if o._post_process_hallucinations(m, lk).strip() != m.strip():
                bad.append((lk, m))
    ok = not bad
    n += _chk(f"{len(MECH)} câu × {len(LOI_KHAI)} lời khai = {len(MECH)*len(LOI_KHAI)} phép thử",
              ok, "" if ok else f"{len(bad)} câu bị đổi: {bad[:2]}")
    f += (not ok)
    return n, f


def test_b2_each_gate_holds():
    """Ca mà (a3) THỰC SỰ chạy, và mỗi cổng là thứ DUY NHẤT giữ mệnh đề lại.

    Khác nhóm B: ở nhóm B, (a3) không hề khớp nên mọi đột biến cổng đều vô hại — nhóm đó không
    khóa được cổng nào. Bốn ca dưới đây đã kiểm: (a3) CÓ chạy, và gỡ cổng tương ứng thì mệnh đề
    bị nuốt oan.
    """
    print("\n== (B2) Mỗi cổng giữ đúng phần nội dung ==")
    o = pipe()
    n = f = 0
    for label, txt, lk, keep in [
        ("cổng Y LÝ giữ 'tại kinh lạc'",
         "Can khí uất kết, tại kinh lạc gây đau tức.", "đau lưng", "tại kinh lạc"),
        ("cổng Y LÝ giữ 'tại tỳ vị'",
         "Thấp trọc ứ đọng, tại tỳ vị gây đầy bụng.", "đau lưng", "tại tỳ vị"),
        ("cổng ĐỘ DÀI giữ trạng ngữ 6 từ",
         "Khí trệ, sau bữa ăn khoảng hai giờ gây đầy bụng.", "đau lưng", "sau bữa ăn khoảng hai giờ"),
        ("cổng LỜI KHAI giữ 'tay chân lạnh'",
         "Tỳ hư, tay chân lạnh gây phù nề.", "tay chân lạnh", "tay chân lạnh"),
    ]:
        out = o._post_process_hallucinations(txt, lk)
        ok = keep in out
        n += _chk(label, ok, "" if ok else repr(out.strip()))
        f += (not ok)
    return n, f


def test_c_input_symptoms_safe():
    """Lời khai THẬT nằm ở mệnh đề đuôi -> censor không được chạm (cổng _is_input_symptom)."""
    print("\n== (C) Lời khai thật ở mệnh đề đuôi ==")
    o = pipe()
    n = f = 0
    for txt, lk in [
        ("Thận dương hư, về đêm gây tiểu nhiều lần.", "tiểu nhiều lần, đau lưng"),
        ("Thận dương hư, về đêm gây mất ngủ.", "mất ngủ"),
        ("Tỳ hư, về đêm gây phù chân.", "phù chân"),
        ("Âm hư nội nhiệt, về chiều gây sốt.", "sốt về chiều"),
    ]:
        out = o._post_process_hallucinations(txt, lk)
        ok = out.strip() == txt.strip()
        n += _chk(f"BẤT BIẾN: {txt[:40]}...", ok, "" if ok else repr(out.strip()))
        f += (not ok)
    return n, f


def test_d_line_structure():
    """Biên trái không bao giờ được là '\\n' — nếu không sẽ nuốt gạch đầu dòng markdown."""
    print("\n== (D) Cấu trúc dòng/markdown ==")
    o = pipe()
    n = f = 0
    src = "- Thận dương hư suy.\n- Về đêm gây mất ngủ.\n- Mạch trầm tế."
    out = o._post_process_hallucinations(src, "đau lưng")
    for label, ok in [
        ("số xuống dòng không đổi", out.count("\n") == src.count("\n")),
        ("không sinh '..'", ".." not in out),
        ("gạch đầu dòng còn nguyên", out.count("\n- ") == src.count("\n- ")),
    ]:
        n += _chk(label, ok, "" if ok else repr(out))
        f += (not ok)
    return n, f


def test_e_golden_corpus():
    """40 ca THẬT (data/audit_sample_report.json) với input THẬT của từng ca.

    ⚠ Bắt buộc dùng lời khai THẬT: chạy với lời khai RỖNG thổi phồng số câu bị đụng ~22 lần,
    cho ra bức tranh rủi ro sai hoàn toàn.
    """
    print("\n== (E) Golden corpus 40 ca thật ==")
    p = os.path.join(ROOT, "data", "audit_sample_report.json")
    if not os.path.exists(p):
        print("[NOTE] thiếu data/audit_sample_report.json — bỏ qua (chạy audit_sample_stability.py để sinh)")
        return 0, 0
    o = pipe()
    cases = json.load(io.open(p, encoding="utf-8"))

    # So với BASELINE (hành vi (a3) CŨ = chỉ gỡ đúng span), không phải so với văn bản gốc — censor
    # vốn đã đụng nhiều câu vì lý do khác, đo kiểu đó là đo nhầm thứ.
    def _a3_baseline(_self, text, rx, _input_terms):
        return rx.sub('', text)

    _patched = F._a3_cut
    total = diff = 0
    changed = []
    for c in cases:
        inp = c.get("input") or ""
        for sent in re.split(r'(?<=[.!?])\s+', (c.get("answer") or "")):
            s = sent.strip()
            if len(s) < 12:
                continue
            total += 1
            F._a3_cut = _patched
            a = o._post_process_hallucinations(s, inp)
            F._a3_cut = _a3_baseline
            b = o._post_process_hallucinations(s, inp)
            if a != b:
                diff += 1
                changed.append((s[:70], a.strip()[:70]))
    F._a3_cut = _patched
    n = f = 0
    ok = total > 200
    n += _chk(f"corpus đủ lớn: {total} câu", ok)
    f += (not ok)
    # Bản vá chỉ được đụng số câu RẤT nhỏ — và mỗi câu đụng phải là mảnh vỡ thật.
    ok = diff <= 3
    n += _chk(f"số câu KHÁC BASELINE = {diff}/{total} (ngưỡng <= 3)", ok,
              "" if ok else str(changed[:3]))
    f += (not ok)
    if changed:
        print("[NOTE] câu bản vá đụng — đọc tay để chắc phần bị gỡ là RÁC, không phải y lý:")
        for _b, _a in changed[:5]:
            print(f"        trước: {_b!r}")
            print(f"        sau  : {_a!r}")
    return n, f


def test_f_determinism():
    """Kết quả phải BYTE-IDENTICAL qua các PYTHONHASHSEED (vocab duyệt trên set)."""
    print("\n== (F) Tất định ==")
    outs = set()
    for seed in ("0", "1", "7"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--emit"],
                           capture_output=True, env=env, cwd=ROOT)
        outs.add(r.stdout)
    ok = len(outs) == 1
    return _chk(f"3 seed -> {len(outs)} kết quả (cần 1)", ok), (0 if ok else 1)


def test_g_source_gates():
    """Mutation-guard: bốn cổng phải còn trong nguồn."""
    print("\n== (G) Bốn cổng còn trong mã nguồn ==")
    src = open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
    n = f = 0
    for label, ok in [
        ("(a3) gọi _a3_cut, không re.sub trần", "new_text = self._a3_cut(" in src),
        ("cổng 1: biên trái chỉ ',' hoặc ';'", '_lm.group(0) in ",;"' in src),
        ("cổng 2: phần dư 1-3 từ", "0 < _nw <= 3" in src),
        ("cổng 3: giữ nội dung y lý", "_A3_YLY_RE.search(_residue)" in src),
        ("cổng 4: giữ lời khai bệnh nhân", "for t in input_terms)" in src),
        ("có log kiểm toán", "[MẢNH VỠ a3]" in src),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    return n, f


def _emit():
    o = pipe()
    for lk in LOI_KHAI:
        for m in MECH + ["do dương hư nên không đủ lực để điều tiết thủy dịch, về đêm gây mất ngủ.",
                         "không đủ lực để điều tiết thủy dịch, về đêm sinh ra phù chân."]:
            sys.stdout.write(o._post_process_hallucinations(m, lk) + "|")


def main():
    if "--emit" in sys.argv:
        _emit()
        return
    tp = tf = 0
    for fn in (test_a_must_fix, test_b_mechanism_intact, test_b2_each_gate_holds,
               test_c_input_symptoms_safe,
               test_d_line_structure, test_e_golden_corpus, test_f_determinism, test_g_source_gates):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
