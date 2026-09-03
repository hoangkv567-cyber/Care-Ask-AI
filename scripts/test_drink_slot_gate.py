#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_drink_slot_gate.py — Khóa cổng Ô PHÂN CỰC "hành vi uống" (_strip_unfounded_drinking_behavior).

VÌ SAO CÓ CỔNG NÀY: ca thật (nữ 34t, dương hư), 4/4 lần chạy LLM đều BỊA tính chất uống, mỗi lần
một cách diễn đạt khác — "không thể uống nhiều nước để giải khát" / "uống nước nhiều vẫn không giải
được" / "khát mà không uống được đủ" / "khát nước dù không uống nhiều". Lời khai chỉ có "Khát, thích
uống". Đây là NGỤY TẠO BẰNG CHỨNG CỦNG CỐ: máy đã chốt Thận dương hư rồi bịa ra đúng dấu xác nhận
nó (渴不欲飲), khiến thầy thuốc KHÔNG hỏi lại câu hỏi phân cực quan trọng nhất của ca.

HAI TRỤC ĐÃ THUA, KHÔNG QUAY LẠI:
  - Prompt cấm-cụm (luật 21): cấm đích danh 7 cụm -> LLM viết cụm thứ 8 NGAY lần chạy kế.
  - Censor từ-điển: "không uống nhiều" KHÔNG có trong vocab 3761; diễn giải vòng là vô hạn.
Cổng này hỏi câu KHÁC: "lời khai có CẤP PHÉP cho ô hành vi uống không?" — thế giới ĐÓNG. Neo vào
ĐỘNG TỪ 'uống' + danh sách bổ ngữ phân cực ĐÓNG, nên KHÔNG tồn tại "cụm thứ 9".

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_drink_slot_gate.py   (exit != 0 nếu fail)
Không cần Neo4j/LLM.
"""
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
logging.disable(logging.INFO)

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402

SRC = open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
_o = F.__new__(F)


def f(text, sym):
    return _o._strip_unfounded_drinking_behavior(text, sym)


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


# Lời khai THẬT của ca — KHÔNG mô tả tính chất uống, nên ô 'hành vi uống' chưa được cấp phép.
LK = ("tiểu tiện trong dài, tinh thần uể oải, nước tiểu trong, khát nước, tiểu đêm, "
      "đau lưng, mỏi gối, rêu trắng mỏng, lưỡi bệu")


def test_must_catch():
    """Năm cách diễn đạt THẬT đã quan sát qua 4 lần chạy + cụm hạ bản vá prompt."""
    print("== (A) Phải bắt mọi cách diễn đạt ==")
    n = fl = 0
    for s in [
        "gây khát nước dù không uống nhiều.",          # cụm thứ 8 — hạ luật 21
        "khát mà không uống được đủ.",
        "uống nước nhiều vẫn không giải được.",
        "không thể uống nhiều nước để giải khát.",
        "khát mà không uống được nhiều.",
        "bệnh nhân chẳng uống nổi bao nhiêu.",         # biến thể từ vựng (chẳng/nổi)
    ]:
        ok = f(s, LK) != s
        n += _chk(f"bắt: {s[:48]!r}", ok)
        fl += (not ok)
    return n, fl


def test_must_keep():
    """Bất biến TỪNG BYTE — nếu hỏng nhóm này thì cổng đang giết biện luận hợp lệ."""
    print("\n== (B) Phải giữ nguyên từng byte ==")
    n = fl = 0
    for s in [
        # Cơ chế Kim quỹ mà luật 21 CHO PHÉP — không chứa động từ 'uống' nên an toàn theo cấu trúc
        "Thận dương hư, khí hóa bất lợi, tân dịch bất thượng thừa nên miệng vẫn khát.",
        "Thủy dịch không được cố nhiếp, rót thẳng xuống bàng quang thành tiểu trong dài.",
        "Bệnh nhân khát và thích uống.",               # KHÔNG có bổ ngữ phân cực
        "Ăn uống kém do Tỳ mất kiện vận.",             # danh từ ghép 'ăn uống'
        "Sắc uống ngày một thang.",                    # ngôn ngữ dùng thuốc
        "Uống thuốc khi còn ấm, chia hai lần.",        # 'uống thuốc' + 'ấm' -> vẫn phải giữ
        "Rêu trắng mỏng là rêu sinh lý, vị khí còn tốt.",
    ]:
        ok = f(s, LK) == s
        n += _chk(f"giữ: {s[:48]!r}", ok, "" if ok else repr(f(s, LK)[:60]))
        fl += (not ok)
    return n, fl


def test_surgical_cut():
    """Cắt PHẪU THUẬT: câu bịa thường dính chung mệnh đề với y lý HỢP LỆ — phải giữ vế đúng.

    Lỗi thật của phiên bản đầu (commit ce2637c): cắt theo CẢ mệnh đề nên
    "Dương hư không hóa tân dịch nên bệnh nhân không uống được nhiều." -> "." — mất luôn vế
    "Dương hư không hóa tân dịch", tức chính cơ chế Kim quỹ mà luật 21 CHO PHÉP. Hệ quả quan sát
    được trên app thật: Mục 3 im lặng hoàn toàn về khát dù bệnh nhân CÓ khai 'khát nước'.
    """
    print("\n== (A2) Cắt phẫu thuật, giữ vế y lý ==")
    n = fl = 0
    for txt, must_keep in [
        ("Dương hư không hóa tân dịch nên bệnh nhân không uống được nhiều.",
         "Dương hư không hóa tân dịch"),
        ("Thận dương suy không khí hóa được thủy dịch, khiến bệnh nhân uống nhiều vẫn không giải khát.",
         "Thận dương suy không khí hóa được thủy dịch"),
        ("Khát nước là do dương hư không hóa được tân dịch, không thể uống nhiều để giải.",
         "dương hư không hóa được tân dịch"),
    ]:
        out = f(txt, LK)
        ok = must_keep in out and out != txt
        n += _chk(f"giữ vế y lý: {must_keep[:38]!r}", ok, "" if ok else repr(out[:70]))
        fl += (not ok)
    # KHÔNG để lại dấu câu mồ côi / khoảng trắng dính
    for txt in ["Dương hư không hóa tân dịch nên bệnh nhân không uống được nhiều.",
                "Bệnh nhân có tiểu đêm; khát mà không uống được đủ.",
                "bệnh nhân chẳng uống nổi bao nhiêu."]:
        out = f(txt, LK)
        ok = not re.match(r'^\s*[,;.]', out) and ",có" not in out and ";có" not in out
        n += _chk(f"không dấu câu mồ côi: {out[:44]!r}", ok)
        fl += (not ok)
    # Ghi chú KHÔNG được lặp ý khi vế giữ lại đã nhắc khát
    out = f("gây khát nước dù không uống nhiều.", LK)
    ok = out.lower().count("khát") == 1
    n += _chk("không lặp 'khát' hai lần", ok, "" if ok else repr(out[:60]))
    fl += (not ok)
    return n, fl


def test_license():
    """Lời khai ĐÃ mô tả tính chất uống -> cổng không được đụng một byte."""
    print("\n== (C) Cổng cấp phép ==")
    n = fl = 0
    for lic in ["khát, thích uống nước lạnh", "khát, thích uống nóng",
                "khát nhưng không muốn uống", "miệng nhạt không khát"]:
        sym = LK + ", " + lic
        bad = [s for s in ["khát mà không uống được đủ.", "uống nước nhiều vẫn không giải được."]
               if f(s, sym) != s]
        ok = not bad
        n += _chk(f"cấp phép bởi {lic!r} -> bất biến", ok)
        fl += (not ok)
    return n, fl


def test_real_corpus():
    """PHÉP ĐO QUYẾT ĐỊNH: 298 câu Mục 3+4 THẬT (40 ca, lời khai thật) phải KHÔNG ĐỔI.

    Khác 0 = cổng đang đụng văn bản lâm sàng thật -> DỪNG, không merge.
    ⚠ Phải dùng lời khai THẬT của từng ca và chỉ lấy Mục 3+4 — đó là phạm vi cổng thật sự chạy.
    """
    print("\n== (D) 298 câu Mục 3+4 thật ==")
    p = os.path.join(ROOT, "data", "audit_sample_report.json")
    if not os.path.exists(p):
        print("[NOTE] thiếu data/audit_sample_report.json — bỏ qua")
        return 0, 0
    tot = chg = 0
    changed = []
    for c in json.load(io.open(p, encoding="utf-8")):
        inp = c.get("input") or ""
        m = re.search(r'### 3\..*?(?=### 5|\Z)', c.get("answer") or "", re.S)
        if not m:
            continue
        for s in re.split(r'(?<=[.!?])\s+', m.group(0)):
            s = s.strip()
            if len(s) < 12:
                continue
            tot += 1
            if f(s, inp) != s:
                chg += 1
                changed.append(s[:70])
    n = fl = 0
    ok = tot > 200
    n += _chk(f"corpus đủ lớn: {tot} câu Mục 3+4", ok)
    fl += (not ok)
    ok = chg == 0
    n += _chk(f"số câu bị đổi = {chg} (yêu cầu ĐÚNG 0)", ok, "" if ok else str(changed[:3]))
    fl += (not ok)
    return n, fl


def test_scope_locked():
    """Cổng CHỈ được chạy trên Mục 3/4.

    Đo được trên corpus thật: dòng Mục 5 'Dùng bài **Hồ ma hoàn (uống trong)**' nằm cùng mệnh đề
    với chữ 'chưa' -> nếu cổng chạy trên Mục 5 thì XÓA MẤT CẢ DÒNG BÀI THUỐC. Khóa phạm vi lại.
    """
    print("\n== (E) Phạm vi bị khóa vào Mục 3/4 ==")
    n = fl = 0
    danger = ("- Trị Bệnh **Bạch biến** — *Thể gần nhất (tham khảo — chưa khớp chính xác hội "
              "chứng cốt lõi)* → Dùng bài **Hồ ma hoàn (uống trong)**")
    ok = f(danger, LK) != danger
    n += _chk("xác nhận dòng Mục 5 SẼ bị cổng ăn nếu chạy nhầm phạm vi", ok)
    fl += (not ok)
    for label, cond in [
        ("gọi ĐÚNG 1 lần", SRC.count("self._strip_unfounded_drinking_behavior(") == 1),
        ("chỉ gọi trên llm_explanation (Mục 3/4), KHÔNG trên final_markdown",
         "llm_explanation = self._strip_unfounded_drinking_behavior(llm_explanation, symptoms_str)" in SRC),
        ("KHÔNG gọi trên markdown đã ghép Mục 5",
         "self._strip_unfounded_drinking_behavior(final_markdown" not in SRC),
    ]:
        n += _chk(label, cond)
        fl += (not cond)
    return n, fl


def test_invariants():
    print("\n== (F) Bất biến chống lỗi lịch sử (censor từng xóa nhầm lời khai) ==")
    n = fl = 0
    # Mọi triệu chứng bệnh nhân KHAI phải còn nguyên số lần xuất hiện.
    txt = ("Bệnh nhân có tiểu đêm, nước tiểu trong, đau lưng và mỏi gối; "
           "khát mà không uống được đủ.")
    out = f(txt, LK)
    for kw in ["tiểu đêm", "nước tiểu trong", "đau lưng", "mỏi gối"]:
        ok = out.count(kw) == txt.count(kw)
        n += _chk(f"giữ nguyên lời khai thật: {kw!r}", ok)
        fl += (not ok)
    # Không xóa im lặng — phải để lại dấu vết cho thầy thuốc biết chỗ cần hỏi lại.
    ok = "cần hỏi lại" in f("khát mà không uống được đủ.", LK)
    n += _chk("để lại DẤU VẾT thay vì xóa im lặng", ok)
    fl += (not ok)
    # 'thích' TUYỆT ĐỐI không được nằm trong bộ bổ ngữ phân cực (sẽ xóa oan lời khai form).
    ok = not F._DRINK_POLARITY_RE.search("thích")
    n += _chk("'thích' KHÔNG phải bổ ngữ phân cực (cấm thêm)", ok)
    fl += (not ok)
    ok = f("", LK) == "" and f(None, LK) is None
    n += _chk("đầu vào rỗng/None -> no-op", ok)
    fl += (not ok)
    return n, fl


def main():
    tp = tf = 0
    for fn in (test_must_catch, test_must_keep, test_surgical_cut, test_license, test_real_corpus,
               test_scope_locked, test_invariants):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
