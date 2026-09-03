#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_menstrual_context_gate.py — Khóa cổng BỐI CẢNH KINH NGUYỆT + chặn bẫy tái sinh cổng.

(A) CỔNG 'kinh hành' — audit ổn định bắt được: lời khai THUẦN đau đầu
        "đau đầu, nhức đầu, đầu thống, đau nửa đầu, nặng đầu, váng đầu"
    bị gọi thành 'Kinh hành đầu thống' — tức gán chứng ĐAU ĐẦU KHI HÀNH KINH cho người không hề
    nhắc kinh nguyệt (và có thể là nam). Cùng lớp lỗi với bệnh thai nghén từng bị gán oan cho ca
    hô hấp; cách vá cũng giống: CỔNG DỮ LIỆU trong data/disease_gates.json, không sửa mù tầng khớp.

(B) BẪY TÁI SINH — quan trọng hơn (A). data/disease_gates.json ĐÃ được sửa TAY nhiều lần sau khi
    sinh, và những sửa đó là bản vá lâm sàng thật:
        - cổng Long bế siết bỏ 'nước tiểu'/'tiểu' trần (chỉ nhận 'bí tiểu/tiểu không thông');
        - cổng Hư lao bỏ 'mệt mỏi' trần, đòi dấu hao mòn/mạn;
        - ba cổng (hen suyễn, THAI SẢN nhâm thần/ố trở, suy nhược thần kinh) từng CHỈ có trong JSON.
    Chạy scripts/build_disease_gates.py sẽ âm thầm HOÀN TÁC tất cả (đo được: 104 cổng -> 101).
    Nhóm (B) khóa việc đó: JSON phải giữ các dấu hiệu của bản vá, và script phải TỪ CHỐI ghi đè.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_menstrual_context_gate.py   (exit != 0 nếu fail)
Không cần Neo4j/LLM.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

GATES_PATH = os.path.join(ROOT, "data", "disease_gates.json")


def _gates():
    return json.load(open(GATES_PATH, encoding="utf-8"))["gates"]


def _find(name):
    for g in _gates():
        if name in g.get("names", []):
            return g
    return None


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_cong_kinh_hanh():
    print("== (A) Cổng bối cảnh kinh nguyệt ==")
    n = f = 0
    g = _find("băng lậu")
    ok = g is not None
    n += _chk("có cổng kinh nguyệt/thai sản", ok)
    f += (not ok)
    if not ok:
        return n, f
    ok = "kinh hành" in g["names"]
    n += _chk("'kinh hành' nằm trong cổng", ok)
    f += (not ok)
    # lời khai thuần đau đầu KHÔNG chạm require nào -> bệnh 'Kinh hành ...' bị loại
    thuan = "đau đầu, nhức đầu, đầu thống, đau nửa đầu, nặng đầu, váng đầu"
    hit = [r for r in g["requires"] if r in thuan]
    ok = not hit
    n += _chk("lời khai thuần đau đầu KHÔNG qua cổng", ok, f"chạm: {hit}")
    f += (not ok)
    # lời khai CÓ bối cảnh kinh nguyệt thì phải qua
    for txt in ("đau đầu khi hành kinh, kinh nguyệt không đều",
                "đau đầu, thấy kinh ra ít", "nhức đầu trước kỳ kinh nguyệt"):
        ok = any(r in txt for r in g["requires"])
        n += _chk(f"có bối cảnh kinh -> QUA cổng: {txt[:38]}", ok)
        f += (not ok)
    return n, f


def test_ban_va_trong_json_con_nguyen():
    print("\n== (B1) Các bản vá chỉ-có-trong-JSON phải còn ==")
    n = f = 0
    lb = _find("long bế")
    ok = lb is not None and not any(r in ("nước tiểu", "tiểu", "đi tiểu", "niệu") for r in lb["requires"])
    n += _chk("Long bế KHÔNG còn require 'nước tiểu'/'tiểu' trần", ok)
    f += (not ok)
    hl = _find("hư lao")
    ok = hl is not None and "mệt mỏi" not in hl["requires"]
    n += _chk("Hư lao KHÔNG còn require 'mệt mỏi' trần", ok)
    f += (not ok)
    for nm, label in (("nhâm thần", "cổng THAI SẢN (nhâm thần/ố trở/tử giản)"),
                      ("hen phế quản", "cổng hen suyễn"),
                      ("suy nhược thần kinh", "cổng suy nhược thần kinh")):
        ok = _find(nm) is not None
        n += _chk(f"còn {label}", ok)
        f += (not ok)
    return n, f


def test_script_tu_choi_ghi_de():
    """Script sinh cổng phải DỪNG khi JSON đã đi trước nó — nếu không, một lần chạy vô ý là mất vá."""
    print("\n== (B2) build_disease_gates.py từ chối ghi đè mù ==")
    before = open(GATES_PATH, encoding="utf-8").read()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, os.path.join("scripts", "build_disease_gates.py")],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)
    after = open(GATES_PATH, encoding="utf-8").read()
    n = f = 0
    ok = after == before
    n += _chk("chạy script KHÔNG làm đổi data/disease_gates.json", ok)
    f += (not ok)
    ok = p.returncode != 0
    n += _chk("thoát với mã lỗi != 0 (để CI bắt được)", ok, f"rc={p.returncode}")
    f += (not ok)
    ok = "DỪNG" in (p.stdout or "")
    n += _chk("in cảnh báo rõ ràng", ok)
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_cong_kinh_hanh, test_ban_va_trong_json_con_nguyen, test_script_tu_choi_ghi_de):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
