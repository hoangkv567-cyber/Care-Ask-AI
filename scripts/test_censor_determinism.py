#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_censor_determinism.py — Bộ chặn bịa triệu chứng phải TẤT ĐỊNH và không xé cụm.

VÌ SAO: _post_process_hallucinations duyệt trên SET (`for term in self._get_symptom_vocab()`).
Thứ tự lặp của set phụ thuộc hash randomization -> CÙNG một lời khai cho ra NHIỀU bản văn khác
nhau giữa các lần chạy. Bệnh án phải tái lập được, nếu không thì không ai gỡ lỗi được ca thật.

Hệ quả thứ hai (ca thật nữ 34t): 'mệt' được xét TRƯỚC 'mệt mỏi' -> luật (a2) cắt ', mệt' đúng biên
từ rồi bỏ lại 'mỏi' mồ côi -> Mục 3 in ra "tinh thần uể oải mỏi".

Bản vá CHỈ đổi thứ tự duyệt (cụm DÀI trước, khóa phụ theo chữ cái) — không bỏ qua term nào, nên
năng lực chống bịa đơn điệu KHÔNG giảm. Test này khóa cả hai mặt.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_censor_determinism.py
Không cần Neo4j/LLM (chạy lại chính nó qua subprocess với PYTHONHASHSEED khác nhau).
"""
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

LOI_KHAI = ("tiểu tiện trong dài, tinh thần uể oải, nước tiểu trong, khát nước, tiểu đêm, "
            "đau lưng, mỏi gối, rêu trắng mỏng, lưỡi bệu")

# Văn bản mẫu: trộn triệu chứng THẬT (phải giữ) và triệu chứng BỊA (phải xóa).
VAN_BAN = (
    "Tỳ thận dương hư là căn nguyên, dẫn đến tinh thần uể oải và mệt mỏi, ăn kém.\n"
    "Thận không cố nhiếp nên tiểu đêm, nước tiểu trong.\n"
    "Dương hư không ôn ấm được nên gây ra sốt cao và táo bón.\n"
    "Bệnh nhân còn ho nhiều.\n"
    "Đau lưng mỏi gối là biểu hiện của Thận hư.\n"
)

# KHE HỞ ĐÃ BIẾT (có sẵn, KHÔNG phải hồi quy của bản vá thứ tự duyệt): censor chỉ gỡ triệu chứng
# bịa nằm trong MẪU CÂU NHÂN-QUẢ ('dẫn đến X', 'gây ra X', 'X là do...'). Câu KHẲNG ĐỊNH TRẦN
# ("Bệnh nhân còn ho nhiều.") không mẫu nào phủ nên đi lọt. Mở rộng sang câu trần là việc RIÊNG,
# phải thẩm định đối kháng: chính bộ này từng xóa nhầm triệu chứng bệnh nhân THẬT khai.
GAP_BARE_ASSERTION = "ho nhiều"


def _censor():
    from src.fusion_pipeline import TCMFusionPipeline as F
    o = F.__new__(F)
    F._load_csv_data(o)      # BẮT BUỘC: từ điển triệu chứng lấy từ CSV; thiếu -> vocab rỗng,
    return o._post_process_hallucinations(VAN_BAN, LOI_KHAI)   # censor không xóa gì, test xanh giả


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_determinism():
    """Chạy lại chính file này dưới nhiều PYTHONHASHSEED — output phải BYTE-IDENTICAL."""
    print("== (1) Tất định qua các PYTHONHASHSEED ==")
    outs = {}
    for seed in ("0", "1", "3", "11", "42"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--emit"],
                           capture_output=True, env=env, cwd=ROOT)
        outs[seed] = r.stdout
    uniq = set(outs.values())
    ok = len(uniq) == 1
    n = _chk(f"5 seed -> {len(uniq)} bản văn khác nhau (cần 1)", ok)
    if not ok:
        for s, v in outs.items():
            print(f"    seed={s}: {v[:160]!r}")
    return n, (0 if ok else 1)


def test_content():
    print("\n== (2) Nội dung: giữ thật, xóa bịa, không xé cụm ==")
    out = _censor()
    low = out.lower()
    n = f = 0
    # Triệu chứng THẬT phải còn nguyên
    for kw in ("tinh thần uể oải", "tiểu đêm", "nước tiểu trong", "đau lưng", "mỏi gối"):
        ok = kw in low
        n += _chk(f"GIỮ lời khai thật: {kw!r}", ok)
        f += (not ok)
    # Triệu chứng BỊA trong mẫu câu nhân-quả phải bị xóa (đây là hợp đồng thật của censor)
    for kw in ("ăn kém", "sốt cao", "táo bón"):
        ok = kw not in low
        n += _chk(f"XÓA bịa (mẫu nhân-quả): {kw!r}", ok)
        f += (not ok)
    # Ghi nhận khe hở đã biết — KHÔNG khẳng định nó bị xóa, nhưng cũng không im lặng bỏ qua:
    # nếu ngày nào đó censor phủ được câu trần thì dòng này báo để cập nhật tài liệu.
    still = GAP_BARE_ASSERTION in low
    print(f"[NOTE] khe hở đã biết — câu khẳng định trần {GAP_BARE_ASSERTION!r}: "
          f"{'vẫn lọt (đúng như ghi nhận)' if still else 'ĐÃ ĐƯỢC PHỦ — cập nhật lại tài liệu'}")
    # Không để lại mảnh mồ côi: 'mỏi' đứng ngay sau 'uể oải' là dấu hiệu cụm bị xé
    ok = "uể oải mỏi" not in low
    n += _chk("KHÔNG còn mảnh mồ côi 'uể oải mỏi'", ok, "" if ok else repr(out[:200]))
    f += (not ok)
    return n, f


def test_sort_in_source():
    """Mutation-guard: gỡ sorted() thì test (1) hết bắt lỗi -> khóa luôn ở mã nguồn."""
    print("\n== (3) Thứ tự duyệt còn trong mã nguồn ==")
    src = open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
    n = f = 0
    for label, ok in [
        ("duyệt qua sorted(), không phải set trần",
         "for term in sorted(self._get_symptom_vocab(), key=lambda t: (-len(t), t)):" in src),
        ("KHÔNG còn dòng duyệt set trần",
         "for term in self._get_symptom_vocab():" not in src),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    return n, f


def main():
    if "--emit" in sys.argv:            # chế độ con: chỉ in kết quả censor để so byte
        sys.stdout.write(_censor())
        return
    tp = tf = 0
    for fn in (test_determinism, test_content, test_sort_in_source):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
