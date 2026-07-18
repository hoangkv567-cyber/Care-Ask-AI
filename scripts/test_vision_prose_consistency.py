#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_vision_prose_consistency.py — Khóa 3 tầng NHẤT QUÁN VỌNG CHẨN <-> BIỆN LUẬN:

(1) _post_process_hallucinations — CỔNG THEO TỪNG CỤM: không được xóa chính triệu chứng bệnh nhân
    KHAI (bug thật: khai 'ợ chua' nhưng key nhóm là 'ợ hơi' -> xóa sạch 'ợ chua', để lại dấu câu
    treo ", –", và chủ chứng biến mất khỏi cả Mục 3 lẫn Mục 4). Vẫn phải CHẶN cụm bịa cùng nhóm
    ('ợ nước' = Vị hàn ẩm, trái cực với 'ợ chua').
(2) _strip_fabricated_tongue_pallor — gỡ 'lưỡi nhợt' BỊA khi vọng chẩn đọc sắc lưỡi SINH LÝ, nhưng
    KHÔNG được đụng lưỡi nhợt THẬT, KHÔNG nhầm 'lưỡi hồng nhạt', KHÔNG cuốn theo bằng chứng NHIỆT.
(3) _tidy_dangling_punctuation — dọn dấu treo, giữ nguyên gạch dài phụ chú / số thập phân / markdown.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_vision_prose_consistency.py   (exit != 0 nếu fail)
Không cần Neo4j / LLM. NÊN chạy đa seed (hàm (1) duyệt set vocab):
    for s in 0 1 2 3; do PYTHONHASHSEED=$s python scripts/test_vision_prose_consistency.py; done
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as F

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


def test_censor_gate():
    """Cổng kiểm duyệt phải phân biệt CỤM bệnh nhân khai vs CỤM bịa cùng nhóm."""
    o = pipe()
    print("== (1) _post_process_hallucinations — cổng theo TỪNG CỤM ==")
    S = "chướng bụng, tiêu lỏng, buồn nôn, đau bụng, mệt mỏi, ợ chua"
    n = f = 0
    # --- PHẢI GIỮ: chính lời khai của bệnh nhân ---
    keep = [
        ("Vị mất hòa giáng khiến Vị khí thượng nghịch, gây ra ợ chua và buồn nôn.", ["ợ chua", "buồn nôn"], S),
        ("Tỳ hư không vận hóa, thấp trọc nội sinh gây buồn nôn và chướng bụng.", ["buồn nôn"], S),
        ("Khí Vị không giáng xuống bình thường, gây ợ chua – biểu hiện của khí Vị thượng nghịch.", ["ợ chua"], S),
        ("Âm hư nội nhiệt gây đạo hãn về đêm.", ["đạo hãn"], "đạo hãn, mệt mỏi"),
    ]
    for text, musts, sym in keep:
        out = o._post_process_hallucinations(text, sym)
        ok = all(m in out for m in musts)
        n += _chk(f"GIỮ {musts} — {text[:44]}...", ok, repr(out[:78]))
        f += (not ok)
    # --- PHẢI CHẶN: cụm KHÔNG khai (kể cả cùng nhóm với cụm đã khai) ---
    block = [
        ("Can khí phạm Vị gây ợ nước trong.", "ợ nước", S),          # trái cực với 'ợ chua'
        ("Nhiệt độc uất kết gây mụn viêm.", "mụn viêm", S),
        ("Âm hư gây mồ hôi trộm về đêm.", "mồ hôi trộm", S),
        ("Can huyết hư gây chóng mặt.", "chóng mặt", S),
    ]
    for text, bad, sym in block:
        out = o._post_process_hallucinations(text, sym)
        ok = bad not in out
        n += _chk(f"CHẶN '{bad}'", ok, repr(out[:70]))
        f += (not ok)
    # --- KHÔNG để lại TỪ MỒ CÔI (khai 'đạo hãn' -> gỡ 'mồ hôi trộm' nguyên cụm, không rớt 'trộm') ---
    out = o._post_process_hallucinations("Âm hư sinh nội nhiệt gây mồ hôi trộm về đêm.", "đạo hãn, mệt mỏi")
    ok = "trộm" not in out
    n += _chk("không rớt từ mồ côi 'trộm'", ok, repr(out[:70]))
    f += (not ok)
    return n, f


def test_tongue_pallor():
    """Gỡ 'lưỡi nhợt' bịa — nhưng phải chừa lưỡi nhợt THẬT, sắc lưỡi SINH LÝ và bằng chứng NHIỆT."""
    o = pipe()
    print("\n== (2) _strip_fabricated_tongue_pallor ==")
    n = f = 0
    NO_TONGUE = "chướng bụng, tiêu lỏng, buồn nôn, đau bụng, mệt mỏi, ợ chua, rêu trắng mỏng, lưỡi bệu"

    # FIRE: câu bịa lưỡi nhợt (ca thật đã gặp)
    src = ("Khí hư không đủ để vận hành huyết, khiến huyết không thể vinh nhuận đầy đủ lên bề mặt "
           "lưỡi, làm lưỡi trở nên nhợt nhạt, thiếu sức sống, kết hợp với thủy thấp ứ đọng tạo "
           "thành hiện tượng lưỡi bệu.")
    out = o._strip_fabricated_tongue_pallor(src, NO_TONGUE)
    ok = ("nhợt nhạt" not in out) and out.strip() != ""
    n += _chk("FIRE: gỡ 'lưỡi trở nên nhợt nhạt' bịa", ok, repr(out[-60:]))
    f += (not ok)

    # NO-FIRE: lưỡi nhợt THẬT trong lời khai
    real = "chướng bụng, mệt mỏi, lưỡi nhợt"
    out = o._strip_fabricated_tongue_pallor("Huyết hư khiến lưỡi nhợt, mạch tế.", real)
    ok = "lưỡi nhợt" in out
    n += _chk("NO-FIRE: lưỡi nhợt THẬT được giữ", ok, repr(out))
    f += (not ok)

    # NO-FIRE: sắc lưỡi SINH LÝ không được coi là nhợt
    for phys in ["Thân lưỡi hồng nhạt, rêu trắng mỏng nhuận.",
                 "Chất lưỡi hồng nhạt cho thấy khí huyết còn đủ.",
                 "Lưỡi nhạt hồng, rêu mỏng.",
                 "Chất lưỡi đạm hồng."]:
        out = o._strip_fabricated_tongue_pallor(phys, NO_TONGUE)
        ok = out.strip() == phys.strip()
        n += _chk(f"NO-FIRE sinh lý: {phys[:38]}...", ok, repr(out[:60]))
        f += (not ok)

    # NO-FIRE: dấu NHIỆT không được cuốn theo (chống lớp lỗi 'mù nhiệt')
    heat = ("Lưỡi nhợt kèm rêu đã ngả vàng và tiểu vàng sẫm cho thấy tà đã bắt đầu hóa nhiệt nhập lý.")
    out = o._strip_fabricated_tongue_pallor(heat, NO_TONGUE)
    ok = ("rêu" in out and "vàng" in out) or "hóa nhiệt" in out
    n += _chk("NO-FIRE: giữ bằng chứng NHIỆT cùng câu", ok, repr(out[:80]))
    f += (not ok)

    # NO-FIRE: câu sách giáo khoa (hedge)
    hedge = "Thể Tỳ khí hư thường gặp lưỡi nhợt, rêu trắng mỏng và mạch nhược."
    out = o._strip_fabricated_tongue_pallor(hedge, NO_TONGUE)
    ok = out.strip() == hedge.strip()
    n += _chk("NO-FIRE: câu 'thường gặp...' giữ nguyên", ok, repr(out[:70]))
    f += (not ok)

    # Câu CHỈ nói về cụm lưỡi-nhợt bịa -> bỏ CẢ CÂU, không để lại mệnh đề MẤT CHỦ NGỮ
    # ("Lưỡi nhợt phản ánh huyết hư." KHÔNG được thành "Phản ánh huyết hư.")
    for stub in ["Lưỡi nhợt nhạt phản ánh huyết hư không vinh nhuận.",
                 "Lưỡi nhợt là dấu hiệu của khí huyết đều hư tổn."]:
        out = o._strip_fabricated_tongue_pallor(stub, NO_TONGUE)
        ok = out.strip() == ""
        n += _chk(f"bỏ cả câu, không để mảnh cụt: {stub[:36]}...", ok, repr(out))
        f += (not ok)
    # ...nhưng câu CHỞ dấu chẩn đoán khác thì phải GIỮ phần đó (chống 'mù nhiệt')
    for keepsrc, must in [("Lưỡi nhợt trong khi rêu vàng dày cho thấy tà hóa nhiệt.", "vàng"),
                          ("Lưỡi nhợt, mạch trầm tế, người sợ lạnh.", "mạch")]:
        out = o._strip_fabricated_tongue_pallor(keepsrc, NO_TONGUE)
        ok = must in out
        n += _chk(f"giữ bằng chứng '{must}' trong câu còn lại", ok, repr(out))
        f += (not ok)

    # KHÔNG lấn domain SẮC MẶT
    face = "Khí huyết hư khiến sắc mặt nhợt nhạt."
    out = o._strip_fabricated_tongue_pallor(face, NO_TONGUE)
    ok = out.strip() == face.strip()
    n += _chk("NO-FIRE: không đụng câu SẮC MẶT", ok, repr(out))
    f += (not ok)
    return n, f


def test_punct():
    """Dọn dấu treo, giữ nguyên dấu hợp lệ."""
    o = pipe()
    print("\n== (3) _tidy_dangling_punctuation ==")
    n = f = 0
    for src, want_gone in [
        ("Khí Vị không giáng xuống bình thường, – biểu hiện của khí Vị thượng nghịch.", ", –"),
        ("Tỳ không vận hóa được thủy thấp, — nguồn gốc của tiêu lỏng.", ", —"),
        ("Chính khí suy ,  – tà khí thừa cơ xâm nhập.", " ,"),
        ("Tỳ khí hư,, gây tiêu lỏng.", ",,"),
    ]:
        out = o._tidy_dangling_punctuation(src)
        ok = want_gone not in out
        n += _chk(f"DỌN '{want_gone}'", ok, repr(out[:70]))
        f += (not ok)
    for keep in [
        "Đại tiện xong đỡ đau — dấu chỉ điểm của thực tích.",
        "Liều dùng 1,5g mỗi thang, sắc uống ấm.",
        "- Trị Bệnh **Viêm đại tràng** — *Bản – gốc bệnh* (Hội chứng Tỳ khí hư)",
        "Bệnh nhân mệt mỏi, chướng bụng, tiêu lỏng.",
    ]:
        out = o._tidy_dangling_punctuation(keep)
        ok = out == keep
        n += _chk(f"GIỮ NGUYÊN: {keep[:44]}...", ok, repr(out[:70]))
        f += (not ok)
    return n, f


def test_o_chua_patch():
    """'ợ chua' bị LLM bỏ sót PHẢI được vá vào Mục 3 với cơ chế đúng luật cứng (VỊ mất hòa giáng),
    và KHÔNG được làm Mục 4 mọc phần Tiêu Thực trái với Bát Cương 'Lý - Hư'."""
    o = pipe()
    print("\n== (4) vá 'ợ chua' bị bỏ sót ==")
    n = f = 0
    S = "chướng bụng, tiêu lỏng, buồn nôn, đau bụng, mệt mỏi, ợ chua"
    md = ("### 3. Phân tích Cơ chế Gốc (Bản Hư)\n"
          "- Tỳ khí hư khiến vận hóa kém, gây chướng bụng và tiêu lỏng, người mệt mỏi.\n\n"
          "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)\n"
          "- Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy.\n")
    out = o._tidy_dangling_punctuation(
        o._sync_tieu_thuc_with_bat_cuong(o._patch_missing_symptoms(md, S), "Lý - Hư", S))
    m3, m4 = out.split("### 4.")[0], out.split("### 4.")[1]
    for label, ok in [
        ("'ợ chua' được giải thích ở Mục 3", "ợ chua" in m3),
        ("cơ chế đúng luật: VỊ mất hòa giáng", "Vị mất hòa giáng" in m3),
        ("KHÔNG quy cho Tỳ khí nghịch (cấm)", "Tỳ khí nghịch" not in out and "khí Tỳ không thể hành xuống" not in out),
        ("Mục 4 vẫn 'Không có Tiêu Thực' (khớp Bát Cương Lý-Hư)", "Không có Tiêu Thực" in m4),
        ("không lặp liên từ 'Ngoài ra,'", out.count("Ngoài ra,") <= 1),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_censor_gate, test_tongue_pallor, test_punct, test_o_chua_patch):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
