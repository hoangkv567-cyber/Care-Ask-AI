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

    # FIRE: cụm nhợt cách từ chỉ lưỡi bởi CẢ MỘT MỆNH ĐỀ (bản đầu chỉ bắt dạng liền kề -> lọt)
    src2 = ("Lưỡi là nơi phản ánh tạng phủ, huyết hư không đủ nuôi dưỡng thân lưỡi, khiến lưỡi bệu "
            "(sưng to), đồng thời huyết hư làm chất lưỡi không đủ vinh nhuận, nhạt, rêu trắng mỏng.")
    out = o._strip_fabricated_tongue_pallor(src2, NO_TONGUE)
    ok = out != src2 and "vinh nhuận, nhạt" not in out
    n += _chk("FIRE: 'chất lưỡi <mệnh đề>, nhạt' (cách xa) cũng bị gỡ", ok, repr(out[-58:]))
    f += (not ok)
    # ...nhưng khoảng chen KHÔNG được vắt sang BỘ PHẬN KHÁC (rêu/mặt/mạch) rồi gỡ nhầm
    cross = "Lưỡi hồng nhạt, rêu trắng mỏng, sắc mặt nhợt nhạt do khí huyết hư."
    ok = o._strip_fabricated_tongue_pallor(cross, NO_TONGUE) == cross
    n += _chk("NO-FIRE: không vắt qua 'rêu/sắc mặt' để gỡ nhầm", ok)
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


def test_acute_onset_caution():
    """Trường 'Thời gian mắc' PHẢI có tác dụng: bệnh mới phát + cốt lõi thể hư (vốn mạn) + dấu thực
    trệ -> cảnh báo cân nhắc thương thực. Trước bản vá, trường này bị bỏ qua hoàn toàn."""
    o = pipe()
    print("\n== (5) onset CẤP vs cốt lõi HƯ MẠN ==")
    n = f = 0
    MD = ("### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)\n"
          "- Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy.\n")
    acute = ("22 tuổi, nữ giới, bệnh vài ngày, chướng bụng, tiêu lỏng, buồn nôn, "
             "đau bụng, mệt mỏi, ợ chua")
    out = o._annotate_acute_onset_caution(MD, "Tỳ khí hư", acute)
    for label, ok in [
        ("FIRE: có cảnh báo thực trệ", "THỰC TRỆ" in out),
        ("nêu đúng dấu căn cứ (ợ chua)", "ợ chua" in out),
        ("KHÔNG đổi câu gốc của Mục 4", "Không có Tiêu Thực" in out),
    ]:
        n += _chk(label, ok)
        f += (not ok)
    # Nhánh NGOẠI CẢM: ố hàn + >=1 dấu biểu khác, onset cấp, core hư
    ngoai_cam = ("22 tuổi, nữ giới, bệnh mới mắc, sợ lạnh, mồ hôi trộm, ăn uống kém, "
                 "người nặng, chóng mặt, hoa mắt, mất ngủ, đau đầu, ăn kém")
    out2 = o._annotate_acute_onset_caution(MD, "Khí huyết hư", ngoai_cam)
    import re as _re
    _m = _re.search(r"biểu \(([^)]*)\)", out2)
    for label, ok in [
        ("FIRE: cảnh báo NGOẠI CẢM", "NGOẠI CẢM" in out2),
        ("khuyên giải biểu trước khi bổ", "giải biểu" in out2),
        # 'ho' là chuỗi con của 'hoa mắt' -> phải khớp theo RANH GIỚI TỪ, không được nêu 'ho'
        ("dẫn chứng KHÔNG khớp nhầm 'ho' trong 'hoa mắt'", bool(_m) and ", ho" not in _m.group(1)),
    ]:
        n += _chk(label, ok, (_m.group(1) if _m else ""))
        f += (not ok)
    # ...nhưng 'ho' THẬT vẫn phải bắt được
    out3 = o._annotate_acute_onset_caution(MD, "Phế khí hư", "bệnh mới mắc, sợ lạnh, ho, đau họng")
    ok = "NGOẠI CẢM" in out3
    n += _chk("'ho' thật vẫn kích hoạt được", ok)
    f += (not ok)

    # NO-FIRE: mọi điều kiện thiếu -> giữ NGUYÊN VĂN
    for label, pri, s in [
        ("mạn tính", "Tỳ khí hư", "bệnh mạn tính lâu ngày, chướng bụng, ợ chua, mệt mỏi"),
        ("không khai thời gian", "Tỳ khí hư", "chướng bụng, ợ chua, mệt mỏi"),
        ("cốt lõi là THỰC", "Thấp nhiệt", "bệnh vài ngày, chướng bụng, ợ chua"),
        ("cấp nhưng <2 dấu thực trệ", "Tỳ khí hư", "bệnh vài ngày, mệt mỏi, tiêu lỏng"),
        # 'sợ lạnh' ĐƠN ĐỘC không đủ (cũng là dấu dương hư nội thương)
        ("chỉ sợ lạnh, không dấu biểu khác", "Tỳ khí hư", "bệnh mới mắc, sợ lạnh, mệt mỏi"),
        # cốt lõi ĐÃ là ngoại cảm biểu -> đang đúng rồi, không cảnh báo
        ("cốt lõi đã là ngoại cảm", "Phong hàn phạm biểu", "bệnh mới mắc, sợ lạnh, đau đầu"),
    ]:
        ok = o._annotate_acute_onset_caution(MD, pri, s) == MD
        n += _chk(f"NO-FIRE: {label}", ok)
        f += (not ok)
    return n, f


def test_no_midsentence_gutting():
    """Cổng kiểm duyệt CHỈ được gỡ triệu chứng bịa khi nó đứng ĐẦU câu/mệnh đề.
    Bug thật: "Rêu trắng mỏng trên lưỡi nhợt là dấu hiệu của Tỳ dương hư. Mất ngủ xuất phát..."
    -> "Rêu trắng mỏng trên Mất ngủ xuất phát..." (nuốt đuôi, dính liền câu sau)."""
    o = pipe()
    print("\n== (6) không nuốt GIỮA CÂU ==")
    n = f = 0
    S = "đại tiện lỏng, khó thở, mất ngủ, mệt mỏi, sợ lạnh, chán ăn, ăn kém, rêu trắng mỏng"
    for src in ["Rêu trắng mỏng trên lưỡi nhợt là dấu hiệu của Tỳ dương hư. Mất ngủ xuất phát từ Tâm thần bất an.",
                "Rêu trắng mỏng trên lưỡi bệu là biểu hiện của thủy thấp. Mất ngủ xuất phát từ Tâm thần bất an."]:
        out = o._post_process_hallucinations(src, S)
        ok = ("trên Mất ngủ" not in out) and ("trên bề mặt Mất" not in out) and ("Mất ngủ" in out)
        n += _chk("không dính đầu câu vào câu sau", ok, repr(out[:82]))
        f += (not ok)
    # Năng lực cũ phải còn: triệu chứng bịa ĐẦU CÂU vẫn bị gỡ
    for src, bad in [("Lưỡi nhợt cũng là dấu hiệu của huyết hư. Tỳ khí hư gây mệt mỏi.", "Lưỡi nhợt"),
                     ("Tỳ khí hư gây mệt mỏi. Lưỡi bệu là biểu hiện của thủy thấp ứ đọng.", "Lưỡi bệu")]:
        out = o._post_process_hallucinations(src, S)
        ok = bad not in out
        n += _chk(f"vẫn gỡ được '{bad}' ở đầu câu", ok, repr(out[:70]))
        f += (not ok)
    # Chuỗi đầy đủ: không còn giới từ treo ('...mỏng trên là dấu hiệu...')
    src = "Rêu trắng mỏng trên lưỡi nhợt là dấu hiệu của Tỳ dương hư. Mất ngủ xuất phát từ Tâm thần bất an."
    out = o._tidy_dangling_punctuation(
        o._strip_fabricated_tongue_pallor(o._post_process_hallucinations(src, S), S))
    for label, ok in [("hết cụm 'lưỡi nhợt' bịa", "lưỡi nhợt" not in out.lower()),
                      ("hết giới từ treo 'trên là'", "trên là" not in out),
                      ("giữ được câu 'Mất ngủ'", "Mất ngủ" in out)]:
        n += _chk(label, ok, repr(out[:78]))
        f += (not ok)
    return n, f


def test_semen_fluid_term():
    """'tinh dịch' (dịch sinh dục nam) dùng nhầm cho 'tân dịch' khi giải thích mồ hôi.
    CHỈ sửa khi bệnh nhân NỮ — với nam, 'tinh dịch' có thể ĐÚNG (di tinh/hoạt tinh, 39 dòng KB)."""
    o = pipe()
    print("\n== (7) thuật ngữ tinh dịch / tân dịch ==")
    n = f = 0
    SWEAT = "Khí hư không đủ để giữ chặt tinh dịch, khiến mồ hôi trộm xuất hiện khi ngủ."
    MALE = "Thận hư không cố nhiếp được tinh dịch gây di tinh, hoạt tinh."
    _old = getattr(o, "_patient_sex", None)
    try:
        o._patient_sex = "nu"
        r = o._fix_semen_fluid_term(SWEAT)
        ok = "tân dịch" in r and "tinh dịch" not in r
        n += _chk("NỮ + câu mồ hôi -> đổi sang 'tân dịch'", ok, repr(r[:56]))
        f += (not ok)
        ok = o._fix_semen_fluid_term(MALE) == MALE
        n += _chk("NỮ nhưng câu KHÔNG về mồ hôi -> giữ nguyên", ok)
        f += (not ok)
        o._patient_sex = "nam"
        for label, src in [("NAM + di tinh -> giữ", MALE), ("NAM + mồ hôi -> vẫn giữ", SWEAT)]:
            ok = o._fix_semen_fluid_term(src) == src
            n += _chk(label, ok)
            f += (not ok)
        o._patient_sex = None
        ok = o._fix_semen_fluid_term(SWEAT) == SWEAT
        n += _chk("KHÔNG biết giới -> không đoán", ok)
        f += (not ok)
    finally:
        o._patient_sex = _old
    return n, f


def test_formula_line_dedup():
    """Hai dòng CÙNG một bài chỉ khác tên viết tắt của một vị -> chỉ in MỘT."""
    o = pipe()
    print("\n== (8) khử trùng dòng bài Mục 5 ==")
    n = f = 0
    for a, b, exp, tag in [
        (frozenset(["ngũ vị tử", "cam thảo"]), frozenset(["ngũ vị", "cam thảo"]), True, "tên viết tắt"),
        (frozenset(["thổ phục (linh)", "phụ tử"]), frozenset(["thổ phục", "phụ tử"]), True, "tắt có ngoặc"),
        (frozenset(["bạch truật", "cam thảo"]), frozenset(["bạch thược", "cam thảo"]), False, "KHÁC vị"),
        (frozenset(["a", "b"]), frozenset(["a", "b", "c"]), False, "khác số vị"),
    ]:
        ok = F._herb_sets_equivalent(a, b) == exp
        n += _chk(f"tương đương bộ vị: {tag}", ok)
        f += (not ok)
    L = ["- Trị Bệnh **Ách nghịch** — *Bản* (Hội chứng X) → Dùng bài **Ôn thận nạp khí phương**\n"
         "  - *Vị thuốc:* Đẳng sâm, Cam thảo, Ngũ vị tử\n",
         "- Trị Bệnh **Ách nghịch** — *Bản* (Hội chứng X) → Dùng bài theo pháp **Ôn thận nạp khí**\n"
         "  - *Vị thuốc:* Đẳng sâm, Cam thảo, Ngũ vị\n"]
    out = o._dedup_formula_lines(L)
    for label, ok in [("2 dòng cùng bài -> còn 1", len(out) == 1),
                      ("giữ bản có PHƯƠNG DANH", bool(out) and "phương" in out[0])]:
        n += _chk(label, ok)
        f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_censor_gate, test_tongue_pallor, test_punct, test_o_chua_patch,
               test_acute_onset_caution, test_no_midsentence_gutting,
               test_semen_fluid_term, test_formula_line_dedup):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
