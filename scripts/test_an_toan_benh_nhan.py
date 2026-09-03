#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_an_toan_benh_nhan.py — Khóa các CỔNG AN TOÀN BỆNH NHÂN.

VÌ SAO CÓ FILE NÀY: ba lỗ dưới đây đều đã TÁI HIỆN SỐNG trên app thật, và trước đó KHÔNG file test
nào chạm tới chúng — nên "toàn xanh" hoàn toàn không bảo chứng được gì ở tầng an toàn.

  1) THAI KỲ (nặng nhất — dọa mất thai). Lời khai "thai 4 tháng, ốm nghén, buồn nôn, ăn không tiêu"
     -> bệnh danh Quỷ thai (chửa trứng) -> kê bài "Ích khí dưỡng huyết hoạt huyết HẠ THAI"
     (Ngưu tất, Ích mẫu), KHÔNG một chữ cảnh báo. Cổng CÓ tồn tại nhưng requires là token thai TRẦN
     ("có thai", "ốm nghén", "que thử thai"...) — điều kiện mà MỌI thai phụ đều thoả, nên cổng mất
     sạch tác dụng lọc. KB không sai: Quỷ thai đúng là chửa trứng và hạ thai đúng là pháp trị; sai
     ở TẦNG KHỚP.
  2) CẢNH BÁO CẤP CỨU NẰM SAU ĐƠN THUỐC. Hàm ghép trả `markdown + red_flag + disclaimer`, nên ca đột
     quỵ có đơn thuốc ở ký tự ~2025 còn dòng "gọi 115" ở ~2361. Nghịch lý: ba cảnh báo NHẸ hơn
     (mâu thuẫn giới, lời khai mỏng, hàn-nhiệt lẫn) đều đã được chèn lên ĐẦU.
  3) NHŨ NHI KHÔNG CÓ TUỔI. _infer_age chỉ hiểu dạng có chữ "tuổi", nên "trẻ sơ sinh", "trẻ 18
     tháng", "em bé 6 tháng" đều trả None -> cổng an toàn nhi im lặng với chính nhóm nhỏ nhất.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_an_toan_benh_nhan.py   (exit != 0 nếu fail)
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import logging  # noqa: E402
logging.disable(logging.INFO)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402


def chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


# ------------------------------------------------------------------ 1. cổng thai kỳ
_BARE_PREG = ("có thai", "mang thai", "thai nghén", "que thử thai", "ốm nghén", "chửa")


def _quy_thai_gate(gates):
    for g in gates:
        if any(n.lower() == "quỷ thai" for n in g.get("names", [])):
            return g
    return None


def test_cong_thai_ky():
    n = f = 0
    # Cổng phải khớp giữa NGUỒN SINH và FILE ĐÍCH. Sửa một nơi mà quên nơi kia thì lần ai đó chạy
    # generator, bản vá bị hoàn tác IM LẶNG (đúng bẫy đã ghi trong ghi chú dự án).
    d = json.load(io.open(os.path.join(ROOT, "data", "disease_gates.json"), encoding="utf-8"))
    g = _quy_thai_gate(d.get("gates", d))
    ok = g is not None
    n += chk("có cổng Quỷ thai trong disease_gates.json", ok)
    f += (not ok)
    if not ok:
        return n, f

    req = [r.lower() for r in g.get("requires", [])]
    bare = [t for t in _BARE_PREG if t in req]
    ok = not bare
    n += chk("cổng Quỷ thai KHÔNG đòi token thai TRẦN", ok, f"còn: {bare}" if bare else "")
    f += (not ok)

    ok = any("ra huyết" in r or "tim thai" in r or "bụng to" in r for r in req)
    n += chk("cổng Quỷ thai ĐÒI dấu đặc hiệu (ra huyết / tim thai / bụng to)", ok)
    f += (not ok)

    src = io.open(os.path.join(ROOT, "scripts", "build_disease_gates.py"), encoding="utf-8").read()
    i = src.find('"quỷ thai"')
    blk = src[i:i + 900] if i >= 0 else ""
    # Chỉ soi phần requires, KHÔNG soi chú thích — chính chú thích giải thích lỗi cũ có chứa nguyên
    # văn các token thai trần (đã dính đúng bẫy tự-bắt-chú-thích này nhiều lần trong dự án).
    m = re.search(r'"requires"\s*:\s*\[(.*?)\]', blk, re.S)
    req_gen = (m.group(1) if m else "")
    leftover = [t for t in _BARE_GEN if t in req_gen]
    ok = bool(m) and not leftover
    n += chk("generator build_disease_gates.py cũng đã vá (không hoàn tác được)", ok,
             f"còn: {leftover}" if leftover else "")
    f += (not ok)
    return n, f


_BARE_GEN = ('"có thai"', '"mang thai"', '"ốm nghén"', '"que thử thai"')


# ------------------------------------------------------------------ 2. thứ tự cảnh báo cấp cứu
def test_canh_bao_cap_cuu_dat_truoc():
    n = f = 0
    o = F.__new__(F)
    CA = "nam 62 tuổi, đột ngột méo miệng, liệt nửa người bên phải, nói khó, đau đầu dữ dội"
    than = ("### 1. Tổng quan\n- **Bệnh danh:** Trúng phong\n\n"
            "### 5. Pháp trị\n- Dùng bài **Trấn Can Tức Phong Thang**\n  - *Vị thuốc:* Ngưu tất, Đại giả thạch\n")
    out = F._append_medical_disclaimer(o, than, CA)

    i_warn = out.find("CẢNH BÁO KHẨN CẤP")
    i_bai = out.find("Vị thuốc:")
    ok = i_warn >= 0
    n += chk("ca đột quỵ có bật cảnh báo cấp cứu", ok)
    f += (not ok)

    ok = i_warn >= 0 and i_bai >= 0 and i_warn < i_bai
    n += chk("cảnh báo cấp cứu nằm TRƯỚC đơn thuốc", ok, f"cảnh báo@{i_warn} đơn@{i_bai}")
    f += (not ok)

    ok = out.lstrip().startswith(">")
    n += chk("cảnh báo là thứ ĐẦU TIÊN người bệnh đọc", ok, repr(out[:24]))
    f += (not ok)

    # Không bật bừa: lời khai lành phải KHÔNG có cảnh báo (bộ cờ đỏ hiện có 0 báo động giả — đó là
    # tài sản, mở rộng từ khóa ẩu sẽ biến cảnh báo thành nhiễu nền và người dùng bỏ qua cả lúc đúng).
    for lanh in ("nữ 30 tuổi, mệt mỏi, ăn kém, đại tiện lỏng",
                 "nam 40 tuổi, liệt dương, lưng gối mỏi",
                 "nữ 25 tuổi, đoản khí, hồi hộp nhẹ"):
        out2 = F._append_medical_disclaimer(o, than, lanh)
        ok = "CẢNH BÁO KHẨN CẤP" not in out2
        n += chk(f"KHÔNG báo động giả: {lanh[:34]}", ok)
        f += (not ok)

    ok = "Miễn trừ trách nhiệm" in out
    n += chk("miễn trừ trách nhiệm vẫn còn", ok)
    f += (not ok)
    return n, f


# ------------------------------------------------------------------ 3. suy tuổi nhũ nhi
def test_suy_tuoi_nhu_nhi():
    n = f = 0
    o = F.__new__(F)
    for t in ("trẻ sơ sinh", "sơ sinh bỏ bú", "nhũ nhi quấy khóc", "trẻ còn bú, tiêu chảy",
              "bé 8 tháng tuổi"):
        a = o._infer_age(t)
        ok = a is not None and a < 16
        n += chk(f"suy được tuổi nhũ nhi: {t}", ok, f"-> {a}")
        f += (not ok)

    # CHIỀU HỎNG NGƯỢC LẠI, NGUY HIỂM HƠN: người lớn bị gán tuổi nhũ nhi -> cổng nhi ẩn oan đơn
    # thuốc chính đáng của họ. Bốn ca đầu là PHẢN VÍ DỤ THẬT đã bắt được một bản vá hỏng: bản đó
    # bắt cả 'trẻ|bé|cháu|con' + 'N tháng', nhưng tiếng Việt không tách từ theo khoảng trắng nên
    # 'bé' khớp trong 'BÉo phì' và 'con' khớp trong 'sinh CON' -> sản phụ 32 tuổi bị gán 0 tuổi
    # DÙ '32 tuổi' nằm ngay trong câu. Giữ nguyên các ca này khi sửa _infer_age về sau.
    for t, mong in (("nữ 32 tuổi, sinh con 6 tháng trước, mất ngủ", 32),
                    ("béo phì 6 tháng rồi, ăn nhiều", None),
                    ("phụ nữ sau sinh con 3 tháng, ít sữa", None),
                    ("nữ 28 tuổi cho con bú, tắc tia sữa", 28),
                    ("nam 45 tuổi, đau dạ dày 3 tháng nay", 45),
                    ("nữ 30 tuổi, mất ngủ 6 tháng", 30),
                    ("ho 2 tháng nay", None),
                    ("kinh nguyệt không đều 5 tháng", None)):
        a = o._infer_age(t)
        ok = a == mong
        n += chk(f"KHÔNG gán oan tuổi nhũ nhi: {t[:38]}", ok, f"-> {a} (mong {mong})")
        f += (not ok)

    # Cổng tuổi phải kiểm 'is not None', KHÔNG được dùng `if age` — tuổi 0 (nhũ nhi) là giá trị
    # FALSY trong Python, dùng `if age` sẽ làm đúng nhóm nhỏ nhất mất bảo vệ mà không ai thấy.
    src = io.open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
    src_nc = re.sub(r"#.*", "", src)
    bad = re.findall(r"if\s+(?:self\.)?_patient_age\s*(?:and|:|\))", src_nc)
    ok = not bad
    n += chk("cổng tuổi không dùng `if age` (bẫy falsy khi tuổi = 0)", ok, str(bad[:2]))
    f += (not ok)
    return n, f


# ------------------------------------------------------------------ 4. bộ dò vị thuốc độc
def test_vi_thuoc_doc():
    n = f = 0
    o = F.__new__(F)

    # Thập Táo Thang: ba vị 有毒 cùng một thang, TRƯỚC ĐÂY IM LẶNG HOÀN TOÀN (bộ dò chỉ biết họ Ô
    # đầu + Ma hoàng).
    fl = F._herb_safety_flags("Cam toại, Nguyên hoa, Đại kích, Đại táo, Cam thảo")
    ok = len(fl["major"]) == 3
    n += chk("Thập Táo Thang: bắt đủ 3 vị độc mạnh", ok, str(fl["major"]))
    f += (not ok)
    ok = bool(fl["pairs"])
    n += chk("Thập Táo Thang: bắt cặp kỵ Cam thảo phản Cam toại/Đại kích/Nguyên hoa", ok)
    f += (not ok)

    ok = "Chu sa" in (F._herb_safety_flags("Chu sa, Hoàng liên")["major"] or [])
    n += chk("Chu sa (thuỷ ngân HgS) được cảnh báo", ok)
    f += (not ok)
    ok = "tích luỹ" in " ".join(
        F._herb_safety_extra_sentences(F._herb_safety_flags("Chu sa"))).lower()
    n += chk("cảnh báo Chu sa nêu độc TÍCH LUỸ (không chỉ 'có độc')", ok)
    f += (not ok)

    # GIẢ DANH — phải im lặng. Đây là lớp bẫy đã có tiền lệ thật trong dự án (_TOXIC_LOOKALIKE).
    for vi, cat in (("Địa phụ tử, Bạch tiễn bì", "aconite"),
                    ("Ma hoàng căn, Phù tiểu mạch", "ephedra")):
        ok = not F._herb_safety_flags(vi)[cat]
        n += chk(f"KHÔNG cảnh báo oan giả danh: {vi[:26]}", ok)
        f += (not ok)
    # 'Thổ bối mẫu' khác CHI với 'bối mẫu' -> không phải cặp kỵ với Phụ tử.
    ok = not F._herb_safety_flags("Phụ tử, Thổ bối mẫu")["pairs"]
    n += chk("KHÔNG báo cặp kỵ oan: Phụ tử + Thổ bối mẫu (khác chi)", ok)
    f += (not ok)
    # Nhưng bối mẫu THẬT thì phải bắt.
    ok = bool(F._herb_safety_flags("Phụ tử, Bối mẫu")["pairs"])
    n += chk("VẪN bắt cặp kỵ thật: Phụ tử + Bối mẫu", ok)
    f += (not ok)

    # Bán hạ TRẦN cố ý KHÔNG cảnh báo ở tầng vị: 143 dòng = 14% KB, và KB có 0 token 'bán hạ sống'
    # -> bán hạ trần trong thang thuốc Việt mặc định là dạng đã chế. Thêm vào sẽ đẩy mật độ cảnh
    # báo lên ~27% và biến cảnh báo thành nhiễu nền — mà "0 báo động giả" là tài sản phải giữ.
    ok = not F._herb_safety_flags("Bán hạ, Trần bì, Phục linh")["minor"]
    n += chk("bán hạ TRẦN không bị cảnh báo ở tầng vị (chống mệt cảnh báo)", ok)
    f += (not ok)
    ok = bool(F._herb_safety_flags("Sinh bán hạ, Trần bì")["minor"])
    n += chk("nhưng 'sinh bán hạ' (dạng SỐNG) thì CÓ", ok)
    f += (not ok)

    # Mật độ cảnh báo phải dưới ngưỡng nhiễu — trên ngưỡng thì người dùng bỏ qua cả lúc đúng.
    rows = list(csv.DictReader(io.open(os.path.join(ROOT, "data", "Medicine_clean.csv"),
                                       encoding="utf-8-sig")))
    hit = sum(1 for r in rows
              if any(F._herb_safety_flags(r.get("vị_thuốc") or "").values()))
    ti_le = 100.0 * hit / max(1, len(rows))
    ok = ti_le < 35.0
    n += chk("mật độ cảnh báo dưới ngưỡng nhiễu", ok, f"{ti_le:.1f}% ({hit}/{len(rows)} dòng)")
    f += (not ok)

    # IDEMPOTENT: chạy hai lần không được nhân đôi (cửa sổ dò phải cắt tới biên KHỐI, không phải
    # một số ký tự cố định — đã đo: cửa sổ 200 và 600 ký tự đều chèn lặp ở một số dòng).
    md = ("- Trị Bệnh **X** → Dùng bài **Thập Táo Thang**\n"
          "  - *Vị thuốc:* Cam toại, Nguyên hoa, Đại kích, Đại táo, Cam thảo\n")
    a = o._annotate_toxic_herb_lines(md, "")
    b = o._annotate_toxic_herb_lines(a, "")
    ok = a.count("An toàn dược") == b.count("An toàn dược") > 0
    n += chk("chèn cảnh báo IDEMPOTENT (chạy 2 lần không nhân đôi)", ok,
             f"{a.count('An toàn dược')} -> {b.count('An toàn dược')}")
    f += (not ok)
    return n, f


import csv  # noqa: E402


# ------------------------------------------------------------------ 6. Mục 1 không in bệnh danh 2 lần
def test_benh_danh_khong_in_lap():
    """Nhánh 'Bệnh danh' và 'Bệnh danh (tham khảo…)' là if/else — KHÔNG được cùng bắn.

    Đã xảy ra: một khối cảnh báo tuổi được chèn vào GIỮA nhánh if và nhánh else, khiến `else` bị
    gắn sang câu if của khối mới. Khi người bệnh CÓ khai tuổi thì else bắn, in thêm dòng
    'Bệnh danh (tham khảo — chưa trùng hội chứng cốt lõi)' ngay dưới dòng bệnh danh đã chốt: cùng
    một bệnh hiện HAI LẦN với hai nhãn mâu thuẫn. Python không báo lỗi vì cú pháp vẫn hợp lệ, và
    test cũ không bắt được vì chỉ kiểm sự CÓ MẶT của cảnh báo, không đếm số dòng bệnh danh.
    """
    n = f = 0
    src = io.open(os.path.join(ROOT, "src", "fusion_pipeline.py"), encoding="utf-8").read()
    # Hai nhánh phải nằm liền kề: giữa chúng chỉ được có phần thân của nhánh if.
    m = re.search(r'if disease_grounded:\s*\n(.*?)\n(\s*)else:\s*\n\s*final_markdown \+= \(\s*\n'
                  r'\s*f"- \*\*Bệnh danh \(tham khảo', src, re.S)
    ok = bool(m)
    n += chk("nhánh 'Bệnh danh' và 'Bệnh danh (tham khảo)' vẫn là if/else liền mạch", ok)
    f += (not ok)
    if m:
        than = m.group(1)
        ok = than.count("final_markdown +=") == 1
        n += chk("thân nhánh if chỉ in ĐÚNG một dòng bệnh danh", ok,
                 f"{than.count('final_markdown +=')} lệnh in")
        f += (not ok)
        ok = "_patient_age" not in than
        n += chk("khối cảnh báo tuổi KHÔNG nằm giữa if và else", ok)
        f += (not ok)
    return n, f


# ------------------------------------------------------------------ 5. cổng không được chặn chính nó
def test_cong_khong_chan_chinh_no():
    """Cổng bệnh danh phải NHẬN được chính dòng KB của bệnh nó bảo vệ.

    Ca thật đã tái hiện: lời khai chép nguyên văn dòng 'Phế ung × Giai đoạn vỡ mủ' (áp xe phổi vỡ
    mủ — ho ộc mủ máu, mùi tanh hôi, sốt) bị cổng CỦA CHÍNH PHẾ UNG loại, vì requires viết
    'ho ra mủ'/'đờm tanh'/'đau ngực' còn KB viết 'ho ÓI ra mủ máu'/'mùi tanh hôi'/'ngực đầy đau
    tức'. Hệ quả: chẩn thành 'Hầu tý (viêm yết hầu)' — sai hẳn tạng phủ, cho một bệnh nhiễm trùng
    phổi hoại tử.
    """
    n = f = 0
    o = F.__new__(F)
    F._load_csv_data(o)
    INP = ("ho ói ra mủ máu hoặc như nước cơm, khó thở không nằm được, ngực đầy đau tức, "
           "khát thích uống, mùi tanh hôi, sốt")
    ok = o._validate_disease_safety("Phế ung (áp xe phế)", [], INP)
    n += chk("Phế ung qua được cổng của CHÍNH NÓ (lời khai = dòng KB của nó)", ok)
    f += (not ok)

    # Đợt đổi bệnh danh đưa tên Tây y vào NGOẶC; cổng chặn-vô-điều-kiện khớp CHUỖI CON nên vẫn
    # dính và khoá vĩnh viễn 47 dòng KB. Các bệnh đã có bệnh danh Đông y KHÔNG được dính cổng đó.
    for b, sym in (("Cường giáp (Basedow)", "hồi hộp, sút cân, run tay, lồi mắt"),
                   ("Tâm quý (suy tim)", "hồi hộp trống ngực, khó thở, phù chân"),
                   ("Si ngốc (Alzheimer)", "hay quên, trí nhớ giảm sút, chậm chạp")):
        ok = o._validate_disease_safety(b, [], sym)
        n += chk(f"bệnh đã có tên Đông y KHÔNG bị chặn vô điều kiện: {b[:26]}", ok)
        f += (not ok)

    # NHƯNG tên Tây y THUẦN (chưa có bệnh danh Đông y) vẫn phải chặn — chẩn các bệnh này cần xét
    # nghiệm hiện đại, không đoán bằng vọng-vấn.
    for b in ("Bệnh bạch huyết", "Gout", "Loãng xương"):
        ok = not o._validate_disease_safety(b, [], "sốt cao, mệt mỏi, đau khớp, tiểu đỏ")
        n += chk(f"tên Tây y thuần VẪN bị chặn: {b}", ok)
        f += (not ok)

    # Cổng phải khớp giữa NGUỒN SINH và FILE ĐÍCH, nếu không lần chạy generator kế tiếp sẽ âm thầm
    # hoàn tác (đã xảy ra: JSON gỡ alzheimer/addison/parkinson nhưng generator vẫn giữ).
    gen = io.open(os.path.join(ROOT, "scripts", "build_disease_gates.py"), encoding="utf-8").read()
    gen_nc = re.sub(r"#.*", "", gen)
    ok = '"basedow"' not in gen_nc and '"alzheimer"' not in gen_nc
    n += chk("generator KHÔNG còn chặn vô điều kiện bệnh đã đổi tên", ok)
    f += (not ok)

    # Cờ đỏ ho ra máu: chuỗi cứng làm một chữ chen vào là trượt.
    for t in ("ho ói ra mủ máu hoặc như nước cơm", "ho khạc đờm lẫn máu tươi", "nôn ra mủ máu"):
        ok = any(rx.search(t) for rx in F._RED_FLAG_PATTERNS)
        n += chk(f"cờ đỏ bắt được ho ra máu: {t[:30]}", ok)
        f += (not ok)
    for t in ("ho, đại tiện ra máu", "ho, chảy máu cam", "ho khan kéo dài, đờm ít"):
        ok = not any(rx.search(t) for rx in F._RED_FLAG_PATTERNS)
        n += chk(f"KHÔNG báo động giả: {t[:30]}", ok)
        f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_cong_thai_ky, test_canh_bao_cap_cuu_dat_truoc, test_suy_tuoi_nhu_nhi,
               test_vi_thuoc_doc, test_cong_khong_chan_chinh_no,
               test_benh_danh_khong_in_lap):
        print(f"\n── {fn.__name__} ──")
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp - tf} PASS, {tf} FAIL / {tp}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
