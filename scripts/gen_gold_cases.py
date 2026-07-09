#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/gen_gold_cases.py — Sinh khung bộ ca chuẩn (gold) từ CSV cho tầng khớp bệnh danh.

Tạo data/gold_cases.json gồm:
  - Ca 'recall': lấy triệu chứng QUAN SÁT ĐƯỢC (bỏ field mạch — hệ không bắt mạch) của một bệnh
    làm đầu vào, kỳ vọng bệnh đó nằm trong top-3 ứng viên. Trải đều theo chuyên khoa.
  - (Ca 'exclude' phân biệt được soạn tay, thêm trực tiếp vào JSON sau khi sinh.)

Đây là KHUNG máy sinh — bác sĩ Đông y nên rà & bổ sung. Chạy 1 lần để tạo file, rồi chỉnh tay.
Chạy:  python scripts/gen_gold_cases.py
"""
import os
import sys
import csv
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F

CSV_PATH = "data/Medicine_clean.csv"
OUT_PATH = "data/gold_cases.json"

# Bệnh đại diện theo chuyên khoa (tên phải khớp cột tên_bệnh trong CSV)
REPRESENTATIVE = [
    # Hô hấp / họng-mũi
    "Khái thấu", "Viêm yết hầu", "Tỵ nục", "Mất tiếng",
    # Tim mạch
    "Suy tim", "Tâm quý",
    # Tiêu hóa
    "Ách nghịch", "Ẩu thổ", "Phúc thống", "Vị quản thống",
    # Phụ khoa
    "Bế kinh", "Thống kinh",
    # Da liễu
    "Ngân tiết", "Đới trạng bào chẩn",
    # Tiết niệu / nam khoa
    "Di niệu", "Dương nuy",
    # Tai / mắt
    "Dạ manh",
    # Thần kinh / đau đầu
    "Đầu thống", "Thiên đầu thống", "Trúng phong", "Động kinh", "Huyễn vựng",
    # Chuyển hóa / toàn thân
    "Tiêu khát", "Hư lao", "Béo phì",
    # Cơ xương khớp / cảm giác
    "Tê bì tứ chi",
    # Mồ hôi
    "Đạo hãn", "Thủ túc xuất hãn",
    # Tiêu hóa dưới
    "Tiết tả" if False else "Ẩu thổ",  # placeholder tránh trùng — bỏ qua nếu không có
]


# Ca EXCLUDE (phân biệt) soạn tay từ các lỗi đã biết (lịch sử commit + test lâm sàng). Mỗi ca: bệnh
# trong 'forbid' TUYỆT ĐỐI không được xuất hiện trong ứng viên. 'expect' (nếu có) phải xuất hiện.
MANUAL_CASES = [
    {"specialty": "hô hấp", "raw": "ho nhiều, khó thở, mệt mỏi",
     "symptoms": ["ho nhiều", "khó thở", "mệt mỏi", "rêu trắng", "lưỡi hồng", "mặt nhợt nhạt"],
     "forbid": ["Áp xe gan", "Bạch tiển (nấm da)", "Dương nuy", "Tĩnh mạch viêm tắc", "Trưng hà", "Long bế"],
     "expect": ["Mạn tính tắc nghẽn phế bệnh"]},
    {"specialty": "tim mạch", "raw": "đau ngực dữ dội, tức ngực, khó thở, hồi hộp",
     "symptoms": ["đau thắt ngực", "tức ngực", "khó thở", "hồi hộp", "mệt mỏi"],
     "forbid": [], "expect": ["Hung tý"]},
    {"specialty": "nam khoa", "raw": "liệt dương, yếu sinh lý, đau lưng",
     "symptoms": ["liệt dương", "đau lưng mỏi gối", "tiểu đêm", "mệt mỏi"],
     "forbid": [], "expect": ["Dương nuy"]},
    {"specialty": "sản khoa", "raw": "đang mang thai, đau bụng, ra huyết âm đạo",
     "symptoms": ["động thai", "đau bụng", "ra huyết"],
     "forbid": [], "expect": ["Động thai"]},
    {"specialty": "phụ khoa", "raw": "bụng dưới đau lạnh, bế kinh, tay chân lạnh",
     "symptoms": ["bụng dưới đau lạnh", "tay chân lạnh", "bế kinh"],
     "forbid": ["Quỷ thai"], "expect": []},
    {"specialty": "tiêu hóa", "raw": "tiêu chảy, phân lỏng nát, sợ lạnh, tay chân lạnh",
     "symptoms": ["tiêu chảy", "phân lỏng nát", "sợ lạnh", "tay chân lạnh", "đau lưng", "đau bụng"],
     "forbid": ["Tâm nhồi máu", "Nguyệt kinh trì kỳ"], "expect": []},
    {"specialty": "tiêu hóa", "raw": "nấc, ưa nóng, miệng nhạt không khát",
     "symptoms": ["nấc", "ưa nóng", "miệng nhạt không khát", "rêu trắng nhuận"],
     "forbid": ["Thiên đầu thống"], "expect": ["Ách nghịch"]},
    {"specialty": "cảm mạo", "raw": "sợ lạnh, sổ mũi, sốt, ho",
     "symptoms": ["sợ lạnh", "sổ mũi", "sốt", "ho", "mặt nhợt nhạt", "lưỡi hồng nhạt", "rêu trắng"],
     "forbid": ["Tâm quý", "Ngân tiết", "Phong chẩn", "Nhũ ung", "Áp xe phế"], "expect": []},
    {"specialty": "khí hư đàm thấp", "raw": "mệt mỏi, đầy bụng, lưỡi bệu, rêu nhớt",
     "symptoms": ["mệt mỏi", "đầy bụng", "lưỡi bệu", "rêu lưỡi trắng nhớt", "rìa lưỡi có hằn răng"],
     "forbid": ["Béo phì", "Khái huyết"], "expect": []},
    {"specialty": "huyết hư", "raw": "chóng mặt, hoa mắt, hồi hộp, mất ngủ, sắc mặt nhợt",
     "symptoms": ["chóng mặt", "hoa mắt", "hồi hộp", "mất ngủ", "mặt nhợt nhạt", "lưỡi nhợt"],
     "forbid": ["Bạo manh", "Canh niên kỳ hội chứng", "Dạ manh"], "expect": []},
    {"specialty": "tỳ khí hư", "raw": "mệt mỏi, ăn kém, đại tiện lỏng, khó thở",
     "symptoms": ["mệt mỏi", "ăn kém", "đại tiện lỏng", "khó thở", "rìa lưỡi có hằn răng", "lưỡi nhợt"],
     "forbid": ["Tỵ cứu", "Khẩu sang", "Sán khí", "Tử cung hạ sa", "Tê bì tứ chi"], "expect": []},
    {"specialty": "đau họng ho khan", "raw": "đau họng, ho khan, họng khô",
     "symptoms": ["đau họng", "ho khan", "họng khô", "lưỡi đỏ", "rêu vàng mỏng"],
     "forbid": ["Béo phì", "Tĩnh mạch viêm tắc"], "expect": []},
    # Ca tỳ vị hư hàn (sợ lạnh + tay chân lạnh + đại tiện lỏng + buồn nôn), KHÔNG một triệu chứng
    # kinh nguyệt nào -> tuyệt đối không được rò bệnh danh phụ khoa 'Nguyệt kinh trì kỳ' (thể Hàn của
    # nó khớp sợ lạnh/tay chân lạnh/đau bụng). Tránh syllable 'kinh' trong raw để không kích cổng
    # phụ khoa nhầm chiều. Kỳ vọng ứng viên tiêu hóa (Viêm đại tràng...).
    {"specialty": "tỳ vị hư hàn", "raw": "tay chân lạnh, đại tiện lỏng, buồn nôn, đau bụng, sợ lạnh, ợ hơi",
     "symptoms": ["tay chân lạnh", "đại tiện lỏng", "buồn nôn", "đau bụng", "mất ngủ", "sợ lạnh",
                  "ợ hơi", "quầng đen dưới mắt", "rêu trắng mỏng"],
     "forbid": ["Nguyệt kinh trì kỳ", "Nguyệt kinh tiên kỳ", "Kinh nguyệt rối loạn"],
     "expect": ["Viêm đại tràng"]},
    # Ca dương hư (mệt + tay chân lạnh + sợ lạnh + đại tiện lỏng) kèm nước tiểu vàng (cô đặc). TUYỆT
    # ĐỐI không được dán bệnh mạch nặng 'Động mạch viêm tắc' (chỉ vì tay chân lạnh — không hề đau
    # chi/đi đau nghỉ đỡ/tím-hoại tử đầu chi) hay 'Hoàng đản' (chỉ vì tiểu vàng — không hề vàng
    # da/mắt). Đây là ca gốc lỗi dán bệnh Tây y nặng cho người 22 tuổi.
    {"specialty": "dương hư (tiểu vàng cô đặc)", "raw": "mệt mỏi, tay chân lạnh, sợ lạnh, đại tiện lỏng",
     "symptoms": ["nước tiểu vàng", "tay chân lạnh", "đại tiện lỏng", "mệt mỏi", "sợ lạnh",
                  "rêu trắng mỏng", "quầng đen dưới mắt"],
     "forbid": ["Động mạch viêm tắc", "Hoàng đản", "Khí hư"], "expect": []},
    # Ca cảm mạo/ho khan thường (chảy nước mũi + khô họng + ho khan) — TUYỆT ĐỐI không được dán bệnh
    # nhiễm 'Bách nhật khái' (ho gà) chỉ vì có 'ho': ho gà đòi hỏi cơn ho rũ rượi + tiếng rít + ho
    # liên tục kéo dài, đây chỉ ho khan cấp.
    {"specialty": "cảm mạo ho khan", "raw": "chảy nước mũi, khô họng, ho khan",
     "symptoms": ["chảy nước mũi", "khô họng", "ho khan", "quầng đen dưới mắt", "rêu trắng mỏng"],
     "forbid": ["Bách nhật khái"], "expect": []},
]


def _observable_fields(trieu_chung: str):
    """Các field quan sát được (bỏ field mạch tượng) — đúng phạm vi vấn+vọng."""
    fields = [s.strip() for s in trieu_chung.split(",") if s.strip()]
    return [f for f in fields if not F._is_pulse_field(f.lower())]


def run():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    by_disease = {}
    for r in rows:
        by_disease.setdefault(r["tên_bệnh"].strip(), []).append(r)

    cases = []
    seen = set()
    for name in REPRESENTATIVE:
        if name in seen or name not in by_disease:
            continue
        seen.add(name)
        # Chọn row có NHIỀU field quan sát được nhất (mô tả đầy đủ nhất)
        best = max(by_disease[name], key=lambda r: len(_observable_fields(r.get("triệu_chứng", ""))))
        obs = _observable_fields(best.get("triệu_chứng", ""))
        if len(obs) < 3:
            continue
        # Lấy tối đa 6 field đầu làm đầu vào (mô phỏng lời khai + vọng chẩn)
        symptoms = obs[:6]
        cases.append({
            "id": f"recall_{len(cases)+1:02d}",
            "type": "recall",
            "specialty": "",
            "symptoms": symptoms,
            "raw": ", ".join(symptoms),
            "expect_disease": name,
            "hoi_chung": best.get("hội_chứng", "").strip(),
            "note": "Máy sinh từ CSV — bác sĩ rà lại.",
        })

    # Ca phân biệt soạn tay (exclude + expect)
    for i, m in enumerate(MANUAL_CASES):
        cases.append({
            "id": f"discrim_{i+1:02d}",
            "type": "discrim",
            "specialty": m.get("specialty", ""),
            "symptoms": m["symptoms"],
            "raw": m["raw"],
            "forbid_disease": m.get("forbid", []),
            "expect_disease_any": m.get("expect", []),
            "note": "Soạn tay từ lỗi đã biết (lịch sử commit + test lâm sàng).",
        })

    out = {
        "version": 1,
        "description": "Bộ ca chuẩn (gold) cho tầng khớp bệnh danh — phạm vi Vấn + Vọng (lưỡi/mặt), "
                       "KHÔNG mạch. Ca 'recall': bệnh phải nằm trong top-3. Ca 'discrim': bệnh trong "
                       "forbid không được xuất hiện, bệnh trong expect_any phải xuất hiện. Cần bác sĩ "
                       "Đông y rà soát & mở rộng.",
        "cases": cases,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Đã sinh {len(cases)} ca recall -> {OUT_PATH}")


if __name__ == "__main__":
    run()
