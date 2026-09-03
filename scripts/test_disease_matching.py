#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_disease_matching.py — BỘ TEST HỒI QUY cho _find_matching_diseases.

MỤC ĐÍCH: khoá lại các fix đã làm (bộ lọc generic-symptom + các guard chuyên khoa) và bắt
regression cho MỌI thay đổi sau này ở tầng khớp bệnh danh. CHỈ ĐỌC CSV (self.csv_rows), KHÔNG
cần Neo4j, KHÔNG cần nhãn chuyên gia — assertion lấy từ các ca lâm sàng đã kiểm chứng.

Chạy:  python scripts/test_disease_matching.py
Exit 0 nếu tất cả PASS, 1 nếu có FAIL.

Mỗi ca gồm:
  symptoms  : danh sách triệu chứng (giống search_terms + vision đưa vào _find_matching_diseases)
  raw       : văn bản người dùng nhập (dùng cho khớp mềm Cách 2)
  forbid    : bệnh danh TUYỆT ĐỐI không được xuất hiện (đã xác nhận vô lý về lâm sàng)
  expect    : bệnh danh NÊN xuất hiện trong ứng viên (khớp hợp lý)
"""
import os
import sys

# Cho phép chạy từ bất kỳ đâu: thêm thư mục gốc dự án (cha của scripts/) vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fusion_pipeline import TCMFusionPipeline as F


def norm(s):
    return s.strip().lower()


CASES = [
    {
        "name": "Hô hấp (ho/khó thở/mệt), có ảnh lưỡi-mặt, KHÔNG đau ngực/không bệnh gan/da/sinh dục",
        "symptoms": ["ho nhiều", "khó thở", "mệt mỏi", "nốt mụn đỏ", "rêu trắng", "lưỡi hồng", "mặt nhợt nhạt"],
        "raw": "ho nhiều, khó thở, mệt mỏi",
        # các bệnh danh từng bị khớp SAI trong quá trình test thực tế:
        "forbid": ["Áp xe gan", "Bạch tiển (nấm da)", "Dương nuy", "Hung tý", "Động thai"],
        "expect": ["Mạn tính tắc nghẽn phế bệnh"],   # ít nhất 1 bệnh hô hấp phải nằm trong ứng viên
    },
    {
        "name": "Đau ngực thật (Hung tý hợp lệ)",
        "symptoms": ["đau thắt ngực", "tức ngực", "khó thở", "hồi hộp", "mệt mỏi"],
        "raw": "đau ngực dữ dội, tức ngực, khó thở, hồi hộp",
        "forbid": [],
        "expect": ["Hung tý"],
    },
    {
        "name": "Nam khoa thật (có liệt dương)",
        "symptoms": ["liệt dương", "đau lưng mỏi gối", "tiểu đêm", "mệt mỏi"],
        "raw": "liệt dương, yếu sinh lý, đau lưng",
        "forbid": [],
        "expect": ["Dương nuy"],
    },
    {
        "name": "Thai phụ thật (dọa sảy)",
        "symptoms": ["động thai", "đau bụng", "ra huyết"],
        "raw": "đang mang thai, đau bụng, ra huyết âm đạo",
        "forbid": [],
        "expect": ["Động thai"],
    },
    {
        "name": "Bế kinh do hàn (KHÔNG mang thai) — cấm Quỷ thai + bài trục thai độc",
        "symptoms": ["bụng dưới đau lạnh", "mạch trầm khẩn", "tay chân lạnh", "bế kinh"],
        "raw": "bụng dưới đau lạnh, bế kinh, tay chân lạnh",
        "forbid": ["Quỷ thai"],
        "expect": [],
    },
    {
        "name": "Tiêu chảy dương hư — cấm Tâm nhồi máu (không có triệu chứng ngực)",
        "symptoms": ["tiêu chảy", "phân lỏng nát", "sợ lạnh", "tay chân lạnh", "đau lưng", "đau bụng"],
        "raw": "tiêu chảy, phân lỏng nát, sợ lạnh, tay chân lạnh",
        "forbid": ["Tâm nhồi máu"],
        "expect": [],
    },
    {
        "name": "Nấc vị hàn — cấm Thiên đầu thống (không đau đầu)",
        "symptoms": ["nấc", "ưa nóng", "miệng nhạt không khát", "mạch trì", "rêu trắng nhuận"],
        "raw": "nấc, ưa nóng, miệng nhạt không khát",
        "forbid": ["Thiên đầu thống"],
        "expect": [],
    },
    {
        "name": "Cảm mạo phong hàn — cấm Tâm quý/Ngân tiết/Phong chẩn/Nhũ ung/Áp xe phế (thiếu triệu chứng chỉ điểm)",
        "symptoms": ["sợ lạnh", "sổ mũi", "sốt", "ho", "mặt nhợt nhạt", "lưỡi hồng nhạt", "rêu trắng"],
        "raw": "sợ lạnh, sổ mũi, sốt, ho",
        "forbid": ["Tâm quý", "Ngân tiết", "Phong chẩn", "Nhũ ung", "Áp xe phế"],
        "expect": [],
    },
]


def run():
    o = F.__new__(F)          # bare instance — không gọi __init__ (không kết nối Neo4j)
    F._load_csv_data(o)
    if not getattr(o, "csv_rows", None):
        print("LỖI: không tải được CSV.")
        return 1

    all_ok = True
    for c in CASES:
        res = F._find_matching_diseases(o, c["symptoms"], raw_user_text=c["raw"])
        names = [norm(m["benh_ly"]) for m in res]
        top = [m["benh_ly"] for m in res[:5]]

        fails = []
        for bad in c["forbid"]:
            if norm(bad) in names:
                fails.append(f"XUẤT HIỆN bệnh cấm '{bad}'")
        for good in c["expect"]:
            if norm(good) not in names:
                fails.append(f"THIẾU bệnh kỳ vọng '{good}'")

        status = "PASS" if not fails else "FAIL"
        if fails:
            all_ok = False
        print(f"[{status}] {c['name']}")
        print(f"        top ứng viên: {top}")
        for fl in fails:
            print(f"        ✗ {fl}")

    print("\n" + ("✅ TẤT CẢ PASS" if all_ok else "❌ CÓ CA FAIL — kiểm tra thay đổi tầng khớp bệnh"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
