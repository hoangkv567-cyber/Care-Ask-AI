#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_payload_phanquyen.py — Ranh giới PHÂN QUYỀN của payload trả về (api._strip_clinical_internals).

VÌ SAO CÓ FILE NÀY: trước đó 0 file test nào chạm tới hàm này, dù nó là ranh giới quyết định
người dùng thường thấy gì. Hai lỗi đã xảy ra ở đúng chỗ không được phủ:
  1) Cắt SẠCH vision_details -> giao diện rơi vào nhánh else và in "AI chưa phát hiện được triệu
     chứng bất thường qua ảnh" cho MỌI bệnh nhân. Câu đó SAI (AI có thấy, chỉ bị cắt theo quyền)
     và trấn an sai chiều — người tải ảnh lưỡi bệu hằn răng được báo là không có gì bất thường.
  2) Cùng màn hình đó, phần biện luận mà bệnh nhân VẪN nhận lại đang bàn về chính cái lưỡi ấy
     ("hằn răng là biểu hiện của thấp trệ ở Tỳ") -> tự mâu thuẫn.

Nay: giữ ĐÚNG hai câu mô tả tiếng Việt cho bệnh nhân, cắt mọi thứ còn lại.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_payload_phanquyen.py   (exit != 0 nếu fail)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("JWT_SECRET", "test-secret-chi-dung-cho-test")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def _mau():
    """Payload đầy đủ như pipeline sinh ra khi CÓ ảnh."""
    return {
        "source": "Tứ chẩn hợp tham (Fusion)",
        "input_fusion": "lưỡi bệu, rìa lưỡi có hằn răng, mệt mỏi",
        "has_makeup": True,
        "status": "completed",
        "questions": [],
        "deep_inquiry": {"state": [], "symptoms": []},
        "vision_details": {
            "tongue_description": "Swollen tongue with teeth marks",       # thô, tiếng Anh
            "face_description": "Slightly pale complexion",
            "tongue_description_vi": "Lưỡi bệu, rìa có hằn răng, rêu trắng mỏng.",
            "face_description_vi": "Sắc mặt hơi nhợt, không phù.",
            "analysis": "lưỡi bệu, rìa lưỡi có hằn răng",                   # nội bộ
            "detected_symptoms": ["lưỡi bệu", "rìa lưỡi có hằn răng"],      # nội bộ
            "structured": {"than_luoi": "nhợt", "dau_rang": "có"},          # JSON thô VLM
            "has_makeup": True,
        },
        "diagnosis_result": {"answer": "## Kết quả", "data": {"kg_tho": [1, 2, 3]}},
        "tayy_reference": {"show": True, "items": [
            {"ten_benh": "Viêm phổi", "icd10_code": "J18.9", "khoa": "hô hấp",
             "trang_thai": "da_kiem_duyet", "trang_thai_label": "Đã kiểm duyệt",
             "do_khop": 0.87, "mo_ta_ngan": "mô tả dịch máy nội bộ",
             "thuoc_tay_tham_khao": ["erythromycin", "amoxicillin"],
             "benh_dong_y_tuong_ung": [{"benh_ly": "Khái thấu", "trang_thai": "xac_nhan"}],
             "ten_may": False},
            {"ten_benh": "Bệnh chưa duyệt", "icd10_code": "X99",
             "trang_thai": "chua_kiem_duyet",
             "trang_thai_label": "(chưa kiểm duyệt — chỉ tham khảo)"},
        ]},
    }


def test_giu_dung_hai_mo_ta():
    from api import _strip_clinical_internals as S
    n = f = 0
    u = S(_mau())
    vd = u.get("vision_details") or {}

    ok = set(vd) == {"tongue_description_vi", "face_description_vi"}
    n += chk("người dùng nhận ĐÚNG 2 mô tả tiếng Việt", ok, str(sorted(vd)))
    f += (not ok)

    ok = "hằn răng" in (vd.get("tongue_description_vi") or "")
    n += chk("mô tả lưỡi tới được bệnh nhân (để họ tự soi AI đọc có đúng ảnh không)", ok)
    f += (not ok)

    # Rò rỉ: mọi khóa kỹ thuật phải BIẾN MẤT khỏi vision_details của người dùng thường.
    for k in ("analysis", "detected_symptoms", "structured",
              "tongue_description", "face_description", "has_makeup"):
        ok = k not in vd
        n += chk(f"KHÔNG rò vision_details.{k}", ok)
        f += (not ok)

    for k in ("input_fusion", "has_makeup"):
        ok = k not in u
        n += chk(f"KHÔNG rò {k} ở cấp trên cùng", ok)
        f += (not ok)

    # [POLICY 2026-08-19] tayy_reference: role thường nhận bản RÚT GỌN (bệnh đã kiểm
    # duyệt + ICD + khuyến cáo đi khám) — TUYỆT ĐỐI không thuốc Tây/mô tả dịch máy/
    # độ khớp/khoa/bệnh Đông y tương ứng. Allow-list: field mới KHÔNG tự lọt.
    tr = u.get("tayy_reference")
    ok = isinstance(tr, dict) and tr.get("show") is True
    n += chk("tayy_reference: role thường nhận bản rút gọn", ok)
    f += (not ok)
    if isinstance(tr, dict):
        ok = len(tr.get("items", [])) == 1 and tr["items"][0]["ten_benh"] == "Viêm phổi"
        n += chk("tayy rút gọn: chỉ bệnh ĐÃ kiểm duyệt", ok)
        f += (not ok)
        keys = set(tr["items"][0].keys())
        ok = keys == {"ten_benh", "icd10_code", "trang_thai_label"}
        n += chk("tayy rút gọn: item đúng allow-list 3 field", ok, str(sorted(keys)))
        f += (not ok)
        blob = str(tr)
        for cam in ("thuoc_tay_tham_khao", "erythromycin", "do_khop", "mo_ta_ngan",
                    "khoa", "benh_dong_y_tuong_ung", "ten_may"):
            ok = cam not in blob
            n += chk(f"tayy rút gọn: KHÔNG rò {cam}", ok)
            f += (not ok)
        ok = bool(tr.get("khuyen_cao"))
        n += chk("tayy rút gọn: có khuyến cáo đi khám", ok)
        f += (not ok)
    # khóa mới thêm vào item sau này KHÔNG tự lọt (allow-list, không phải block-list)
    m2 = _mau()
    m2["tayy_reference"]["items"][0]["truong_nhay_cam_moi"] = "bí mật"
    tr2 = S(m2).get("tayy_reference") or {}
    ok = "truong_nhay_cam_moi" not in str(tr2)
    n += chk("tayy rút gọn: khóa MỚI thêm không tự lọt", ok)
    f += (not ok)

    ok = "data" not in (u.get("diagnosis_result") or {})
    n += chk("KHÔNG rò dữ liệu KG thô (diagnosis_result.data)", ok)
    f += (not ok)

    ok = (u.get("diagnosis_result") or {}).get("answer") == "## Kết quả"
    n += chk("phần kết luận đọc được vẫn còn nguyên", ok)
    f += (not ok)
    return n, f


def test_danh_sach_CHO_PHEP_khong_phai_danh_sach_chan():
    """Khóa MỚI thêm vào vision_details sau này phải mặc định KHÔNG lọt ra người dùng thường.

    Đây là điểm khác biệt then chốt: nếu cài bằng danh sách CHẶN thì mỗi lần ai đó thêm trường
    nội bộ mới, nó lọt ra âm thầm cho tới khi có người để ý.
    """
    from api import _strip_clinical_internals as S
    n = f = 0
    m = _mau()
    m["vision_details"]["truong_noi_bo_moi_toanh"] = "bí mật"
    m["vision_details"]["confidence_scores"] = {"tongue": 0.4}
    vd = (S(m).get("vision_details") or {})
    ok = "truong_noi_bo_moi_toanh" not in vd and "confidence_scores" not in vd
    n += chk("khóa mới thêm KHÔNG tự lọt ra (dùng danh sách cho phép)", ok, str(sorted(vd)))
    f += (not ok)
    return n, f


def test_ba_trang_thai_vong_chan():
    """Ba tình huống PHẢI phân biệt được, vì giao diện in ba câu khác nhau."""
    from api import _strip_clinical_internals as S
    n = f = 0

    ok = "vision_details" not in S({**_mau(), "vision_details": None})
    n += chk("không tải ảnh -> không có vision_details (UI: 'Chưa tải ảnh')", ok)
    f += (not ok)

    ok = "vision_details" not in S({**_mau(), "vision_details": {"analysis": "", "structured": {}}})
    n += chk("có ảnh nhưng VLM không đọc được -> không có vision_details "
             "(UI: 'đã soi nhưng chưa ghi nhận')", ok)
    f += (not ok)

    ok = "vision_details" in S(_mau())
    n += chk("có ảnh + đọc được -> CÓ vision_details (UI: in mô tả thật)", ok)
    f += (not ok)
    return n, f


def test_bac_si_van_nhan_du():
    """Đường bác sĩ KHÔNG đi qua hàm cắt — payload phải nguyên vẹn."""
    n = f = 0
    src = open(os.path.join(ROOT, "api.py"), encoding="utf-8").read()
    src_nc = re.sub(r"#.*", "", src)          # gỡ chú thích: chính chú thích có nhắc tên hàm
    hits = re.findall(r"result if is_doctor else _strip_clinical_internals\(result\)", src_nc)
    hits += re.findall(r"raw_result if is_doctor else _strip_clinical_internals\(raw_result\)", src_nc)
    ok = len(hits) >= 2
    n += chk("cắt CHỈ áp cho non-doctor, ở cả chẩn đoán lẫn lịch sử", ok, f"{len(hits)} chỗ")
    f += (not ok)
    return n, f


def test_giao_dien_khong_noi_doi():
    """Giao diện KHÔNG được in 'AI chưa phát hiện...' khi lý do thật là chưa tải ảnh / bị cắt."""
    n = f = 0
    html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    # Gỡ chú thích JS, vì chú thích giải thích lỗi cũ có chứa nguyên văn câu sai.
    html_nc = re.sub(r"//[^\n]*", "", html)

    ok = "AI chưa phát hiện được triệu chứng bất thường qua ảnh" not in html_nc
    n += chk("câu trấn an sai đã bị gỡ khỏi mã", ok)
    f += (not ok)

    ok = "Chưa tải ảnh" in html_nc
    n += chk("có câu riêng cho trường hợp CHƯA TẢI ẢNH", ok)
    f += (not ok)

    ok = "noVisionMsg" in html_nc and html_nc.count("noVisionMsg") >= 3
    n += chk("cả hai nhánh else dùng chung hàm phân biệt tình huống", ok,
             f"{html_nc.count('noVisionMsg')} lần")
    f += (not ok)

    # [POLICY 2026-08-19] Khối bác sĩ hiển thị thuốc Tây y tham khảo -> caveat
    # "KHÔNG phải đơn thuốc" phải có mặt trong UI (disclaimer hardcode frontend).
    ok = "KHÔNG phải đơn thuốc" in html_nc
    n += chk("UI có caveat 'KHÔNG phải đơn thuốc' cho dòng thuốc Tây y", ok)
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_giu_dung_hai_mo_ta, test_danh_sach_CHO_PHEP_khong_phai_danh_sach_chan,
               test_ba_trang_thai_vong_chan, test_bac_si_van_nhan_du,
               test_giao_dien_khong_noi_doi):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
