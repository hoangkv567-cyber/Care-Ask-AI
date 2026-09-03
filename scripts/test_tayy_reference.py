#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_tayy_reference.py — An toàn khối "Tây y tham khảo (ICD-10)" (offline, 0 lượt LLM).

Khối này đưa dữ liệu bệnh Tây y DỊCH MÁY CHƯA KIỂM DUYỆT vào kết quả chẩn đoán (chỉ role
bác sĩ), nên ranh giới phải được khóa bằng test:
  1) CỘT CẤM (tỉ_lệ_chữa_khỏi, đề_xuất_thuốc, thuốc_phổ_biến, thông_tin_thuốc...) không
     bao giờ xuất hiện trong index lẫn payload — kể cả GIÁ TRỊ mẫu, không chỉ tên cột.
  2) Schema + ngưỡng: show:false khi <2 triệu chứng khớp; do_khop giảm dần; <=5 item;
     mọi item mang nhãn kiểm duyệt.
  3) PHÂN QUYỀN: api._strip_clinical_internals phải cắt 'tayy_reference' khỏi payload
     role thường (bệnh nhân đọc tên bệnh Tây y gợi ý rất dễ tự chẩn đoán).
  4) PROMPT LLM SẠCH (kiểm cấu trúc): _compute_tayy_reference chỉ được gọi SAU
     _generate_explainable_answer trong run_diagnosis; không mã ICD/tên hàm tayy nào
     lọt vào _generate_explainable_answer (nơi dựng prompt Mục 2-4) — luật 1/7 của
     prompt sẽ ép LLM biện luận bệnh Tây y bằng y lý Đông y nếu bị bơm vào.
  5) MATCHER ĐÔNG Y BẤT BIẾN: _find_matching_diseases cho kết quả identical trước/sau
     khi nạp index Tây y.
  6) Hiệu năng: _compute_tayy_reference (đã warm) < 300ms/lượt.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_tayy_reference.py   (exit != 0 nếu fail)
"""
import csv
import inspect
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

csv.field_size_limit(10**9)
os.environ.setdefault("JWT_SECRET", "test-secret-chi-dung-cho-test")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "data", "tayy_reference_index.json")
CLEAN_PATH = os.path.join(ROOT, "data", "TayY_clean.csv")

BANNED_COLUMNS = (
    "tỉ_lệ_chữa_khỏi", "đề_xuất_thuốc", "thuốc_phổ_biến", "thông_tin_thuốc",
    "đề_xuất_món_ăn", "nên_ăn_thực_phẩm_chứa", "cách_phòng_tránh",
    "điều_trị_tách_từ_phòng_tránh",
)

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402


def chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return not cond   # trả số lỗi (0/1) để cộng dồn


def _drug_values(columns, limit=30):
    values = []
    if not os.path.exists(CLEAN_PATH):
        return values
    with open(CLEAN_PATH, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            for col in columns:
                raw = (row.get(col) or "").strip()
                if raw and len(raw) > 20:
                    try:
                        items = json.loads(raw)
                        if isinstance(items, list) and items:
                            first = str(items[0]).strip()
                            if len(first) >= 12:
                                values.append(first)
                    except json.JSONDecodeError:
                        pass
            if len(values) >= limit:
                break
    return values[:limit]


def _values_cam_tuyet_doi(limit=30):
    """Giá trị từ cột CẤM TUYỆT ĐỐI (đề_xuất_thuốc — dịch bịa) — không được xuất
    hiện ở BẤT KỲ đầu ra nào, kể cả payload bác sĩ."""
    return _drug_values(("đề_xuất_thuốc",), limit)


def _values_thuoc_pho_bien(limit=30):
    """Giá trị cột thuốc_phổ_biến — policy 2026-08-19 (user duyệt): CHO PHÉP trong
    payload bác sĩ (kèm nhãn), vẫn CẤM trong index + payload đã strip (bệnh nhân)."""
    return _drug_values(("thuốc_phổ_biến",), limit)


def _banned_values(limit=30):
    """Cả hai nhóm — dùng cho các bề mặt cấm tuyệt đối (index)."""
    return (_values_cam_tuyet_doi(limit // 2) + _values_thuoc_pho_bien(limit // 2))[:limit]


def main():
    fails = 0

    # ---- 0. Có index chưa (thiếu -> khối tự tắt, vẫn phải pass các test degrade) ----
    has_index = os.path.exists(INDEX_PATH)
    print(f"Index: {INDEX_PATH} ({'có' if has_index else 'KHÔNG có — test degrade'})")

    o = F.__new__(F)              # bare instance — không __init__, không Neo4j, không LLM

    # ---- 1. Cột cấm không có trong index ----
    if has_index:
        raw_index = open(INDEX_PATH, encoding="utf-8").read()
        for col in BANNED_COLUMNS:
            fails += chk(f"index KHÔNG chứa tên cột cấm {col!r}", col not in raw_index)
        leaked = [v for v in _banned_values() if v in raw_index]
        fails += chk(
            "index KHÔNG chứa giá trị thuốc mẫu nào (30 mẫu từ CSV)",
            not leaked, str(leaked[:3]),
        )

    # ---- 2. Schema + ngưỡng của _compute_tayy_reference ----
    empty = F._compute_tayy_reference(o, [])
    fails += chk("0 triệu chứng -> show:false", empty == {"show": False, "items": []})
    one = F._compute_tayy_reference(o, ["ho"])
    fails += chk("1 triệu chứng -> show:false", one.get("show") is False)

    if has_index:
        index = F._get_tayy_reference_index(o)
        fails += chk("index nạp được (>1000 bệnh)", len(index["items"]) > 1000,
                     str(len(index["items"])))
        # Ca khớp rõ: lấy 1 bệnh thật trong index có >=4 triệu chứng đặc hiệu rồi khai
        # đúng các triệu chứng đó -> bệnh ấy phải đứng top.
        target = None
        idf = index["idf"]
        for it in index["items"]:
            distinctive = [s for s in it["tc"] if idf.get(s, 0) >= 3.0 and len(s) >= 6]
            if len(distinctive) >= 4:
                target = (it, distinctive[:4])
                break
        fails += chk("tìm được bệnh mẫu có triệu chứng đặc hiệu", target is not None)
        if target:
            it, syms = target
            res = F._compute_tayy_reference(o, syms)
            fails += chk("ca khớp rõ -> show:true", res.get("show") is True)
            items = res.get("items") or []
            fails += chk("<=5 item", len(items) <= 5, str(len(items)))
            scores = [x["do_khop"] for x in items]
            fails += chk("do_khop giảm dần", scores == sorted(scores, reverse=True), str(scores))
            names = [x["ten_benh"] for x in items]
            fails += chk("bệnh mẫu xuất hiện trong kết quả", it["ten"] in names,
                         f"mẫu={it['ten']!r} top={names[:3]}")
            for x in items:
                fails += chk(
                    f"item {x['ten_benh'][:28]!r} có nhãn kiểm duyệt",
                    bool(x.get("trang_thai_label")),
                )
            # A3 (policy 2026-08-19): cột cấm tuyệt đối (đề_xuất_thuốc) KHÔNG BAO GIỜ
            # trong payload; thuốc_phổ_biến ĐƯỢC PHÉP ở payload đầy đủ (bác sĩ) nhưng
            # PHẢI VẮNG ở payload đã strip (bệnh nhân) — kiểm ở A3b.
            payload = json.dumps(res, ensure_ascii=False)
            leaked = [v for v in _values_cam_tuyet_doi(10) if v in payload]
            fails += chk("payload KHÔNG chứa giá trị cột cấm tuyệt đối", not leaked)
            if os.path.exists(os.path.join(ROOT, "data", "tayy_drugs.json")):
                fails += chk(
                    "item bác sĩ có field thuoc_tay_tham_khao dạng list",
                    all(isinstance(x.get("thuoc_tay_tham_khao"), list) for x in items),
                )
            # A3b: payload ĐÃ STRIP (role thường) không chứa thuốc nào + không field thuốc
            from api import _strip_clinical_internals as _S
            stripped_payload = json.dumps(
                _S({"tayy_reference": res, "diagnosis_result": {"answer": "x"}}),
                ensure_ascii=False,
            )
            leaked_strip = [
                v for v in (_values_cam_tuyet_doi(10) + _values_thuoc_pho_bien(10))
                if v in stripped_payload
            ]
            fails += chk("payload ĐÃ STRIP không chứa giá trị thuốc nào", not leaked_strip)
            fails += chk("payload ĐÃ STRIP không có field thuoc_tay_tham_khao",
                         "thuoc_tay_tham_khao" not in stripped_payload)

            # ---- 6. Hiệu năng (đã warm) ----
            t0 = time.perf_counter()
            for _ in range(5):
                F._compute_tayy_reference(o, syms + ["mệt mỏi", "chán ăn", "sốt nhẹ"])
            ms = (time.perf_counter() - t0) / 5 * 1000
            fails += chk("hiệu năng < 300ms/lượt (warm)", ms < 300, f"{ms:.0f}ms")

    # ---- 3. Phân quyền: role thường nhận bản RÚT GỌN (policy 2026-08-19) ----
    from api import TAYY_KHUYEN_CAO, _strip_clinical_internals as S
    sample = {
        "diagnosis_result": {"answer": "## KQ", "data": {}},
        "tayy_reference": {"show": True, "items": [
            {"ten_benh": "Viêm phổi", "icd10_code": "J18.9", "trang_thai": "da_kiem_duyet",
             "trang_thai_label": "Đã kiểm duyệt", "do_khop": 0.9, "khoa": "hô hấp",
             "mo_ta_ngan": "mô tả dịch máy", "thuoc_tay_tham_khao": ["erythromycin"],
             "benh_dong_y_tuong_ung": [{"benh_ly": "Khái thấu"}], "ten_may": False},
            {"ten_benh": "Bệnh chưa duyệt", "icd10_code": "X99", "trang_thai": "chua_kiem_duyet",
             "trang_thai_label": "(chưa kiểm duyệt — chỉ tham khảo)"},
        ]},
        "deep_inquiry": {"show": False, "factors": []},
        "status": "completed",
    }
    stripped = S(sample)
    reduced = stripped.get("tayy_reference")
    fails += chk("role thường NHẬN bản rút gọn tayy_reference", isinstance(reduced, dict))
    if isinstance(reduced, dict):
        fails += chk("khuyen_cao đúng nguyên văn backend",
                     reduced.get("khuyen_cao") == TAYY_KHUYEN_CAO)
        fails += chk("chỉ còn bệnh ĐÃ kiểm duyệt", len(reduced.get("items", [])) == 1)
        item_keys = set(reduced["items"][0].keys())
        fails += chk("item rút gọn đúng ALLOW-LIST {ten_benh, icd10_code, trang_thai_label}",
                     item_keys == {"ten_benh", "icd10_code", "trang_thai_label"},
                     str(sorted(item_keys)))
    # history cũ (item không có mã trang_thai) -> key phải biến mất
    old = {"tayy_reference": {"show": True, "items": [{"ten_benh": "X", "icd10_code": "A00"}]},
           "diagnosis_result": {"answer": "## KQ"}}
    fails += chk("history cũ (thiếu mã trang_thai) -> key biến mất",
                 "tayy_reference" not in S(old))
    fails += chk("role thường vẫn nhận answer", stripped.get("diagnosis_result", {}).get("answer"))

    # ---- 4. Prompt LLM sạch (kiểm cấu trúc mã nguồn) ----
    run_src = inspect.getsource(F.run_diagnosis)
    gen_src = inspect.getsource(F._generate_explainable_answer)
    pos_gen = run_src.find("_generate_explainable_answer")
    pos_tayy = run_src.find("_compute_tayy_reference")
    fails += chk(
        "_compute_tayy_reference gọi SAU _generate_explainable_answer trong run_diagnosis",
        0 <= pos_gen < pos_tayy,
        f"gen@{pos_gen} tayy@{pos_tayy}",
    )
    fails += chk(
        "_generate_explainable_answer (nơi dựng prompt LLM) không đụng dữ liệu Tây y",
        "tayy" not in gen_src.lower(),
    )
    fails += chk(
        "run_diagnosis không nhét tayy_reference vào answer markdown",
        not re.search(r"final_markdown\s*[+]?=\s*.*tayy", run_src, re.IGNORECASE),
    )

    # ---- 5. Matcher Đông y bất biến trước/sau nạp index Tây y ----
    o2 = F.__new__(F)
    F._load_csv_data(o2)
    if getattr(o2, "csv_rows", None):
        case_syms = ["sợ lạnh", "sổ mũi", "ho", "đau đầu"]
        raw = "sợ lạnh, sổ mũi, ho, đau đầu"
        before = F._find_matching_diseases(o2, case_syms, raw_user_text=raw)
        F._get_tayy_reference_index(o2)          # ép nạp index
        F._compute_tayy_reference(o2, case_syms)
        after = F._find_matching_diseases(o2, case_syms, raw_user_text=raw)
        fails += chk("matcher Đông y identical trước/sau nạp index Tây y", before == after)
    else:
        fails += chk("matcher: tải được Medicine_clean.csv", False)

    print("\n" + ("✅ TẤT CẢ PASS" if fails == 0 else f"❌ {fails} KIỂM TRA FAIL"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
