#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_tayy_drugs.py — Bộ lọc danh mục thuốc Tây y tham khảo (build_tayy_drugs.py).

Vì sao phải test: cột thuốc nguồn là dịch máy có 3 lớp rác đã đo — (1) 63 dòng nhiễm
văn bản luận văn '1- Tên đề tài: ...', (2) ~6% item là Trung dược thành phẩm trá hình
(hoàn/thang/tán/cao/đan — kể cả '...hoàn viên'), (3) filter 'tán' NGÂY THƠ sẽ giết oan
kháng sinh thật dạng 'viên nén phân tán' (dispersible tablet). Danh mục này hiển thị
cho BÁC SĨ trong khối Tây y tham khảo — lọt rác là mất uy tín cả khối.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_tayy_drugs.py   (exit != 0 nếu fail)
"""
import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from collections import Counter  # noqa: E402

from build_tayy_drugs import DONG_Y_RE, DISPERSIBLE_SUFFIX, clean_drug_items  # noqa: E402


def chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return not cond


def _clean(items):
    return clean_drug_items(json.dumps(items, ensure_ascii=False), Counter())


def main():
    fails = 0

    # ---- 1. Rác luận văn + Đông y trá hình bị loại ----
    out = _clean([
        "1- Tên đề tài: Giải pháp nâng cao hiệu quả đầu tư phát triển",
        "sáu vị địa hoàng hoàn",
        "Xuyên tâm liên hoàn viên",
        "mười hương đan",
        "hoạt huyết giảm đau tán",
        "Bổ trung ích khí thang",
        "Cao dán hồng hoa cao",
        "Viên nén Levofloxacin lactate",
    ])
    fails += chk("chỉ giữ thuốc Tây thật", out == ["Viên nén Levofloxacin lactate"], str(out))

    # ---- 2. NGOẠI LỆ 'phân tán': dispersible tablet là thuốc Tây thật ----
    out = _clean([
        "amoxicillin-kali clavulanate (4: 1) viên nén phân tán",
        "Viên nén phân tán Azithromycin",
        "an cung ngưu hoàng hoàn",
    ])
    fails += chk("'viên nén phân tán' KHÔNG bị giết oan",
                 out == ["amoxicillin-kali clavulanate (4: 1) viên nén phân tán",
                         "Viên nén phân tán Azithromycin"], str(out))
    fails += chk("regex DONG_Y_RE bắt '...hoàn viên'",
                 bool(DONG_Y_RE.search("xuyên tâm liên hoàn viên")))
    fails += chk("hằng số ngoại lệ đúng", DISPERSIBLE_SUFFIX == "phân tán")

    # ---- 3. Dedupe casefold + cap + item ngắn ----
    out = _clean(["Ibuprofen", "ibuprofen", "IBUPROFEN viên", "SP", "pH4"])
    fails += chk("dedupe casefold + loại item <6 ký tự",
                 out == ["Ibuprofen", "IBUPROFEN viên"], str(out))
    out = _clean([f"Thuốc thử nghiệm số {i}" for i in range(10)])
    fails += chk("cap 6 thuốc/bệnh", len(out) == 6, str(len(out)))

    # ---- 4. JSON hỏng -> rỗng, không nổ ----
    fails += chk("JSON hỏng degrade về rỗng",
                 clean_drug_items("[hỏng...", Counter()) == [])

    # ---- 5. Integration: file thật (nếu có) không còn rác ----
    drugs_path = os.path.join(_ROOT, "data", "tayy_drugs.json")
    if os.path.exists(drugs_path):
        data = json.load(open(drugs_path, encoding="utf-8"))
        fails += chk("schema_version đúng", data.get("schema_version") == "tayy-drugs-v1")
        bad_junk = bad_dongy = 0
        total = 0
        for names in data.get("drugs", {}).values():
            for name in names:
                total += 1
                low = name.casefold()
                if "tên đề tài" in low:
                    bad_junk += 1
                if DONG_Y_RE.search(low) and not low.endswith(DISPERSIBLE_SUFFIX):
                    bad_dongy += 1
        fails += chk(f"file thật ({total} tên): 0 rác đề tài", bad_junk == 0, str(bad_junk))
        fails += chk("file thật: 0 Đông y trá hình", bad_dongy == 0, str(bad_dongy))
        fails += chk("mọi disease_id đúng dạng",
                     all(k.startswith("TAYY-") for k in data.get("drugs", {})))
    else:
        print("[SKIP] Chưa có data/tayy_drugs.json — chạy build_tayy_drugs.py --write trước")

    print("\n" + ("✅ TẤT CẢ PASS" if fails == 0 else f"❌ {fails} KIỂM TRA FAIL"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
