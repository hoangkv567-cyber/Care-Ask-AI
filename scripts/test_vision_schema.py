#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_vision_schema.py — Kiểm vọng chẩn CÓ CẤU TRÚC (src/vision_schema.py).

Bảo đảm: parse JSON chịu lỗi (rào ```json, prose thừa), map DETERMINISTIC JSON -> triệu chứng chuẩn,
'không rõ' KHÔNG sinh triệu chứng (không bịa), và mapper khớp với bộ chuẩn hóa của fusion_pipeline.

Chạy:  python scripts/test_vision_schema.py    (Exit 0 nếu PASS — không cần Neo4j/VLM)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vision_schema import (
    parse_vlm_json, tongue_json_to_symptoms, face_json_to_symptoms,
    tongue_json_to_prose, face_json_to_prose, is_makeup,
)
from src.fusion_pipeline import TCMFusionPipeline as F


def _c(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def run():
    ok = True

    print("── parse_vlm_json: chịu rào & prose thừa ──")
    ok &= _c("JSON trần", parse_vlm_json('{"than_luoi":"đỏ"}') == {"than_luoi": "đỏ"})
    ok &= _c("rào ```json", parse_vlm_json('```json\n{"than_luoi":"nhợt"}\n```') == {"than_luoi": "nhợt"})
    ok &= _c("prose bao quanh", parse_vlm_json('Kết quả: {"phu":"có"} (hết)') == {"phu": "có"})
    ok &= _c("rác -> None", parse_vlm_json("không phải json") is None)
    ok &= _c("rỗng -> None", parse_vlm_json("") is None)

    print("── tongue_json_to_symptoms: deterministic ──")
    t1 = tongue_json_to_symptoms({"than_luoi": "nhợt", "reu_mau": "trắng", "reu_day": "mỏng",
                                  "dau_rang": "có", "luoi_beu": "có", "vet_nut": "không", "reu_chat": "nhuận"})
    ok &= _c(f"khí hư đàm thấp: {t1}",
             set(t1) == {"lưỡi nhợt", "lưỡi bệu", "rìa lưỡi có hằn răng", "rêu trắng mỏng"})
    t2 = tongue_json_to_symptoms({"than_luoi": "đỏ", "reu_mau": "vàng", "reu_day": "dày", "reu_chat": "nhớt"})
    ok &= _c(f"nhiệt (rêu vàng nhớt ưu tiên): {t2}", "lưỡi đỏ" in t2 and "rêu vàng nhớt" in t2)
    t3 = tongue_json_to_symptoms({"than_luoi": "hồng nhạt", "reu_mau": "trắng", "reu_day": "mỏng"})
    ok &= _c(f"hồng nhạt = sinh lý -> KHÔNG sinh 'lưỡi ...' màu: {t3}",
             not any("lưỡi" in s and "bệu" not in s for s in t3) and "rêu trắng mỏng" in t3)
    t4 = tongue_json_to_symptoms({"than_luoi": "không rõ", "reu_mau": "không rõ", "dau_rang": "không rõ"})
    ok &= _c(f"toàn 'không rõ' -> rỗng (không bịa): {t4}", t4 == [])
    t5 = tongue_json_to_symptoms({"reu_mau": "không rêu"})
    ok &= _c(f"không rêu: {t5}", t5 == ["lưỡi không có rêu"])

    print("── face_json_to_symptoms: deterministic ──")
    f1 = face_json_to_symptoms({"sac_mat": "trắng nhợt", "phu": "có", "go_ma_do": "không", "trang_diem": "không"})
    ok &= _c(f"trắng nhợt + phù: {f1}", set(f1) == {"mặt nhợt nhạt", "mặt phù"})
    f2 = face_json_to_symptoms({"sac_mat": "hồng hào bình thường", "go_ma_do": "không rõ"})
    ok &= _c(f"hồng hào + không rõ -> rỗng: {f2}", f2 == [])
    ok &= _c("is_makeup True", is_makeup({"trang_diem": "có"}))
    ok &= _c("is_makeup False", not is_makeup({"trang_diem": "không"}))

    print("── prose render ──")
    p = tongue_json_to_prose({"than_luoi": "nhợt", "reu_mau": "trắng", "reu_day": "mỏng", "dau_rang": "có"})
    ok &= _c(f"prose lưỡi: {p!r}", "lưỡi" in p.lower() and "răng" in p.lower())
    pf = face_json_to_prose({"sac_mat": "trắng nhợt", "phu": "có"})
    ok &= _c(f"prose mặt: {pf!r}", "sắc mặt" in pf.lower())

    print("── Tên đầu ra tương thích bộ chuẩn hóa fusion (không bị xung đột loại bỏ) ──")
    # Đưa qua _resolve_symptom_conflicts: các tên hợp lệ không được bị rơi hết
    o = F.__new__(F)
    resolved = F._resolve_symptom_conflicts(o, list(t1))
    ok &= _c(f"resolve giữ dấu hằn răng: {resolved}",
             any("hằn răng" in s or "dấu răng" in s or "vết răng" in s for s in resolved))

    # ── HỢP ĐỒNG PROMPT ↔ SCHEMA ──────────────────────────────────────────────────────
    # Đo được: TRƯỚC bài test này, 0/43 file scripts/test_*.py import src.prompts. Nghĩa là mọi
    # thay đổi trong prompt vọng chẩn đều KHÔNG được test nào phủ — "41/41 XANH" rỗng nghĩa ở đó.
    # Bug thật đã xảy ra: trường "luoi_beu" được KHAI BÁO trong schema JSON nhưng KHÔNG có một luật
    # nào hướng dẫn cách nhận định -> mô hình mặc định "không rõ" -> bỏ sót dấu lưỡi bệu trên ảnh
    # có hằn răng + thân lưỡi bệu rõ. Đường prose (cũ) thì có hẳn LUẬT 8 và LUẬT 11 "MUST OBEY"
    # cho đúng dấu này — công sức đó không được mang sang khi chuyển schema JSON.
    print("── Hợp đồng prompt ↔ schema: mọi trường khai báo phải CÓ luật ──")
    import re as _re
    from src import prompts as _P
    # Neo TƯỜNG MINH cho từng trường, KHÔNG đếm số lần xuất hiện tên khóa: 'than_luoi' chỉ xuất hiện
    # 1 lần (dòng schema) nhưng CÓ luật thật — luật về ánh sáng ấm làm lưỡi trông đỏ hơn thực, và luật
    # đó không gọi tên khóa. Đếm ngây thơ sẽ báo fail OAN cho nó (đã dính đúng bẫy này khi viết test).
    _RULE_ANCHOR = {
        "TONGUE_JSON_PROMPT_VI": {
            "than_luoi": "ánh sáng ấm làm lưỡi trông đỏ hơn thực",
            "reu_mau": '"reu_mau"="không rêu" CHỈ KHI',
            "reu_day": '"reu_day"="mỏng"',
            "dau_rang": "Vết lõm gợn sóng ở mép lưỡi",
            "vet_nut": '"vet_nut"="có" CHỈ KHI',
            "luoi_beu": '"luoi_beu"="có" CHỈ KHI',
        },
        "FACE_JSON_PROMPT_VI": {
            "sac_mat": 'MẶC ĐỊNH "hồng hào bình thường"',
            "go_ma_do": '"go_ma_do"="có" CHỈ KHI',
            "ban_do": '"ban_do"="có" CHỈ KHI',
            "quang_tham": '"quang_tham"="có"',
            "trang_diem": '"trang_diem": "có"',
        },
    }
    # Nợ ĐÃ BIẾT, ghi nhận thay vì fail (tách PR riêng): trường khai báo mà chưa có luật nào.
    # Nợ P1 (thẩm định xếp "tiêu chí LỆCH MỘT CHIỀU"): reu_chat CHỈ có ràng buộc NHẤT QUÁN với
    # reu_mau, KHÔNG có tiêu chí phân biệt nhuận/nhớt/khô/bong tróc — mà 'rêu nhớt' là dấu đàm-thấp
    # then chốt. reu_day chỉ có nhánh ép "mỏng", không có tiêu chí "dày".
    _KNOWN_GAPS = {"FACE_JSON_PROMPT_VI": {"phu"},
                   "TONGUE_JSON_PROMPT_VI": {"reu_chat"}}
    for _pname, _ptxt in (("TONGUE_JSON_PROMPT_VI", _P.TONGUE_JSON_PROMPT_VI),
                          ("FACE_JSON_PROMPT_VI", _P.FACE_JSON_PROMPT_VI)):
        _decl = _re.findall(r'"([a-z_]+)":\s*"[^"]*"\s*\|', _ptxt)
        for _k in _decl:
            if _k in _KNOWN_GAPS.get(_pname, ()):
                print(f"[NOTE] {_pname}.{_k}: chưa có luật (nợ đã biết, tách PR riêng)")
                continue
            _anchor = _RULE_ANCHOR.get(_pname, {}).get(_k)
            # Trường MỚI mà chưa khai neo -> fail, buộc người thêm trường phải khai luật cho nó.
            ok &= _c(f"{_pname}.{_k} có luật hướng dẫn",
                     bool(_anchor) and _anchor in _ptxt)

    print("── Luật luoi_beu: BẢO THỦ, KHÔNG bắc cầu từ dau_rang ──")
    _t = _P.TONGUE_JSON_PROMPT_VI
    ok &= _c("đòi >=2 trong 3 dấu (không phải >=1)", "ÍT NHẤT HAI" in _t)
    ok &= _c("mặc định 'không rõ' khi chưa đủ", 'MẶC ĐỊNH "không rõ"' in _t)
    ok &= _c("cấm bắc cầu dau_rang -> luoi_beu",
             "KHÔNG được suy từ vết hằn răng" in _t)
    # Giọng mệnh lệnh trống của LUẬT 11 (đường prose) KHÔNG được bê sang: ở JSON không có tiền đề
    # văn bản nào, mệnh lệnh trống sẽ đẩy mô hình nghiêng "có" -> tái diễn over-report âm-hư.
    ok &= _c("KHÔNG dùng giọng 'MUST OBEY/Do NOT skip'",
             "MUST OBEY" not in _t and "Do NOT skip" not in _t)
    # "không rõ" phải là câm lặng ở tầng ánh xạ (không sinh triệu chứng nào)
    ok &= _c("luoi_beu='không rõ' -> KHÔNG sinh triệu chứng",
             not any("bệu" in s.lower() for s in tongue_json_to_symptoms({"luoi_beu": "không rõ"})))
    ok &= _c("luoi_beu='có' -> CÓ sinh triệu chứng",
             any("bệu" in s.lower() for s in tongue_json_to_symptoms({"luoi_beu": "có"})))

    print("\n" + ("✅ TẤT CẢ PASS" if ok else "❌ CÓ CA FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
