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

    print("\n" + ("✅ TẤT CẢ PASS" if ok else "❌ CÓ CA FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
