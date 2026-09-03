#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dựng ``data/tayy_reference_index.json`` — index gọn cho khối "Tây y tham khảo".

Nguồn: data/TayY_clean.csv (8.7k bệnh Tây y đã làm sạch + mã ICD-10 nếu có) và
(tùy chọn) data/tayy_dup_adjudication.jsonl (phân xử trùng tên bằng LLM).

Nguyên tắc:
* CHỈ các cột được phép hiển thị. TUYỆT ĐỐI KHÔNG mang: tỉ_lệ_chữa_khỏi(*),
  đề_xuất_thuốc, thuốc_phổ_biến, thông_tin_thuốc, đề_xuất_món_ăn, nên_ăn/không_nên_ăn,
  cách_phòng_tránh, điều_trị_tách_từ_phòng_tránh (dữ liệu dịch máy chưa kiểm duyệt,
  cấm đưa cho bệnh nhân — xem data/tayy_clean_report.json ghi_chú).
* Loại hẳn dòng ui/tcm (classify_risks của mapper): "Trang chủ", "Bản mẫu:...",
  bài Đông y kinh điển — không phải bệnh Tây y, hiện cho bác sĩ vẫn là rác.
* Adjudication (nếu có, đối chiếu row_sha256, lệch là bỏ qua record):
  - cụm same-disease: dòng thua -> ``an: true`` (ẩn khỏi matcher, đỡ trùng lặp);
  - nhóm tách >1 cụm: ``ten`` = distinct_name + ``ten_may: true`` (chú thích
    "tên phân biệt do máy đề xuất" khi hiển thị).

Tiêu thụ bởi: src/fusion_pipeline.py::_get_tayy_reference_index (env override
``TCM_TAYY_INDEX_PATH``; thiếu file -> app degrade im lặng, không lỗi).

Chạy:  python scripts/build_tayy_reference_index.py          # dry-run thống kê
       python scripts/build_tayy_reference_index.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from map_tayy_icd10 import classify_risks, extract_lead  # noqa: E402
from review_tayy_icd10 import (  # noqa: E402
    ContractError,
    _iter_json_records,
    load_clean,
    sha256_file,
)

ROOT = SCRIPT_DIR.parent
DEFAULT_CLEAN = ROOT / "data" / "TayY_clean.csv"
DEFAULT_ADJUDICATION = ROOT / "data" / "tayy_dup_adjudication.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "tayy_reference_index.json"

SCHEMA_VERSION = "tayy-reference-index-v1"
# Cờ chất lượng còn ý nghĩa với người đọc khối tham khảo (các cờ khác là nội bộ).
DISPLAY_FLAGS = ("trung_ten_khac_noi_dung", "danh_sach_chua_tach")
MAX_SYMPTOMS = 12


def _parse_json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return []


def _clean_khoa(raw: str) -> str:
    """'[Nhi khoa, Nội khoa Nhi khoa]' -> 'nhi khoa, nội khoa nhi khoa' (tối đa 2 mục)."""
    text = (raw or "").strip().strip("[]").strip()
    parts, seen = [], set()
    for part in text.split(","):
        item = part.strip().strip("'\"").strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            parts.append(item)
    return ", ".join(parts[:2])


def _load_adjudication(path: Path, row_hashes: dict[str, str]) -> tuple[dict, dict, int]:
    """-> (hidden: {id: True}, renamed: {id: distinct_name}, stale_count)."""
    hidden: dict[str, bool] = {}
    renamed: dict[str, str] = {}
    stale = 0
    if not path.is_file():
        return hidden, renamed, stale
    for _, record in _iter_json_records(path, "adjudication sidecar"):
        if record.get("schema_version") != "tayy-dup-adjudication-v1":
            continue
        hashes = record.get("row_sha256") or {}
        ids = record.get("disease_ids") or []
        if not ids or any(row_hashes.get(i) != hashes.get(i) for i in ids):
            stale += 1
            continue
        consensus = record.get("consensus") or {}
        clusters = consensus.get("clusters") or []
        verdict = consensus.get("verdict")
        if verdict in (None, "no_consensus") or not clusters:
            continue
        multi = len(clusters) > 1
        for cluster in clusters:
            cluster_ids = [str(x) for x in (cluster.get("ids") or [])]
            keep = str(cluster.get("keep") or "")
            distinct = str(cluster.get("distinct_name") or "").strip()
            for disease_id in cluster_ids:
                if len(cluster_ids) >= 2 and disease_id != keep:
                    hidden[disease_id] = True
                if multi and distinct:
                    renamed[disease_id] = distinct
    return hidden, renamed, stale


def build_index(clean_path: Path, adjudication_path: Path) -> tuple[dict[str, Any], Counter]:
    clean = load_clean(clean_path)
    hidden, renamed, stale = _load_adjudication(adjudication_path, clean.row_hashes)

    stats: Counter = Counter(adjudication_stale=stale)
    items: list[dict[str, Any]] = []
    for row in clean.rows:
        disease_id = row["disease_id"]
        name = (row.get("tên_bệnh") or "").strip()
        description = row.get("mô_tả_bệnh") or ""
        first_sentence, _lead = extract_lead(description)
        risks = classify_risks(name, description, first_sentence)
        if risks["ui"] or risks["tcm"]:
            stats["loại_ui_tcm"] += 1
            continue

        symptoms = [
            s.casefold() for s in _parse_json_list(row.get("triệu_chứng") or "")
            if len(s) >= 2
        ][:MAX_SYMPTOMS]
        flags = [
            f for f in (row.get("cờ_chất_lượng") or "").split(";") if f in DISPLAY_FLAGS
        ]
        item: dict[str, Any] = {
            "id": disease_id,
            "ten": renamed.get(disease_id, name),
            "icd": (row.get("icd10_code") or "").strip(),
            "tt": (row.get("trạng_thái_kiểm_duyệt") or "chua_kiem_duyet").strip(),
            "khoa": _clean_khoa(row.get("khoa_điều_trị") or ""),
            "mo_ta": (first_sentence or description).strip()[:200],
            "tc": symptoms,
            "co": flags,
            "an": bool(hidden.get(disease_id)),
        }
        if disease_id in renamed:
            item["ten_may"] = True
        items.append(item)
        stats["giữ"] += 1
        if item["an"]:
            stats["ẩn_bản_trùng"] += 1
        if item["icd"]:
            stats["có_icd"] += 1
        if disease_id in renamed:
            stats["tên_máy_đề_xuất"] += 1
        if not symptoms:
            stats["không_triệu_chứng"] += 1

    index = {
        "schema_version": SCHEMA_VERSION,
        "source": "data/TayY_clean.csv",
        "source_sha256": sha256_file(clean_path),
        "rows": len(items),
        "items": items,
    }
    return index, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dựng index tham khảo Tây y cho app. Mặc định dry-run."
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true", help="Ghi file (mặc định chỉ thống kê)")
    args = parser.parse_args(argv)

    index, stats = build_index(args.clean, args.adjudication)
    for key in sorted(stats):
        print(f"  {key:<24} {stats[key]}")
    print(f"Items: {index['rows']}")

    # Chốt chặn cột cấm: tên cột cấm không được xuất hiện trong JSON đầu ra.
    payload = json.dumps(index, ensure_ascii=False, separators=(",", ":"))
    banned = (
        "tỉ_lệ_chữa_khỏi", "đề_xuất_thuốc", "thuốc_phổ_biến", "thông_tin_thuốc",
        "đề_xuất_món_ăn", "nên_ăn_thực_phẩm_chứa", "cách_phòng_tránh",
        "điều_trị_tách_từ_phòng_tránh",
    )
    leaked = [b for b in banned if b in payload]
    if leaked:
        raise ContractError(f"Index lộ cột cấm: {leaked}")

    if not args.write:
        size_mb = len(payload.encode("utf-8")) / 1e6
        print(f"[DRY-RUN] Chưa ghi; kích thước dự kiến ~{size_mb:.1f}MB. Thêm --write để ghi.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"[OK] {args.output} ({len(payload.encode('utf-8'))/1e6:.1f}MB, {index['rows']} items)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[LỖI CONTRACT] {exc}", file=sys.stderr)
        raise SystemExit(1)
