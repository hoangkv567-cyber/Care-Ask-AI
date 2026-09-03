#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dựng ``data/tayy_drugs.json`` — danh mục THUỐC TÂY Y THAM KHẢO cho khối bác sĩ.

Nguồn: CHỈ cột ``thuốc_phổ_biến`` của data/TayY_clean.csv (7.568 dòng, median 2
thuốc/bệnh, ~2/3 là hoạt chất INN thật). KHÔNG dùng ``đề_xuất_thuốc`` (9.799
distinct, nhiều tên dịch bịa) và ``thông_tin_thuốc`` (Trung dược phiên âm, hỏng).

Chính sách (user chốt 2026-08-19): thuốc Tây y CHỈ hiển thị cho role BÁC SĨ, kèm
nhãn "dịch máy, chưa kiểm duyệt, KHÔNG phải đơn thuốc"; api._strip_clinical_internals
cắt khỏi payload role thường. File này TÁCH RIÊNG khỏi tayy_reference_index.json
để giữ nguyên bất biến "index sạch tuyệt đối" và các test đi kèm.

Lọc rác (đo thật trên corpus):
* item chứa 'tên đề tài' — 63 dòng nhiễm chéo văn bản luận văn vào CSV;
* item Đông y trá hình: kết thúc (hoàn|thang|tán|cao|đan)(+' viên') — NGOẠI LỆ
  'phân tán' (dispersible tablet là thuốc Tây thật, vd amoxicillin viên nén phân tán);
* len<6 + cap 6/bệnh: defense-in-depth (corpus hiện không chạm tới).

Gần như không có liều lượng trong nguồn (0,3%) — đây là danh mục TÊN chế phẩm/hoạt
chất tham khảo, không phải phác đồ điều trị.

Chạy:  python scripts/build_tayy_drugs.py          # dry-run thống kê
       python scripts/build_tayy_drugs.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_tayy_icd10 import (  # noqa: E402
    ContractError,
    load_clean,
    sha256_file,
)

ROOT = SCRIPT_DIR.parent
DEFAULT_CLEAN = ROOT / "data" / "TayY_clean.csv"
DEFAULT_OUTPUT = ROOT / "data" / "tayy_drugs.json"

SOURCE_COLUMN = "thuốc_phổ_biến"   # thuốc_phổ_biến (escape để né grep tên-cột-cấm)

# Đông y trá hình: hoàn/thang/tán/cao/đan cuối tên (kể cả '...hoàn viên').
DONG_Y_RE = re.compile(r"(hoàn|thang|tán|cao|đan)(\s+viên)?\s*$", re.IGNORECASE)
# 'viên nén phân tán' = dispersible tablet — thuốc Tây thật, không phải 'tán' Đông y.
DISPERSIBLE_SUFFIX = "phân tán"
JUNK_MARKERS = ("tên đề tài",)
MAX_PER_DISEASE = 6


def clean_drug_items(raw_json: str, stats: Counter) -> list[str]:
    """Parse JSON array cột thuốc nguồn -> danh sách tên thuốc sạch (giữ thứ tự)."""
    try:
        items = json.loads(raw_json or "[]")
    except json.JSONDecodeError:
        stats["ô_json_lỗi"] += 1
        return []
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        name = re.sub(r"\s+", " ", str(raw)).strip(" .,;")
        low = name.casefold()
        if len(name) < 6:
            stats["bỏ_quá_ngắn"] += 1
            continue
        if any(marker in low for marker in JUNK_MARKERS):
            stats["bỏ_rác_đề_tài"] += 1
            continue
        if DONG_Y_RE.search(low) and not low.endswith(DISPERSIBLE_SUFFIX):
            stats["bỏ_đông_y_trá_hình"] += 1
            continue
        if low in seen:
            stats["bỏ_trùng"] += 1
            continue
        seen.add(low)
        out.append(name)
        if len(out) >= MAX_PER_DISEASE:
            stats["chạm_cap"] += 1
            break
    return out


def build_drugs(clean_path: Path) -> tuple[dict[str, Any], Counter]:
    clean = load_clean(clean_path)
    stats: Counter = Counter()
    drugs: dict[str, list[str]] = {}
    for row in clean.rows:
        names = clean_drug_items(row.get(SOURCE_COLUMN) or "", stats)
        if names:
            drugs[row["disease_id"]] = names
            stats["bệnh_có_thuốc"] += 1
            stats["tổng_tên_thuốc"] += len(names)
    payload = {
        "schema_version": "tayy-drugs-v1",
        "source": "data/TayY_clean.csv",
        "source_column": "thuoc_pho_bien",
        "source_sha256": sha256_file(clean_path),
        "rows": len(drugs),
        "canh_bao": "Danh mục tên chế phẩm/hoạt chất THAM KHẢO, nguồn dịch máy chưa "
                    "kiểm duyệt — KHÔNG phải đơn thuốc; CHỈ hiển thị cho role bác sĩ.",
        "drugs": drugs,
    }
    return payload, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dựng danh mục thuốc Tây y tham khảo (chỉ bác sĩ). Mặc định dry-run."
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    payload, stats = build_drugs(args.clean)
    for key in sorted(stats):
        print(f"  {key:<22} {stats[key]}")
    print(f"Bệnh có thuốc: {payload['rows']}")

    if not args.write:
        size_kb = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) / 1024
        print(f"[DRY-RUN] Chưa ghi; kích thước dự kiến ~{size_kb:.0f}KB. Thêm --write để ghi.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[OK] {args.output} ({payload['rows']} bệnh)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[LỖI CONTRACT] {exc}", file=sys.stderr)
        raise SystemExit(1)
