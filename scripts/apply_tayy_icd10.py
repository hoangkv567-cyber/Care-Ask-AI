#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize ICD-10 proposals/reviews into ``data/TayY_clean.csv`` safely.

Precedence is explicit:

* expert ``approved`` -> reviewed code + ``da_kiem_duyet``;
* expert ``unmappable_*`` -> blank code + ``khong_anh_xa``;
* expert ``rejected``/``needs_more_information`` -> blank code and unreviewed;
* otherwise a valid machine ``provisional_code`` (Tier A) may be materialized,
  but the status remains ``chua_kiem_duyet``;
* otherwise an LLM consensus code (``tayy_icd10_llm_proposals.jsonl``, sinh bởi
  scripts/llm_propose_tayy_icd10.py) may be materialized — vẫn
  ``chua_kiem_duyet`` (máy đề xuất, chờ chuyên gia);
* no decision/proposal -> blank code + ``chua_kiem_duyet``.

Tier A luôn thắng LLM khi hai bên lệch nhau (đếm ``conflict_tier_a_vs_llm``
trong stats để reviewer soi).

The command is a dry-run by default.  ``--apply`` validates the complete corpus,
writes a temporary CSV, verifies that only the two mapping fields changed,
backs up the old CSV under ``data/icd10/``, then atomically replaces it.  Running
it again with identical inputs performs zero writes and creates no new backup.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from review_tayy_icd10 import (  # noqa: E402
    DEFAULT_BACKUP_DIR,
    DEFAULT_CATALOG,
    DEFAULT_CATALOG_META,
    DEFAULT_CLEAN,
    DEFAULT_LLM_PROPOSALS,
    DEFAULT_PROPOSALS,
    DEFAULT_REVIEWS,
    MUTABLE_COLUMNS,
    UNMAPPABLE_DECISIONS,
    CatalogData,
    CleanData,
    ContractError,
    _display_path,
    load_catalog,
    load_clean,
    load_llm_proposals,
    load_proposals,
    load_reviews,
)


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

csv.field_size_limit(10**9)

STATUS_UNREVIEWED = "chua_kiem_duyet"
STATUS_REVIEWED = "da_kiem_duyet"
STATUS_UNMAPPABLE = "khong_anh_xa"
VALID_STATUSES = frozenset((STATUS_UNREVIEWED, STATUS_REVIEWED, STATUS_UNMAPPABLE))


def _desired_mapping(
    disease_id: str,
    proposals: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, str]],
    llm_proposals: dict[str, dict[str, Any]],
) -> tuple[str, str, str]:
    """Return ``(code, status, source)`` for one clean row."""

    proposal = proposals.get(disease_id)
    review = reviews.get(disease_id)
    provisional = str((proposal or {}).get("provisional_code") or "")
    llm_code = str((llm_proposals.get(disease_id) or {}).get("_code") or "")

    if review:
        decision = review["decision"]
        if decision == "approved":
            return review["icd10_code"], STATUS_REVIEWED, "expert_approved"
        if decision in UNMAPPABLE_DECISIONS:
            return "", STATUS_UNMAPPABLE, decision
        if decision in ("rejected", "needs_more_information"):
            # Do not keep displaying a code which the expert rejected or could
            # not substantiate.  A future proposal/review must supersede it.
            return "", STATUS_UNREVIEWED, decision
        # pending intentionally falls through to the current machine proposal.

    if provisional:
        return provisional, STATUS_UNREVIEWED, "machine_provisional"
    if llm_code:
        # Đồng thuận LLM (>=2 phiếu, đã validate hash/eligibility) — vẫn là mã
        # máy đề xuất: trạng thái giữ nguyên chưa kiểm duyệt.
        return llm_code, STATUS_UNREVIEWED, "machine_llm_consensus"
    return "", STATUS_UNREVIEWED, "none"


def _validate_current_pair(
    disease_id: str, row: dict[str, str], catalog: CatalogData
) -> tuple[str, str]:
    raw_code = (row.get("icd10_code") or "").strip()
    status = (row.get("trạng_thái_kiểm_duyệt") or "").strip()
    if status not in VALID_STATUSES:
        raise ContractError(
            f"{disease_id}: trạng_thái_kiểm_duyệt={status!r} không hợp lệ; "
            f"cần một trong {', '.join(sorted(VALID_STATUSES))}"
        )
    canonical = catalog.canonical_code(raw_code)
    if raw_code and not canonical:
        raise ContractError(f"{disease_id}: icd10_code hiện tại ngoài catalog: {raw_code!r}")
    if raw_code and raw_code.upper() != canonical:
        raise ContractError(
            f"{disease_id}: icd10_code hiện tại không canonical ({raw_code!r} -> {canonical!r})"
        )
    if status == STATUS_REVIEWED and not canonical:
        raise ContractError(f"{disease_id}: da_kiem_duyet nhưng icd10_code rỗng")
    if status == STATUS_UNMAPPABLE and canonical:
        raise ContractError(f"{disease_id}: khong_anh_xa nhưng icd10_code={canonical}")
    return canonical, status


def _is_orphaned_existing_value(
    current_code: str,
    current_status: str,
    proposal: dict[str, Any] | None,
    review: dict[str, str] | None,
    llm_proposal: dict[str, Any] | None,
) -> bool:
    """Whether applying would erase/downgrade a value with no sidecar authority."""

    if not current_code and current_status == STATUS_UNREVIEWED:
        return False
    decision = (review or {}).get("decision")
    if decision and decision != "pending":
        return False  # an explicit expert decision authorizes the transition
    if current_status in (STATUS_REVIEWED, STATUS_UNMAPPABLE):
        return True   # terminal state must have a non-pending review sidecar
    if current_code and not (proposal or llm_proposal):
        return True   # machine code must have its proposal/LLM sidecar lineage
    return False


def build_plan(
    clean: CleanData,
    catalog: CatalogData,
    proposals: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, str]],
    llm_proposals: dict[str, dict[str, Any]],
    *,
    reconcile_orphans: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter]:
    planned_rows: list[dict[str, str]] = []
    changes: list[dict[str, str]] = []
    stats: Counter = Counter()
    orphan_ids: list[str] = []

    for row in clean.rows:
        disease_id = row["disease_id"]
        current_code, current_status = _validate_current_pair(disease_id, row, catalog)
        desired_code, desired_status, source = _desired_mapping(
            disease_id, proposals, reviews, llm_proposals
        )
        proposal = proposals.get(disease_id)
        review = reviews.get(disease_id)
        llm_proposal = llm_proposals.get(disease_id)

        provisional = str((proposal or {}).get("provisional_code") or "")
        llm_code = str((llm_proposal or {}).get("_code") or "")
        if provisional and llm_code and provisional != llm_code:
            stats["conflict_tier_a_vs_llm"] += 1

        if (
            (current_code != desired_code or current_status != desired_status)
            and _is_orphaned_existing_value(
                current_code, current_status, proposal, review, llm_proposal
            )
            and not reconcile_orphans
        ):
            orphan_ids.append(disease_id)
            continue

        new_row = dict(row)
        new_row["icd10_code"] = desired_code
        new_row["trạng_thái_kiểm_duyệt"] = desired_status
        planned_rows.append(new_row)
        stats[f"final_status::{desired_status}"] += 1
        stats[f"source::{source}"] += 1
        if current_code != desired_code or current_status != desired_status:
            changes.append(
                {
                    "disease_id": disease_id,
                    "name": row.get("tên_bệnh", ""),
                    "old_code": current_code,
                    "new_code": desired_code,
                    "old_status": current_status,
                    "new_status": desired_status,
                    "source": source,
                }
            )

    if orphan_ids:
        preview = ", ".join(orphan_ids[:12])
        more = f" ... (+{len(orphan_ids) - 12})" if len(orphan_ids) > 12 else ""
        raise ContractError(
            f"Có {len(orphan_ids)} mapping/status hiện hữu không có lineage tương ứng trong "
            f"proposal/review sidecar: {preview}{more}. Từ chối xóa/hạ trạng thái mù. "
            "Nếu chủ đích lấy sidecar làm nguồn thật và dọn các orphan này, chạy lại với "
            "--reconcile-orphans."
        )

    if len(planned_rows) != len(clean.rows):
        raise ContractError(
            f"Lập kế hoạch thiếu dòng: {len(planned_rows)} != {len(clean.rows)}"
        )
    return planned_rows, changes, stats


def _assert_only_mapping_columns_changed(
    before: CleanData,
    planned_rows: list[dict[str, str]],
) -> None:
    if len(before.rows) != len(planned_rows):
        raise ContractError("Apply làm thay đổi số dòng")
    immutable = [c for c in before.fieldnames if c not in MUTABLE_COLUMNS]
    for index, (old, new) in enumerate(zip(before.rows, planned_rows), 2):
        for column in immutable:
            if (old.get(column) or "") != (new.get(column) or ""):
                raise ContractError(
                    f"Bất biến bị vi phạm dòng {index}, cột {column!r}: apply chỉ được đổi "
                    "icd10_code và trạng_thái_kiểm_duyệt"
                )


def _write_temp_csv(
    target: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> Path:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=fieldnames, extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
        return tmp_path
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _validate_temp_output(
    before: CleanData,
    temp_path: Path,
    expected_rows: list[dict[str, str]],
) -> None:
    after = load_clean(temp_path)
    if after.fieldnames != before.fieldnames:
        raise ContractError("CSV tạm làm thay đổi header hoặc thứ tự cột")
    if after.corpus_source_sha256 != before.corpus_source_sha256:
        raise ContractError(
            "CSV tạm làm thay đổi dữ liệu ngoài hai cột mapping "
            f"({before.corpus_source_sha256} -> {after.corpus_source_sha256})"
        )
    if [r["disease_id"] for r in after.rows] != [r["disease_id"] for r in before.rows]:
        raise ContractError("CSV tạm làm thay đổi thứ tự disease_id")
    for line_no, (actual, expected) in enumerate(zip(after.rows, expected_rows), 2):
        for column in MUTABLE_COLUMNS:
            if (actual.get(column) or "") != (expected.get(column) or ""):
                raise ContractError(
                    f"CSV tạm dòng {line_no}: {column} không khớp kế hoạch"
                )


def _atomic_apply(
    clean_path: Path,
    clean: CleanData,
    planned_rows: list[dict[str, str]],
    backup_dir: Path,
) -> Path:
    _assert_only_mapping_columns_changed(clean, planned_rows)
    temp_path = _write_temp_csv(clean_path, clean.fieldnames, planned_rows)
    try:
        _validate_temp_output(clean, temp_path, planned_rows)
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = backup_dir / f"{clean_path.stem}.bak.{stamp}{clean_path.suffix}"
        shutil.copy2(clean_path, backup)
        os.replace(temp_path, clean_path)
        return backup
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _print_plan(
    clean: CleanData,
    catalog: CatalogData,
    proposals: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, str]],
    llm_proposals: dict[str, dict[str, Any]],
    changes: list[dict[str, str]],
    stats: Counter,
) -> None:
    print(f"Clean rows              : {len(clean.rows)}")
    print(f"Corpus source SHA-256   : {clean.corpus_source_sha256}")
    print(f"Catalog codes           : {len(catalog.records)}")
    print(f"Catalog SHA-256         : {catalog.sha256}")
    print(f"Proposal rows           : {len(proposals)}")
    print(f"LLM consensus rows      : {len(llm_proposals)}")
    print(f"Review rows             : {len(reviews)}")
    print(f"Rows changing           : {len(changes)}")
    for key in sorted(stats):
        print(f"  {key:<45} {stats[key]}")
    if changes:
        print("\nVí dụ thay đổi:")
        for item in changes[:12]:
            print(
                f"  {item['disease_id']} {item['name'][:42]!r}: "
                f"{item['old_code'] or '∅'}/{item['old_status']} -> "
                f"{item['new_code'] or '∅'}/{item['new_status']} "
                f"[{item['source']}]"
            )
        if len(changes) > 12:
            print(f"  ... và {len(changes) - 12} dòng nữa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize proposal/review ICD-10 vào TayY_clean.csv. Mặc định dry-run."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Backup + ghi atomic vào CSV")
    mode.add_argument(
        "--check",
        action="store_true",
        help="Chỉ kiểm tra; exit 1 nếu CSV chưa khớp sidecars",
    )
    parser.add_argument(
        "--reconcile-orphans",
        action="store_true",
        help="Cho phép dọn code/status không có lineage sidecar (mặc định từ chối)",
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--catalog-meta", type=Path, default=DEFAULT_CATALOG_META)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--llm-proposals", type=Path, default=DEFAULT_LLM_PROPOSALS)
    parser.add_argument(
        "--no-llm-proposals",
        action="store_true",
        help="Bỏ qua sidecar đề xuất LLM (chỉ materialize Tier A + review chuyên gia)",
    )
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clean = load_clean(args.clean)
    catalog = load_catalog(args.catalog, args.catalog_meta)
    proposals = load_proposals(args.proposals, clean, catalog)
    reviews = load_reviews(args.reviews, clean, catalog, required=False)
    llm_proposals = (
        {}
        if args.no_llm_proposals
        else load_llm_proposals(args.llm_proposals, clean, catalog, proposals)
    )
    planned_rows, changes, stats = build_plan(
        clean,
        catalog,
        proposals,
        reviews,
        llm_proposals,
        reconcile_orphans=args.reconcile_orphans,
    )
    _assert_only_mapping_columns_changed(clean, planned_rows)
    _print_plan(clean, catalog, proposals, reviews, llm_proposals, changes, stats)

    if args.check:
        if changes:
            print(f"[CHECK FAIL] {len(changes)} dòng chưa khớp sidecars.", file=sys.stderr)
            return 1
        print("[CHECK OK] CSV đã khớp hoàn toàn với proposal/review sidecars.")
        return 0
    if not args.apply:
        print(f"\n[DRY-RUN] Chưa ghi gì; {len(changes)} dòng sẽ đổi. Thêm --apply để ghi.")
        return 0
    if not changes:
        print("\n[OK] CSV đã đúng; 0 thay đổi, không ghi lại/không tạo backup.")
        return 0

    backup = _atomic_apply(args.clean, clean, planned_rows, args.backup_dir)
    # Re-load after replacement: proves the materialized file still satisfies the
    # immutable-source fingerprint and exact desired mapping projection.
    applied = load_clean(args.clean)
    _validate_temp_output(clean, args.clean, planned_rows)
    print(f"\nBackup                 : {_display_path(backup)}")
    print(f"[OK] Đã materialize {len(changes)} dòng vào {_display_path(args.clean)}.")
    print(f"     Corpus source SHA-256 vẫn giữ: {applied.corpus_source_sha256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[LỖI CONTRACT] {exc}", file=sys.stderr)
        raise SystemExit(1)
