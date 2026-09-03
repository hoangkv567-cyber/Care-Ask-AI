#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and import expert ICD-10 reviews for ``data/TayY_clean.csv``.

This command never edits ``TayY_clean.csv``.  Machine proposals and expert
decisions deliberately live in separate files so regenerating proposals cannot
erase a human verdict.

Examples (run from the repository root)::

    python scripts/review_tayy_icd10.py validate
    python scripts/review_tayy_icd10.py validate --input data/reviewed_queue.csv
    python scripts/review_tayy_icd10.py import --input data/reviewed_queue.csv
    python scripts/review_tayy_icd10.py import --input data/reviewed_queue.csv --apply

``import`` is a dry-run unless ``--apply`` is present.  Existing non-pending
decisions are never replaced implicitly; pass ``--supersede-id TAYY-.....`` for
each intentional replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

csv.field_size_limit(10**9)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEAN = ROOT / "data" / "TayY_clean.csv"
DEFAULT_CATALOG = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.jsonl"
DEFAULT_CATALOG_META = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.meta.json"
DEFAULT_PROPOSALS = ROOT / "data" / "icd10" / "tayy_icd10_proposals.jsonl"
DEFAULT_LLM_PROPOSALS = ROOT / "data" / "icd10" / "tayy_icd10_llm_proposals.jsonl"
DEFAULT_REVIEWS = ROOT / "data" / "tayy_icd10_reviews.csv"
DEFAULT_BACKUP_DIR = ROOT / "data" / "icd10"

SCHEMA_VERSION = "1"
ID_RE = re.compile(r"^TAYY-\d{5}$")
MUTABLE_COLUMNS = frozenset(("icd10_code", "trạng_thái_kiểm_duyệt"))

REVIEW_FIELDS = [
    "schema_version",
    "disease_id",
    "row_sha256",
    "catalog_sha256",
    "decision",
    "icd10_code",
    "reviewer",
    "reviewed_at",
    "review_note",
]

REVIEW_DECISIONS = frozenset(
    (
        "pending",
        "approved",
        "rejected",
        "needs_more_information",
        "unmappable_not_disease",
        "unmappable_no_single_code",
    )
)
UNMAPPABLE_DECISIONS = frozenset(
    ("unmappable_not_disease", "unmappable_no_single_code")
)
NONTERMINAL_DECISIONS = frozenset(
    ("pending", "rejected", "needs_more_information")
)

PROPOSAL_BASE_FIELDS = frozenset(
    ("schema_version", "disease_id", "name", "provisional_code", "candidates")
)
PROPOSAL_SCHEMA_VERSIONS = frozenset((SCHEMA_VERSION, "tayy-icd10-proposal-v1"))
LLM_PROPOSAL_SCHEMA_VERSION = "tayy-icd10-llm-proposal-v1"
LLM_CODE_KINDS = frozenset(("in_top5", "outside_top5"))
LLM_NO_CODE_KINDS = frozenset(("none", "insufficient", "no_consensus", "ineligible_code"))


class ContractError(ValueError):
    """A sidecar or source file violates the ICD mapping contract."""


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_row_sha256(row: dict[str, str], fieldnames: Iterable[str]) -> str:
    """Hash every clean-row column except the two fields materialized by apply.

    The exact serialization is shared with the proposal generator.  It remains
    stable when the cleaner resets ICD/status, while any medical-content, ID, or
    quality-flag drift invalidates the review/proposal conservatively.
    """

    # Dictionary keys are sorted by the canonical JSON encoder, so an incidental
    # CSV column reorder does not invalidate expert work.  The field name alias
    # ``row_fingerprint`` is accepted when reading older queues, but its value
    # must match this current contract.
    payload = {
        name: row.get(name, "")
        for name in fieldnames
        if name not in MUTABLE_COLUMNS
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class CleanData:
    fieldnames: list[str]
    rows: list[dict[str, str]]
    by_id: dict[str, dict[str, str]]
    row_hashes: dict[str, str]
    corpus_source_sha256: str


@dataclass
class CatalogData:
    records: dict[str, dict[str, Any]]
    aliases: dict[str, str]
    sha256: str
    meta: dict[str, Any]

    def canonical_code(self, raw_code: Any) -> str:
        code = str(raw_code or "").strip().upper()
        if not code:
            return ""
        if code in self.records:
            return code
        alias = re.sub(r"[.\s]", "", code)
        return self.aliases.get(alias, "")


def load_clean(path: Path) -> CleanData:
    if not path.is_file():
        raise ContractError(f"Không thấy CSV sạch: {_display_path(path)}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required = {"disease_id", "tên_bệnh", *MUTABLE_COLUMNS}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ContractError(
            f"{_display_path(path)} thiếu cột bắt buộc: {', '.join(missing)}"
        )
    if len(fieldnames) != len(set(fieldnames)):
        raise ContractError(f"{_display_path(path)} có tên cột trùng")

    by_id: dict[str, dict[str, str]] = {}
    row_hashes: dict[str, str] = {}
    corpus = hashlib.sha256()
    for line_no, row in enumerate(rows, 2):
        disease_id = (row.get("disease_id") or "").strip()
        if not ID_RE.fullmatch(disease_id):
            raise ContractError(
                f"{_display_path(path)} dòng {line_no}: disease_id không hợp lệ {disease_id!r}"
            )
        if disease_id in by_id:
            raise ContractError(
                f"{_display_path(path)} dòng {line_no}: disease_id trùng {disease_id}"
            )
        digest = canonical_row_sha256(row, fieldnames)
        by_id[disease_id] = row
        row_hashes[disease_id] = digest
        corpus.update(disease_id.encode("ascii"))
        corpus.update(b"\0")
        corpus.update(digest.encode("ascii"))
        corpus.update(b"\n")

    return CleanData(fieldnames, rows, by_id, row_hashes, corpus.hexdigest())


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Không thấy {label}: {_display_path(path)}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Không đọc được {label} {_display_path(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} {_display_path(path)} phải là JSON object")
    return value


def _iter_json_records(path: Path, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    """Read JSONL, or a JSON array/object wrapper for forward-compatible tooling."""

    if not path.is_file():
        raise ContractError(f"Không thấy {label}: {_display_path(path)}")

    # JSONL is the canonical format.  A JSON array or {records/proposals: [...]} is
    # accepted to make review/apply resilient to harmless container changes.
    with path.open("r", encoding="utf-8-sig") as fh:
        first = ""
        while True:
            ch = fh.read(1)
            if not ch:
                break
            if not ch.isspace():
                first = ch
                break

    if first == "[":
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ContractError(f"JSON lỗi trong {_display_path(path)}: {exc}") from exc
        if not isinstance(data, list):
            raise ContractError(f"{_display_path(path)} phải chứa JSON array")
        for index, record in enumerate(data, 1):
            if not isinstance(record, dict):
                raise ContractError(f"{label} record #{index} không phải object")
            yield index, record
        return

    if first == "{":
        # A one-object JSON file can be distinguished from JSONL by parsing the
        # whole file.  If it has no wrapper key, fall back to ordinary JSONL.
        raw = path.read_text(encoding="utf-8-sig")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            wrapped = data.get("records", data.get("proposals"))
            if wrapped is not None:
                if not isinstance(wrapped, list):
                    raise ContractError(
                        f"{_display_path(path)}: records/proposals phải là array"
                    )
                for index, record in enumerate(wrapped, 1):
                    if not isinstance(record, dict):
                        raise ContractError(f"{label} record #{index} không phải object")
                    yield index, record
                return
            # A single proposal object is accepted as one record.
            if "disease_id" in data:
                yield 1, data
                return

    with path.open("r", encoding="utf-8-sig") as fh:
        record_no = 0
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record_no += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{_display_path(path)} dòng {line_no}: JSON lỗi: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ContractError(
                    f"{_display_path(path)} dòng {line_no}: record không phải object"
                )
            yield record_no, record


def load_catalog(path: Path, meta_path: Path) -> CatalogData:
    if not path.is_file():
        raise ContractError(f"Không thấy catalog ICD-10: {_display_path(path)}")
    meta = _load_json_object(meta_path, "catalog metadata")
    digest = sha256_file(path)
    expected = str(meta.get("catalog_sha256") or "").strip().lower()
    if not expected:
        raise ContractError(
            f"{_display_path(meta_path)} thiếu catalog_sha256; từ chối dùng catalog không khóa hash"
        )
    if expected != digest:
        raise ContractError(
            f"Catalog SHA-256 lệch: metadata={expected}, file={digest}. "
            "Hãy dựng lại catalog/meta cùng một lần."
        )

    records: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for record_no, record in _iter_json_records(path, "catalog ICD-10"):
        code = str(record.get("code") or "").strip().upper()
        title = str(
            record.get("title_vi")
            or record.get("who_title_en")
            or record.get("title_en")
            or ""
        ).strip()
        if not code:
            raise ContractError(f"Catalog record #{record_no} thiếu code")
        if not title:
            raise ContractError(f"Catalog code {code} thiếu title")
        if code in records:
            raise ContractError(f"Catalog có code trùng: {code}")
        records[code] = record
        alias = str(record.get("code_nodot") or re.sub(r"[.\s]", "", code)).upper()
        previous = aliases.get(alias)
        if previous and previous != code:
            raise ContractError(
                f"Catalog alias {alias} trỏ cả {previous} và {code}"
            )
        aliases[alias] = code

    if not records:
        raise ContractError(f"Catalog {_display_path(path)} rỗng")
    return CatalogData(records, aliases, digest, meta)


def _candidate_codes(raw_candidates: Any, where: str, catalog: CatalogData) -> set[str]:
    if not isinstance(raw_candidates, list):
        raise ContractError(f"{where}: candidates phải là JSON array")
    codes: set[str] = set()
    for index, candidate in enumerate(raw_candidates, 1):
        raw_code = candidate.get("code") if isinstance(candidate, dict) else candidate
        code = catalog.canonical_code(raw_code)
        if not code:
            raise ContractError(
                f"{where}: candidate #{index} có code ngoài catalog: {raw_code!r}"
            )
        if code in codes:
            raise ContractError(f"{where}: candidate code trùng {code}")
        codes.add(code)
    return codes


def load_proposals(
    path: Path, clean: CleanData, catalog: CatalogData
) -> dict[str, dict[str, Any]]:
    proposals: dict[str, dict[str, Any]] = {}
    for record_no, raw in _iter_json_records(path, "proposal sidecar"):
        where = f"{_display_path(path)} record #{record_no}"
        missing = sorted(PROPOSAL_BASE_FIELDS - set(raw))
        if missing:
            raise ContractError(f"{where}: thiếu field {', '.join(missing)}")
        schema_version = str(raw.get("schema_version"))
        if schema_version not in PROPOSAL_SCHEMA_VERSIONS:
            raise ContractError(
                f"{where}: schema_version={raw.get('schema_version')!r}, cần một trong "
                f"{', '.join(sorted(PROPOSAL_SCHEMA_VERSIONS))}"
            )
        disease_id = str(raw.get("disease_id") or "").strip()
        if disease_id in proposals:
            raise ContractError(f"{where}: disease_id trùng {disease_id}")
        if disease_id not in clean.by_id:
            raise ContractError(f"{where}: disease_id không có trong clean: {disease_id}")
        row_sha = str(raw.get("row_sha256") or "").lower()
        row_fingerprint = str(raw.get("row_fingerprint") or "").lower()
        if row_sha and row_fingerprint and row_sha != row_fingerprint:
            raise ContractError(
                f"{where}: row_sha256 và row_fingerprint mâu thuẫn cho {disease_id}"
            )
        row_hash = row_sha or row_fingerprint
        if not row_hash:
            raise ContractError(f"{where}: thiếu row_sha256/row_fingerprint")
        if row_hash != clean.row_hashes[disease_id]:
            raise ContractError(
                f"{where}: row_sha256 stale cho {disease_id}; hãy sinh lại proposal"
            )
        provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
        catalog_hash_top = str(raw.get("catalog_sha256") or "").lower()
        catalog_hash_provenance = str(provenance.get("catalog_sha256") or "").lower()
        if (
            catalog_hash_top
            and catalog_hash_provenance
            and catalog_hash_top != catalog_hash_provenance
        ):
            raise ContractError(
                f"{where}: catalog_sha256 top-level và provenance mâu thuẫn"
            )
        catalog_hash = catalog_hash_top or catalog_hash_provenance
        if not catalog_hash:
            raise ContractError(f"{where}: thiếu catalog_sha256 (top-level hoặc provenance)")
        if catalog_hash != catalog.sha256:
            raise ContractError(
                f"{where}: catalog_sha256 stale cho {disease_id}; hãy sinh lại proposal"
            )
        if str(raw.get("name") or "") != (clean.by_id[disease_id].get("tên_bệnh") or ""):
            raise ContractError(f"{where}: name không khớp clean cho {disease_id}")
        algorithm_version = str(
            raw.get("algorithm_version") or provenance.get("mapper_version") or ""
        ).strip()
        if not algorithm_version:
            raise ContractError(
                f"{where}: thiếu algorithm_version (top-level hoặc provenance.mapper_version)"
            )
        excluded = provenance.get("row_fingerprint_excluded_columns")
        if excluded is not None:
            if not isinstance(excluded, list) or set(excluded) != set(MUTABLE_COLUMNS):
                raise ContractError(
                    f"{where}: fingerprint phải loại đúng hai cột "
                    "icd10_code,trạng_thái_kiểm_duyệt"
                )

        candidates = _candidate_codes(raw.get("candidates"), where, catalog)
        provisional_raw = raw.get("provisional_code")
        provisional = catalog.canonical_code(provisional_raw)
        if str(provisional_raw or "").strip() and not provisional:
            raise ContractError(
                f"{where}: provisional_code ngoài catalog: {provisional_raw!r}"
            )
        if provisional:
            if candidates and provisional not in candidates:
                raise ContractError(
                    f"{where}: provisional_code {provisional} không có trong candidates"
                )
            cat_record = catalog.records[provisional]
            if cat_record.get("auto_single_code_eligible") is not True:
                raise ContractError(
                    f"{where}: provisional_code {provisional} không đủ điều kiện "
                    "auto_single_code_eligible"
                )
        record = dict(raw)
        record["provisional_code"] = provisional
        # Normalized internal aliases make downstream apply independent of the
        # flat-vs-provenance proposal container without rewriting the sidecar.
        record["_row_sha256"] = row_hash
        record["_catalog_sha256"] = catalog_hash
        record["_algorithm_version"] = algorithm_version
        proposals[disease_id] = record
    return proposals


def load_llm_proposals(
    path: Path,
    clean: CleanData,
    catalog: CatalogData,
    proposals: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Nạp sidecar đề xuất LLM (scripts/llm_propose_tayy_icd10.py).

    File vắng -> {} (sidecar tùy chọn, như reviews). File là append-log: record
    SAU CÙNG của mỗi disease_id thắng (resume ghi record mới cho dòng đã đổi).
    Chỉ trả về record MANG MÃ (kind in_top5/outside_top5) — kind còn lại không
    materialize gì. Record mang mã mà stale (row/catalog hash lệch) là
    ``ContractError``: apply từ chối chạy trên dữ liệu trôi, hãy chạy lại
    llm_propose_tayy_icd10.py để sinh record fresh.
    """

    if not path.exists():
        return {}
    if not path.is_file():
        raise ContractError(f"LLM proposal sidecar không phải file: {_display_path(path)}")

    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    for record_no, raw in _iter_json_records(path, "LLM proposal sidecar"):
        where = f"{_display_path(path)} record #{record_no}"
        if str(raw.get("schema_version")) != LLM_PROPOSAL_SCHEMA_VERSION:
            raise ContractError(
                f"{where}: schema_version={raw.get('schema_version')!r}, cần "
                f"{LLM_PROPOSAL_SCHEMA_VERSION}"
            )
        disease_id = str(raw.get("disease_id") or "").strip()
        if not ID_RE.fullmatch(disease_id):
            raise ContractError(f"{where}: disease_id không hợp lệ {disease_id!r}")
        latest[disease_id] = (record_no, raw)

    out: dict[str, dict[str, Any]] = {}
    for disease_id, (record_no, raw) in latest.items():
        where = f"{_display_path(path)} record #{record_no}"
        consensus = raw.get("consensus") if isinstance(raw.get("consensus"), dict) else {}
        kind = str(consensus.get("kind") or "")
        if kind not in (LLM_CODE_KINDS | LLM_NO_CODE_KINDS):
            raise ContractError(f"{where}: consensus.kind={kind!r} không hợp lệ")
        if kind in LLM_NO_CODE_KINDS:
            continue                      # không có gì để materialize

        if disease_id not in clean.by_id:
            raise ContractError(f"{where}: disease_id không có trong clean: {disease_id}")
        row_hash = str(raw.get("row_sha256") or "").lower()
        if not row_hash:
            raise ContractError(f"{where}: thiếu row_sha256")
        if row_hash != clean.row_hashes[disease_id]:
            raise ContractError(
                f"{where}: row_sha256 stale cho {disease_id}; chạy lại "
                "scripts/llm_propose_tayy_icd10.py"
            )
        catalog_hash = str(raw.get("catalog_sha256") or "").lower()
        if not catalog_hash:
            raise ContractError(f"{where}: thiếu catalog_sha256")
        if catalog_hash != catalog.sha256:
            raise ContractError(
                f"{where}: catalog_sha256 stale cho {disease_id}; chạy lại "
                "scripts/llm_propose_tayy_icd10.py theo catalog hiện tại"
            )

        # Dòng exact-conflict (tên-exact và lead-exact trỏ 2 bộ mã RỜI NHAU) là ca
        # nhập nhằng danh tính — mã máy (kể cả LLM đồng thuận) không được materialize;
        # chỉ chuyên gia phán qua review sidecar. Bỏ qua record, không phải lỗi.
        base_proposal = proposals.get(disease_id) or {}
        base_exact = (base_proposal.get("components") or {}).get("exact") or {}
        if base_exact.get("conflict"):
            continue

        raw_code = consensus.get("code")
        code = catalog.canonical_code(raw_code)
        if not code:
            raise ContractError(f"{where}: consensus.code ngoài catalog: {raw_code!r}")
        if catalog.records[code].get("auto_single_code_eligible") is not True:
            raise ContractError(
                f"{where}: consensus.code {code} không auto_single_code_eligible"
            )
        proposal = proposals.get(disease_id)
        if kind == "in_top5":
            if proposal is None:
                raise ContractError(f"{where}: in_top5 nhưng không có proposal gốc")
            candidate_codes = _candidate_codes(
                proposal.get("candidates"), where, catalog
            )
            if code not in candidate_codes:
                raise ContractError(
                    f"{where}: kind=in_top5 nhưng {code} không nằm trong candidates gốc"
                )
        agreement = str(consensus.get("agreement") or "")
        try:
            agree_count = int(agreement.split("/", 1)[0])
        except ValueError:
            raise ContractError(f"{where}: agreement không đọc được: {agreement!r}")
        if agree_count < 2:
            raise ContractError(
                f"{where}: agreement {agreement!r} < 2 phiếu — không đủ đồng thuận để materialize"
            )

        record = dict(raw)
        record["_code"] = code
        out[disease_id] = record
    return out


def _iso_datetime_ok(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return "T" in value and parsed.tzinfo is not None
    except (TypeError, ValueError):
        return False


def _raw_review_value(row: dict[str, Any], primary: str, *aliases: str) -> str:
    value = row.get(primary)
    if value is not None and str(value).strip():
        return str(value).strip()
    for alias in aliases:
        value = row.get(alias)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_review_record(
    raw: dict[str, Any],
    where: str,
    clean: CleanData,
    catalog: CatalogData,
    *,
    allow_blank_decision: bool,
) -> dict[str, str] | None:
    disease_id = _raw_review_value(raw, "disease_id")
    decision = _raw_review_value(raw, "decision", "review_decision").lower()
    if not decision and allow_blank_decision:
        return None
    if not disease_id:
        raise ContractError(f"{where}: thiếu disease_id")
    if disease_id not in clean.by_id:
        raise ContractError(f"{where}: disease_id không có trong clean: {disease_id}")
    if decision not in REVIEW_DECISIONS:
        raise ContractError(
            f"{where}: decision {decision!r} không hợp lệ; cần một trong "
            f"{', '.join(sorted(REVIEW_DECISIONS))}"
        )

    schema_version = _raw_review_value(raw, "schema_version") or SCHEMA_VERSION
    if schema_version != SCHEMA_VERSION:
        raise ContractError(
            f"{where}: schema_version={schema_version!r}, cần {SCHEMA_VERSION}"
        )
    row_sha = _raw_review_value(raw, "row_sha256").lower()
    row_fingerprint = _raw_review_value(raw, "row_fingerprint").lower()
    if row_sha and row_fingerprint and row_sha != row_fingerprint:
        raise ContractError(f"{where}: row_sha256 và row_fingerprint mâu thuẫn")
    row_hash = row_sha or row_fingerprint
    if not row_hash:
        raise ContractError(f"{where}: thiếu row_sha256")
    if row_hash != clean.row_hashes[disease_id]:
        raise ContractError(
            f"{where}: row_sha256 stale cho {disease_id}; không nhập review trên nội dung đã đổi"
        )
    catalog_hash = _raw_review_value(raw, "catalog_sha256").lower()
    # The mapper's spreadsheet queue contains row_fingerprint but intentionally
    # keeps global catalog provenance in its report/proposal.  On import we lock
    # the canonical review to the already hash-validated current catalog.
    if not catalog_hash and allow_blank_decision:
        catalog_hash = catalog.sha256
    if not catalog_hash:
        raise ContractError(f"{where}: thiếu catalog_sha256")
    if catalog_hash != catalog.sha256:
        raise ContractError(
            f"{where}: catalog_sha256 stale cho {disease_id}; cần review theo catalog hiện tại"
        )

    raw_code = _raw_review_value(raw, "icd10_code", "approved_code")
    code = catalog.canonical_code(raw_code)
    if raw_code and not code:
        raise ContractError(f"{where}: icd10_code ngoài catalog: {raw_code!r}")
    reviewer = _raw_review_value(raw, "reviewer")
    reviewed_at = _raw_review_value(raw, "reviewed_at")
    note = _raw_review_value(raw, "review_note", "note")

    if decision == "approved":
        if not code:
            raise ContractError(f"{where}: approved bắt buộc có đúng một icd10_code")
    elif code:
        raise ContractError(f"{where}: decision={decision} bắt buộc để trống icd10_code")

    if decision != "pending":
        if not reviewer:
            raise ContractError(f"{where}: decision={decision} thiếu reviewer")
        if not reviewed_at:
            raise ContractError(f"{where}: decision={decision} thiếu reviewed_at")
        if not _iso_datetime_ok(reviewed_at):
            raise ContractError(
                f"{where}: reviewed_at phải là ISO-8601 datetime có múi giờ: {reviewed_at!r}"
            )
    elif reviewer or reviewed_at or note or code:
        raise ContractError(
            f"{where}: pending không được mang code/reviewer/reviewed_at/review_note"
        )

    if decision in (UNMAPPABLE_DECISIONS | {"rejected", "needs_more_information"}) and not note:
        raise ContractError(f"{where}: decision={decision} bắt buộc có review_note")

    return {
        "schema_version": SCHEMA_VERSION,
        "disease_id": disease_id,
        "row_sha256": row_hash,
        "catalog_sha256": catalog_hash,
        "decision": decision,
        "icd10_code": code,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "review_note": note,
    }


def load_reviews(
    path: Path,
    clean: CleanData,
    catalog: CatalogData,
    *,
    required: bool = False,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        if required:
            raise ContractError(f"Không thấy review sidecar: {_display_path(path)}")
        return {}
    if not path.is_file():
        raise ContractError(f"Review sidecar không phải file: {_display_path(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = set(reader.fieldnames or [])
        missing_headers = sorted(set(REVIEW_FIELDS) - headers)
        if missing_headers:
            raise ContractError(
                f"{_display_path(path)} thiếu cột review: {', '.join(missing_headers)}"
            )
        out: dict[str, dict[str, str]] = {}
        for line_no, raw in enumerate(reader, 2):
            record = normalize_review_record(
                raw,
                f"{_display_path(path)} dòng {line_no}",
                clean,
                catalog,
                allow_blank_decision=False,
            )
            assert record is not None
            disease_id = record["disease_id"]
            if disease_id in out:
                raise ContractError(
                    f"{_display_path(path)} dòng {line_no}: disease_id trùng {disease_id}"
                )
            out[disease_id] = record
    return out


def load_review_queue(
    path: Path, clean: CleanData, catalog: CatalogData
) -> tuple[dict[str, dict[str, str]], int]:
    if not path.is_file():
        raise ContractError(f"Không thấy CSV queue: {_display_path(path)}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = set(reader.fieldnames or [])
        if "disease_id" not in headers:
            raise ContractError(f"{_display_path(path)} thiếu cột disease_id")
        if "decision" not in headers and "review_decision" not in headers:
            raise ContractError(f"{_display_path(path)} thiếu cột decision")
        if "row_sha256" not in headers and "row_fingerprint" not in headers:
            raise ContractError(
                f"{_display_path(path)} bắt buộc có row_sha256 hoặc row_fingerprint"
            )

        out: dict[str, dict[str, str]] = {}
        skipped_blank = 0
        seen_ids: set[str] = set()
        for line_no, raw in enumerate(reader, 2):
            disease_id = _raw_review_value(raw, "disease_id")
            if disease_id:
                if disease_id in seen_ids:
                    raise ContractError(
                        f"{_display_path(path)} dòng {line_no}: disease_id trùng {disease_id}"
                    )
                seen_ids.add(disease_id)
            record = normalize_review_record(
                raw,
                f"{_display_path(path)} dòng {line_no}",
                clean,
                catalog,
                allow_blank_decision=True,
            )
            if record is None:
                skipped_blank += 1
                continue
            out[record["disease_id"]] = record
    return out, skipped_blank


def _reviews_equal(left: dict[str, str], right: dict[str, str]) -> bool:
    return all((left.get(field) or "") == (right.get(field) or "") for field in REVIEW_FIELDS)


def merge_reviews(
    existing: dict[str, dict[str, str]],
    incoming: dict[str, dict[str, str]],
    supersede_ids: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    merged = {disease_id: dict(record) for disease_id, record in existing.items()}
    stats = {"inserted": 0, "updated_pending": 0, "superseded": 0, "unchanged": 0}
    unused_supersede = set(supersede_ids)

    for disease_id, new in incoming.items():
        old = merged.get(disease_id)
        if old is None:
            merged[disease_id] = dict(new)
            stats["inserted"] += 1
            continue
        if _reviews_equal(old, new):
            stats["unchanged"] += 1
            unused_supersede.discard(disease_id)
            continue
        if old["decision"] == "pending":
            merged[disease_id] = dict(new)
            stats["updated_pending"] += 1
            unused_supersede.discard(disease_id)
            continue
        if disease_id not in supersede_ids:
            raise ContractError(
                f"Review conflict tại {disease_id}: hiện là {old['decision']}"
                f"{(' ' + old['icd10_code']) if old['icd10_code'] else ''}, incoming là "
                f"{new['decision']}{(' ' + new['icd10_code']) if new['icd10_code'] else ''}. "
                f"Muốn thay có chủ đích, thêm --supersede-id {disease_id}."
            )
        if not new.get("review_note"):
            raise ContractError(
                f"Review supersede tại {disease_id} bắt buộc có review_note giải thích lý do."
            )
        merged[disease_id] = dict(new)
        stats["superseded"] += 1
        unused_supersede.discard(disease_id)

    if unused_supersede:
        raise ContractError(
            "--supersede-id không ứng với conflict nào: " + ", ".join(sorted(unused_supersede))
        )
    return merged, stats


def _atomic_write_reviews(
    path: Path,
    reviews: dict[str, dict[str, str]],
    backup_dir: Path,
) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    backup_path: Path | None = None
    try:
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS, extrasaction="raise")
            writer.writeheader()
            for disease_id in sorted(reviews):
                writer.writerow(reviews[disease_id])

        # Parse the complete temporary artifact before it is allowed to replace
        # the canonical sidecar.  Structural/source validation already happened.
        with tmp_path.open("r", encoding="utf-8-sig", newline="") as fh:
            check = list(csv.DictReader(fh))
        if len(check) != len(reviews):
            raise ContractError(
                f"Tệp review tạm ghi thiếu dòng: {len(check)} != {len(reviews)}"
            )

        if path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = backup_dir / f"{path.stem}.bak.{stamp}{path.suffix}"
            shutil.copy2(path, backup_path)
        os.replace(tmp_path, path)
        return backup_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _decision_counts(reviews: dict[str, dict[str, str]]) -> dict[str, int]:
    counts = {decision: 0 for decision in sorted(REVIEW_DECISIONS)}
    for record in reviews.values():
        counts[record["decision"]] += 1
    return counts


def _print_common_summary(
    clean: CleanData,
    catalog: CatalogData,
    proposals: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, str]],
) -> None:
    provisional = sum(1 for p in proposals.values() if p.get("provisional_code"))
    print(f"Clean rows              : {len(clean.rows)}")
    print(f"Corpus source SHA-256   : {clean.corpus_source_sha256}")
    print(f"Catalog codes           : {len(catalog.records)}")
    print(f"Catalog SHA-256         : {catalog.sha256}")
    print(f"Proposal rows           : {len(proposals)} (provisional={provisional})")
    print(f"Canonical review rows   : {len(reviews)}")
    for decision, count in _decision_counts(reviews).items():
        if count:
            print(f"  review::{decision:<26} {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate/import review chuyên gia cho ánh xạ ICD-10 TayY. Import mặc định dry-run."
    )
    parser.add_argument("command", choices=("validate", "import"))
    parser.add_argument("--input", type=Path, help="CSV queue đã được chuyên gia điền")
    parser.add_argument("--apply", action="store_true", help="Ghi review sidecar (import mặc định dry-run)")
    parser.add_argument(
        "--supersede-id",
        action="append",
        default=[],
        metavar="TAYY-#####",
        help="Cho phép thay có chủ đích review non-pending hiện hữu; lặp cờ cho nhiều ID",
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--catalog-meta", type=Path, default=DEFAULT_CATALOG_META)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import" and args.input is None:
        raise ContractError("Lệnh import bắt buộc có --input <reviewed_queue.csv>")
    if args.command == "validate" and args.apply:
        raise ContractError("validate là chỉ đọc; không dùng --apply")
    if args.command == "validate" and args.supersede_id:
        raise ContractError("--supersede-id chỉ dùng với import")

    clean = load_clean(args.clean)
    catalog = load_catalog(args.catalog, args.catalog_meta)
    proposals = load_proposals(args.proposals, clean, catalog)
    reviews = load_reviews(args.reviews, clean, catalog, required=False)
    _print_common_summary(clean, catalog, proposals, reviews)

    if args.command == "validate":
        if args.input:
            incoming, skipped = load_review_queue(args.input, clean, catalog)
            print(f"Input queue valid        : {len(incoming)} decision(s), {skipped} dòng để trống")
        print("[OK] Catalog, proposals, clean fingerprints và reviews hợp lệ.")
        return 0

    supersede_ids = {str(value).strip() for value in args.supersede_id}
    bad_ids = sorted(value for value in supersede_ids if not ID_RE.fullmatch(value))
    if bad_ids:
        raise ContractError("--supersede-id không hợp lệ: " + ", ".join(bad_ids))
    incoming, skipped = load_review_queue(args.input, clean, catalog)
    merged, stats = merge_reviews(reviews, incoming, supersede_ids)
    changes = stats["inserted"] + stats["updated_pending"] + stats["superseded"]
    print(f"Input decisions         : {len(incoming)} ({skipped} dòng decision trống được bỏ qua)")
    print(
        "Merge plan             : "
        f"insert={stats['inserted']}, pending->review={stats['updated_pending']}, "
        f"supersede={stats['superseded']}, unchanged={stats['unchanged']}"
    )

    if not args.apply:
        print(f"[DRY-RUN] Chưa ghi gì; {changes} review sẽ thay đổi. Thêm --apply để ghi.")
        return 0
    if not changes:
        print("[OK] Review sidecar đã đúng; 0 thay đổi, không ghi lại/không tạo backup.")
        return 0

    backup = _atomic_write_reviews(args.reviews, merged, args.backup_dir)
    if backup:
        print(f"Backup                 : {_display_path(backup)}")
    print(f"[OK] Đã ghi {_display_path(args.reviews)} ({len(merged)} review; {changes} thay đổi).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[LỖI CONTRACT] {exc}", file=sys.stderr)
        raise SystemExit(1)
