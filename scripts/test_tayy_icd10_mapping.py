#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-corpus regression for the TayY -> VN ICD-10 mapping pipeline.

This is deliberately a standalone executable rather than a sampled unit test.
It validates all 8,724 clean rows, all 15,844 catalog records, every proposal,
and every review when the canonical review sidecar exists.  Materialization is
exercised only on a copy inside ``TemporaryDirectory``; repository artifacts
are hashed before and after the run and are never written by this script.

Run from any directory::

    python scripts/test_tayy_icd10_mapping.py

The process exits non-zero on the first violated invariant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_tayy_icd10 as apply_module  # noqa: E402
from apply_tayy_icd10 import (  # noqa: E402
    STATUS_REVIEWED,
    STATUS_UNMAPPABLE,
    STATUS_UNREVIEWED,
    VALID_STATUSES,
    build_plan,
)
from review_tayy_icd10 import (  # noqa: E402
    LLM_CODE_KINDS,
    MUTABLE_COLUMNS,
    UNMAPPABLE_DECISIONS,
    CatalogData,
    CleanData,
    ContractError,
    canonical_row_sha256 as production_row_sha256,
    load_clean,
    load_llm_proposals,
    load_reviews,
)


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

csv.field_size_limit(10**9)

DEFAULT_CLEAN = ROOT / "data" / "TayY_clean.csv"
DEFAULT_CATALOG = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.jsonl"
DEFAULT_CATALOG_META = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.meta.json"
DEFAULT_PROPOSALS = ROOT / "data" / "icd10" / "tayy_icd10_proposals.jsonl"
DEFAULT_REVIEW_QUEUE = ROOT / "data" / "icd10" / "tayy_icd10_review_queue.csv"
DEFAULT_REVIEWS = ROOT / "data" / "tayy_icd10_reviews.csv"
DEFAULT_LLM_PROPOSALS = ROOT / "data" / "icd10" / "tayy_icd10_llm_proposals.jsonl"
DEFAULT_CLEAN_REPORT = ROOT / "data" / "tayy_clean_report.json"
DEFAULT_MAPPING_REPORT = ROOT / "data" / "tayy_icd10_report.json"
DEFAULT_SOURCE = ROOT / "data" / "TayY.csv"
DEFAULT_QUARANTINE = ROOT / "data" / "TayY_quarantine.csv"

EXPECTED_CLEAN_ROWS = 8_724
EXPECTED_CLEAN_COLUMNS = 25
EXPECTED_SOURCE_ROWS = 8_900
EXPECTED_SOURCE_COLUMNS = 18
EXPECTED_QUARANTINE_ROWS = 176
EXPECTED_CATALOG_ROWS = 15_844
EXPECTED_DUPLICATE_ROWS = 958
EXPECTED_DUPLICATE_GROUPS = 423
EXPECTED_PROPOSAL_SCHEMA = "tayy-icd10-proposal-v1"
EXPECTED_CATALOG_SHA256 = (
    "e79a61fd521ab132d7f03c12a1824177505c24fad055d50ebc06625c5ef74604"
)
EXPECTED_SOURCE_SHA256 = (
    "b8270f047e8ef0ac849db639bdbb5ffcceeef7b58dd8910699a18710ca3f1035"
)
EXPECTED_QUARANTINE_SHA256 = (
    "6fb4761b09ca9c2feef2f6e076ecaec018732c199775342fc75960dd0dc19cff"
)

DUPLICATE_FLAG = "trung_ten_khac_noi_dung"
TOP_K = 5
CODE_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,2})?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CLEAN_FIELDS = [
    "disease_id",
    "tên_bệnh",
    "mô_tả_bệnh",
    "loại_bệnh",
    "cách_phòng_tránh",
    "điều_trị_tách_từ_phòng_tránh",
    "nguyên_nhân",
    "triệu_chứng",
    "đối_tượng_dễ_mắc_bệnh",
    "bệnh_đi_kèm",
    "phương_pháp",
    "khoa_điều_trị",
    "tỉ_lệ_chữa_khỏi",
    "tỉ_lệ_chữa_khỏi_min",
    "tỉ_lệ_chữa_khỏi_max",
    "kiểm_tra",
    "nên_ăn_thực_phẩm_chứa",
    "không_nên_ăn_thực_phẩm_chứa",
    "đề_xuất_món_ăn",
    "đề_xuất_thuốc",
    "thuốc_phổ_biến",
    "thông_tin_thuốc",
    "icd10_code",
    "trạng_thái_kiểm_duyệt",
    "cờ_chất_lượng",
]

JSON_ARRAY_COLUMNS = (
    "loại_bệnh",
    "triệu_chứng",
    "bệnh_đi_kèm",
    "kiểm_tra",
    "nên_ăn_thực_phẩm_chứa",
    "không_nên_ăn_thực_phẩm_chứa",
    "đề_xuất_món_ăn",
    "đề_xuất_thuốc",
    "thuốc_phổ_biến",
)
EXPECTED_JSON_NONEMPTY = {
    "loại_bệnh": 8_685,
    "triệu_chứng": 8_682,
    "bệnh_đi_kèm": 8_445,
    "kiểm_tra": 8_720,
    "nên_ăn_thực_phẩm_chứa": 5_505,
    "không_nên_ăn_thực_phẩm_chứa": 5_505,
    "đề_xuất_món_ăn": 5_505,
    "đề_xuất_thuốc": 7_555,
    "thuốc_phổ_biến": 7_568,
}
EXPECTED_DUPLICATE_GROUP_SIZE_DISTRIBUTION = {
    "2": 369,
    "3": 38,
    "4": 6,
    "5": 6,
    "6": 1,
    "11": 1,
    "15": 1,
    "20": 1,
}

PROPOSAL_FIELDS = {
    "schema_version",
    "disease_id",
    "source_csv_line",
    "row_sha256",
    "row_fingerprint",
    "source_state",
    "name",
    "evidence",
    "proposal_tier",
    "provisional_code",
    "requires_human_review",
    "decision_reasons",
    "components",
    "candidates",
    "provenance",
}
EVIDENCE_FIELDS = {
    "description",
    "types",
    "symptoms",
    "tests",
    "comorbidities",
    "department",
    "cause",
    "method",
}
CANDIDATE_FIELDS = {
    "rank",
    "code",
    "title_vi",
    "title_en",
    "category_vi",
    "guidance_vi",
    "retrieved_by",
    "score",
    "components",
    "eligibility",
    "source_audit",
}
CANDIDATE_COMPONENT_FIELDS = {
    "bm25",
    "bm25_by_tayy_field",
    "name_exact",
    "name_exact_variant_types",
    "lead_exact",
    "lead_exact_variant_types",
    "exact_rank_bonus",
}
CANDIDATE_ELIGIBILITY_FIELDS = {
    "auto_single_code_eligible",
    "codable",
    "morbidity_eligible",
    "primary_eligible",
    "primary_eligibility",
    "dagger",
    "asterisk",
    "dual_coding_marker",
    "pair_resolution_required",
    "who_references",
    "restrictions",
}
CANDIDATE_SOURCE_AUDIT_FIELDS = {
    "source_page",
    "source_code_raw",
    "source_anomaly",
    "source_title_en_status",
    "source_title_en_warning",
}
BM25_TAYY_FIELDS = {
    "name",
    "lead",
    "description",
    "types",
    "symptoms",
    "tests",
    "comorbidities",
    "department",
    "cause",
    "method",
}
EXACT_RANK_BONUS = 10_000.0

REVIEW_COLUMNS = [
    "priority",
    "disease_id",
    "tên_bệnh",
    "mô_tả_bệnh",
    "lead",
    "loại_bệnh",
    "triệu_chứng",
    "kiểm_tra",
    "khoa_điều_trị",
    "nguyên_nhân",
    "bệnh_đi_kèm",
    "phương_pháp",
    "proposal_tier",
    "provisional_code",
    "decision_reasons",
    "exact_name_codes",
    "exact_lead_codes",
    "top_candidates",
    "cờ_chất_lượng",
    "ui_flag",
    "tcm_flag",
    "low_info_flag",
    "exact_conflict",
    "current_icd10_code",
    "current_review_status",
    "decision",
    "icd10_code",
    "reviewer",
    "reviewed_at",
    "review_note",
    "row_sha256",
    "row_fingerprint",
]
REVIEW_EDIT_COLUMNS = (
    "decision",
    "icd10_code",
    "reviewer",
    "reviewed_at",
    "review_note",
)
EXPECTED_ROW_FINGERPRINT_ALGORITHM = (
    "sha256(utf8(json-object-of-raw-row;exclude=icd10_code,"
    "trạng_thái_kiểm_duyệt;ensure_ascii=false;sort_keys=true;"
    "separators=(',',':')))"
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row_sha256(
    row: Mapping[str, str], fieldnames: Iterable[str]
) -> str:
    """Independent implementation of the mapper/apply fingerprint contract."""

    payload = {
        field: row.get(field, "")
        for field in fieldnames
        if field not in MUTABLE_COLUMNS
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_corpus_sha256(
    rows: Sequence[Mapping[str, str]], row_hashes: Mapping[str, str]
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        disease_id = row["disease_id"]
        digest.update(disease_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(row_hashes[disease_id].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class CleanCorpus:
    data: CleanData
    parsed_lists: dict[str, dict[str, list[str]]]
    flags: dict[str, tuple[str, ...]]
    json_nonempty_counts: dict[str, int]
    raw_sha256: str


def load_clean_contract(path: Path, label: str) -> CleanCorpus:
    require(path.is_file(), f"Thiếu {label}: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        require(
            fieldnames == EXPECTED_CLEAN_FIELDS,
            f"{label}: schema/order cột lệch; gặp {fieldnames!r}",
        )
        rows = list(reader)

    require(
        len(rows) == EXPECTED_CLEAN_ROWS,
        f"{label}: chờ {EXPECTED_CLEAN_ROWS} dòng, gặp {len(rows)}",
    )
    require(
        len(fieldnames) == EXPECTED_CLEAN_COLUMNS,
        f"{label}: chờ {EXPECTED_CLEAN_COLUMNS} cột, gặp {len(fieldnames)}",
    )

    by_id: dict[str, dict[str, str]] = {}
    row_hashes: dict[str, str] = {}
    parsed_lists: dict[str, dict[str, list[str]]] = {}
    flags_by_id: dict[str, tuple[str, ...]] = {}
    json_nonempty_counts: Counter[str] = Counter()
    for index, raw_row in enumerate(rows, 1):
        require(None not in raw_row, f"{label} dòng {index}: thừa field ngoài header")
        row = {field: raw_row.get(field) or "" for field in fieldnames}
        rows[index - 1] = row
        disease_id = row["disease_id"]
        expected_id = f"TAYY-{index:05d}"
        require(
            disease_id == expected_id,
            f"{label} dòng {index}: chờ {expected_id}, gặp {disease_id!r}",
        )
        require(disease_id not in by_id, f"{label}: disease_id trùng {disease_id}")
        require(row["tên_bệnh"].strip(), f"{disease_id}: tên_bệnh rỗng")

        parsed: dict[str, list[str]] = {}
        for column in JSON_ARRAY_COLUMNS:
            raw_value = row[column]
            try:
                value = [] if raw_value == "" else json.loads(raw_value)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"{disease_id}: {column} không parse được JSON array: {exc}"
                ) from exc
            require(
                isinstance(value, list)
                and all(isinstance(item, str) for item in value),
                f"{disease_id}: {column} phải là JSON array chỉ chứa string",
            )
            if raw_value:
                require(value, f"{disease_id}: {column} dùng [] thay vì canonical blank")
                json_nonempty_counts[column] += 1
            parsed[column] = value
        parsed_lists[disease_id] = parsed

        flags = tuple(part.strip() for part in row["cờ_chất_lượng"].split(";") if part.strip())
        require(
            flags == tuple(sorted(set(flags))),
            f"{disease_id}: cờ_chất_lượng phải sorted/unique",
        )
        flags_by_id[disease_id] = flags

        digest = canonical_row_sha256(row, fieldnames)
        require(
            digest == production_row_sha256(row, fieldnames),
            f"{disease_id}: fingerprint độc lập lệch implementation production",
        )
        by_id[disease_id] = row
        row_hashes[disease_id] = digest

    require(
        len(set(row_hashes.values())) == EXPECTED_CLEAN_ROWS,
        "Canonical row hashes không unique trên 8.724 dòng",
    )
    require(
        dict(json_nonempty_counts) == EXPECTED_JSON_NONEMPTY,
        "Số ô JSON-list nonblank lệch golden corpus: "
        f"chờ {EXPECTED_JSON_NONEMPTY}, gặp {dict(json_nonempty_counts)}",
    )
    clean_data = CleanData(
        fieldnames=fieldnames,
        rows=rows,
        by_id=by_id,
        row_hashes=row_hashes,
        corpus_source_sha256=logical_corpus_sha256(rows, row_hashes),
    )
    return CleanCorpus(
        data=clean_data,
        parsed_lists=parsed_lists,
        flags=flags_by_id,
        json_nonempty_counts=dict(json_nonempty_counts),
        raw_sha256=sha256_file(path),
    )


def validate_duplicate_name_contract(corpus: CleanCorpus) -> set[str]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    flagged_ids: set[str] = set()
    for row in corpus.data.rows:
        disease_id = row["disease_id"]
        groups[row["tên_bệnh"].casefold()].append(disease_id)
        if DUPLICATE_FLAG in corpus.flags[disease_id]:
            flagged_ids.add(disease_id)

    duplicate_groups = {name: ids for name, ids in groups.items() if len(ids) > 1}
    duplicate_ids = {disease_id for ids in duplicate_groups.values() for disease_id in ids}
    require(
        len(flagged_ids) == EXPECTED_DUPLICATE_ROWS,
        f"Cờ trùng tên: chờ {EXPECTED_DUPLICATE_ROWS} dòng, gặp {len(flagged_ids)}",
    )
    require(
        len(duplicate_groups) == EXPECTED_DUPLICATE_GROUPS,
        f"Nhóm tên casefold: chờ {EXPECTED_DUPLICATE_GROUPS}, gặp {len(duplicate_groups)}",
    )
    require(
        flagged_ids == duplicate_ids,
        "Cờ trung_ten_khac_noi_dung không phủ đúng toàn bộ nhóm tên casefold trùng",
    )
    size_distribution = Counter(len(ids) for ids in duplicate_groups.values())
    require(
        {str(size): count for size, count in sorted(size_distribution.items())}
        == EXPECTED_DUPLICATE_GROUP_SIZE_DISTRIBUTION,
        f"Phân bố kích thước nhóm trùng lệch: {dict(size_distribution)}",
    )
    return flagged_ids


def load_catalog_contract(path: Path, meta_path: Path) -> CatalogData:
    require(path.is_file(), f"Thiếu catalog: {path}")
    require(meta_path.is_file(), f"Thiếu catalog metadata: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    require(isinstance(meta, dict), "Catalog metadata phải là JSON object")

    digest = sha256_file(path)
    require(
        digest == EXPECTED_CATALOG_SHA256,
        f"Catalog final hash lệch: chờ {EXPECTED_CATALOG_SHA256}, gặp {digest}",
    )
    require(
        meta.get("catalog_sha256") == digest,
        "catalog_sha256 trong metadata không khớp file JSONL",
    )

    records: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip(), f"Catalog có dòng rỗng tại {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"Catalog JSON lỗi dòng {line_number}: {exc}") from exc
            require(isinstance(record, dict), f"Catalog dòng {line_number} không phải object")
            code = record.get("code")
            require(
                isinstance(code, str) and CODE_RE.fullmatch(code),
                f"Catalog dòng {line_number}: code không canonical {code!r}",
            )
            require(code not in records, f"Catalog trùng code {code}")
            require(
                record.get("code_nodot") == code.replace(".", ""),
                f"Catalog {code}: code_nodot lệch",
            )
            require(
                isinstance(record.get("title_vi"), str) and record["title_vi"].strip(),
                f"Catalog {code}: title_vi rỗng",
            )
            require(
                type(record.get("auto_single_code_eligible")) is bool,
                f"Catalog {code}: auto_single_code_eligible không phải bool",
            )
            records[code] = record
            alias = code.replace(".", "")
            require(alias not in aliases, f"Catalog alias trùng {alias}")
            aliases[alias] = code

    require(
        len(records) == EXPECTED_CATALOG_ROWS,
        f"Catalog chờ {EXPECTED_CATALOG_ROWS} dòng, gặp {len(records)}",
    )
    require(meta.get("rows") == len(records), "Metadata catalog.rows lệch JSONL")
    require(meta.get("unique_codes") == len(records), "Metadata unique_codes lệch JSONL")
    return CatalogData(records=records, aliases=aliases, sha256=digest, meta=meta)


def validate_status_code_pairs(corpus: CleanCorpus, catalog: CatalogData, label: str) -> Counter:
    counts: Counter[str] = Counter()
    for row in corpus.data.rows:
        disease_id = row["disease_id"]
        code = row["icd10_code"]
        status = row["trạng_thái_kiểm_duyệt"]
        require(status in VALID_STATUSES, f"{label} {disease_id}: status lạ {status!r}")
        if code:
            require(code in catalog.records, f"{label} {disease_id}: code ngoài catalog {code!r}")
        if status == STATUS_REVIEWED:
            require(code, f"{label} {disease_id}: da_kiem_duyet nhưng code rỗng")
        if status == STATUS_UNMAPPABLE:
            require(not code, f"{label} {disease_id}: khong_anh_xa nhưng có code {code}")
        counts[status] += 1
    return counts


def require_exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{where} phải là JSON object")
    actual = set(value)
    require(
        actual == expected,
        f"{where}: schema lệch; thiếu={sorted(expected - actual)}, lạ={sorted(actual - expected)}",
    )
    return value


def _validate_candidate(
    candidate: Any,
    expected_rank: int,
    catalog: CatalogData,
    exact: Mapping[str, Any],
    where: str,
) -> str:
    item = require_exact_keys(candidate, CANDIDATE_FIELDS, where)
    require(item["rank"] == expected_rank, f"{where}: rank phải là {expected_rank}")
    code = item.get("code")
    require(code in catalog.records, f"{where}: candidate code ngoài catalog {code!r}")
    record = catalog.records[code]
    require(item.get("title_vi") == record.get("title_vi"), f"{where}: title_vi lệch catalog")
    require(item.get("title_en") == record.get("title_en", ""), f"{where}: title_en lệch catalog")
    require(item.get("category_vi") == record.get("category_vi", ""), f"{where}: category_vi lệch catalog")
    require(item.get("guidance_vi") == record.get("guidance_vi", ""), f"{where}: guidance_vi lệch catalog")

    components = require_exact_keys(
        item.get("components"), CANDIDATE_COMPONENT_FIELDS, f"{where}.components"
    )
    bm25 = components.get("bm25")
    require(
        isinstance(bm25, (int, float)) and not isinstance(bm25, bool) and bm25 >= 0,
        f"{where}: components.bm25 không phải số không âm",
    )
    bm25_by_field = components.get("bm25_by_tayy_field")
    require(isinstance(bm25_by_field, dict), f"{where}: bm25_by_tayy_field không phải object")
    require(
        set(bm25_by_field).issubset(BM25_TAYY_FIELDS),
        f"{where}: bm25_by_tayy_field có field lạ {sorted(set(bm25_by_field) - BM25_TAYY_FIELDS)}",
    )
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
            for value in bm25_by_field.values()
        ),
        f"{where}: bm25_by_tayy_field phải chỉ chứa score dương",
    )
    name_exact = components.get("name_exact")
    lead_exact = components.get("lead_exact")
    require(type(name_exact) is bool, f"{where}: name_exact không phải bool")
    require(type(lead_exact) is bool, f"{where}: lead_exact không phải bool")
    require(name_exact is (code in exact["name_codes"]), f"{where}: name_exact lệch exact.name_codes")
    require(lead_exact is (code in exact["lead_codes"]), f"{where}: lead_exact lệch exact.lead_codes")
    name_variants = components.get("name_exact_variant_types")
    lead_variants = components.get("lead_exact_variant_types")
    for label, enabled, variants in (
        ("name", name_exact, name_variants),
        ("lead", lead_exact, lead_variants),
    ):
        require(
            isinstance(variants, list)
            and all(value in ("primary", "bracket_stripped") for value in variants)
            and variants == sorted(set(variants)),
            f"{where}: {label}_exact_variant_types sai schema/order",
        )
        require(bool(variants) is enabled, f"{where}: {label}_exact không khớp variant types")
    exact_bonus = EXACT_RANK_BONUS * (int(name_exact) + int(lead_exact))
    require(
        components.get("exact_rank_bonus") == exact_bonus,
        f"{where}: exact_rank_bonus lệch exact booleans",
    )

    retrieved_by = item.get("retrieved_by")
    require(
        isinstance(retrieved_by, list) and all(isinstance(value, str) for value in retrieved_by),
        f"{where}: retrieved_by không phải list[str]",
    )
    expected_retrieved = [f"name_exact:{kind}" for kind in name_variants]
    expected_retrieved.extend(f"lead_exact:{kind}" for kind in lead_variants)
    if bm25 > 0:
        expected_retrieved.append("sparse_bm25")
    require(retrieved_by == expected_retrieved, f"{where}: retrieved_by lệch components")
    require(
        isinstance(item.get("score"), (int, float)) and not isinstance(item.get("score"), bool),
        f"{where}: score không phải số",
    )
    require(
        item["score"] == round(bm25 + exact_bonus, 6),
        f"{where}: score != round(bm25 + exact_bonus, 6)",
    )

    eligibility = require_exact_keys(
        item.get("eligibility"), CANDIDATE_ELIGIBILITY_FIELDS, f"{where}.eligibility"
    )
    for field in (
        "auto_single_code_eligible",
        "codable",
        "morbidity_eligible",
        "primary_eligible",
        "primary_eligibility",
        "dagger",
        "asterisk",
        "dual_coding_marker",
        "restrictions",
    ):
        require(
            eligibility.get(field) == record.get(field),
            f"{where}: eligibility.{field} lệch catalog",
        )
    require(
        eligibility.get("pair_resolution_required") == bool(record.get("dual_coding_marker")),
        f"{where}: pair_resolution_required lệch dual marker",
    )
    expected_refs = record.get("who_references", []) if record.get("dual_coding_marker") else []
    require(
        eligibility.get("who_references") == expected_refs,
        f"{where}: who_references lệch catalog",
    )
    source_audit = require_exact_keys(
        item.get("source_audit"), CANDIDATE_SOURCE_AUDIT_FIELDS, f"{where}.source_audit"
    )
    for field in CANDIDATE_SOURCE_AUDIT_FIELDS:
        require(
            source_audit.get(field) == record.get(field),
            f"{where}: source_audit.{field} lệch catalog",
        )
    return code


@dataclass
class ProposalCorpus:
    by_id: dict[str, dict[str, Any]]
    source_states: dict[str, dict[str, str]]
    input_sha256: str
    provisional_count: int


def validate_proposals(
    path: Path,
    clean: CleanCorpus,
    catalog: CatalogData,
    duplicate_ids: set[str],
) -> ProposalCorpus:
    require(path.is_file(), f"Thiếu proposal JSONL: {path}")
    proposals: dict[str, dict[str, Any]] = {}
    source_states: dict[str, dict[str, str]] = {}
    input_hashes: set[str] = set()
    provisional_count = 0
    duplicate_provisional: list[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for record_number, line in enumerate(handle, 1):
            require(line.strip(), f"Proposal có dòng rỗng tại {record_number}")
            try:
                proposal = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"Proposal JSON lỗi dòng {record_number}: {exc}") from exc
            proposal = dict(require_exact_keys(proposal, PROPOSAL_FIELDS, f"proposal #{record_number}"))
            require(
                record_number <= EXPECTED_CLEAN_ROWS,
                f"Proposal dư dòng ngoài {EXPECTED_CLEAN_ROWS}",
            )
            source_row = clean.data.rows[record_number - 1]
            disease_id = source_row["disease_id"]
            where = f"proposal {disease_id}"
            require(proposal["schema_version"] == EXPECTED_PROPOSAL_SCHEMA, f"{where}: schema_version lệch")
            require(proposal["disease_id"] == disease_id, f"{where}: sai thứ tự/ID")
            require(proposal["source_csv_line"] == record_number + 1, f"{where}: source_csv_line lệch")
            require(disease_id not in proposals, f"Proposal disease_id trùng {disease_id}")
            expected_hash = clean.data.row_hashes[disease_id]
            require(proposal["row_sha256"] == expected_hash, f"{where}: row_sha256 stale")
            require(proposal["row_fingerprint"] == expected_hash, f"{where}: row_fingerprint stale")
            require(proposal["name"] == source_row["tên_bệnh"].strip(), f"{where}: name lệch clean")

            source_state = require_exact_keys(
                proposal["source_state"], set(MUTABLE_COLUMNS), f"{where}.source_state"
            )
            source_states[disease_id] = {
                "icd10_code": str(source_state["icd10_code"] or ""),
                "trạng_thái_kiểm_duyệt": str(source_state["trạng_thái_kiểm_duyệt"] or ""),
            }

            evidence = require_exact_keys(proposal["evidence"], EVIDENCE_FIELDS, f"{where}.evidence")
            parsed = clean.parsed_lists[disease_id]
            expected_evidence = {
                "description": source_row["mô_tả_bệnh"],
                "types": parsed["loại_bệnh"],
                "symptoms": parsed["triệu_chứng"],
                "tests": parsed["kiểm_tra"],
                "comorbidities": parsed["bệnh_đi_kèm"],
                "department": source_row["khoa_điều_trị"],
                "cause": source_row["nguyên_nhân"],
                "method": source_row["phương_pháp"],
            }
            require(dict(evidence) == expected_evidence, f"{where}: full evidence lệch clean")

            components = proposal["components"]
            require(isinstance(components, dict), f"{where}: components không phải object")
            exact = components.get("exact")
            quality = components.get("quality")
            risks = components.get("risk_gates")
            require(isinstance(exact, dict), f"{where}: components.exact thiếu/lạ")
            require(isinstance(quality, dict), f"{where}: components.quality thiếu/lạ")
            require(isinstance(risks, dict), f"{where}: components.risk_gates thiếu/lạ")
            require(
                quality.get("flags") == list(clean.flags[disease_id]),
                f"{where}: quality.flags lệch clean",
            )
            require(
                quality.get("unflagged") is (not clean.flags[disease_id]),
                f"{where}: quality.unflagged lệch flags",
            )
            require(
                quality.get("duplicate_name_unresolved") is (disease_id in duplicate_ids),
                f"{where}: duplicate_name_unresolved lệch cờ clean",
            )

            candidates = proposal["candidates"]
            require(isinstance(candidates, list), f"{where}: candidates không phải array")
            require(len(candidates) == TOP_K, f"{where}: chờ đúng top-{TOP_K}, gặp {len(candidates)}")
            candidate_codes = [
                _validate_candidate(
                    item,
                    rank,
                    catalog,
                    exact,
                    f"{where}.candidates[{rank}]",
                )
                for rank, item in enumerate(candidates, 1)
            ]
            require(len(candidate_codes) == len(set(candidate_codes)), f"{where}: candidate code trùng")

            for key in ("name_codes", "lead_codes"):
                values = exact.get(key)
                require(
                    isinstance(values, list)
                    and values == sorted(set(values))
                    and all(code in catalog.records for code in values),
                    f"{where}: exact.{key} chứa code ngoài catalog hoặc sai type",
                )
            name_codes = exact["name_codes"]
            lead_codes = exact["lead_codes"]
            expected_conflict = bool(
                name_codes and lead_codes and set(name_codes).isdisjoint(lead_codes)
            )
            require(exact.get("conflict") is expected_conflict, f"{where}: exact.conflict lệch")
            same_unique_code = exact.get("same_unique_code")
            expected_same = (
                name_codes[0]
                if len(name_codes) == 1 and name_codes == lead_codes
                else None
            )
            require(same_unique_code == expected_same, f"{where}: same_unique_code lệch exact sets")

            provisional = proposal["provisional_code"]
            tier = proposal["proposal_tier"]
            require(tier in (None, "A"), f"{where}: proposal_tier lạ {tier!r}")
            require(provisional is None or provisional in catalog.records, f"{where}: provisional ngoài catalog")
            eligible = bool(
                same_unique_code
                and catalog.records[same_unique_code].get("auto_single_code_eligible") is True
            )
            qualifies_for_tier_a = bool(
                same_unique_code
                and eligible
                and not exact["conflict"]
                and quality.get("unflagged") is True
                and quality.get("duplicate_name_unresolved") is False
                and risks.get("ui") is False
                and risks.get("tcm") is False
                and risks.get("low_info") is False
            )
            expected_tier = "A" if qualifies_for_tier_a else None
            expected_provisional = same_unique_code if qualifies_for_tier_a else None
            require(
                tier == expected_tier,
                f"{where}: tier={tier!r}, hard gates yêu cầu {expected_tier!r}",
            )
            require(
                provisional == expected_provisional,
                f"{where}: provisional={provisional!r}, hard gates yêu cầu "
                f"{expected_provisional!r}",
            )
            require(proposal["requires_human_review"] is True, f"{where}: phải requires_human_review")
            reasons = proposal["decision_reasons"]
            require(
                isinstance(reasons, list)
                and all(isinstance(reason, str) for reason in reasons)
                and len(reasons) == len(set(reasons)),
                f"{where}: decision_reasons không phải list[str] unique",
            )
            if provisional is not None:
                provisional_count += 1
                require(provisional in candidate_codes, f"{where}: provisional ngoài top-{TOP_K}")
                require(
                    "tier_a_exact_consensus_requires_human_review" in reasons,
                    f"{where}: Tier A thiếu audit reason",
                )
                if disease_id in duplicate_ids:
                    duplicate_provisional.append(disease_id)
            else:
                require(tier is None, f"{where}: tier non-null nhưng provisional null")

            provenance = proposal["provenance"]
            require(isinstance(provenance, dict), f"{where}: provenance không phải object")
            require(
                provenance.get("catalog_sha256") == catalog.sha256,
                f"{where}: catalog_sha256 stale",
            )
            require(
                provenance.get("row_fingerprint_algorithm") == EXPECTED_ROW_FINGERPRINT_ALGORITHM,
                f"{where}: row_fingerprint_algorithm lệch contract",
            )
            require(
                set(provenance.get("row_fingerprint_excluded_columns") or ())
                == set(MUTABLE_COLUMNS),
                f"{where}: fingerprint không loại đúng hai cột mutable",
            )
            input_sha = str(provenance.get("input_sha256") or "").lower()
            require(SHA256_RE.fullmatch(input_sha), f"{where}: input_sha256 không hợp lệ")
            input_hashes.add(input_sha)
            proposals[disease_id] = proposal

    require(
        len(proposals) == EXPECTED_CLEAN_ROWS,
        f"Proposal chờ {EXPECTED_CLEAN_ROWS} dòng, gặp {len(proposals)}",
    )
    require(not duplicate_provisional, f"Dòng trùng tên có provisional: {duplicate_provisional[:10]}")
    require(len(input_hashes) == 1, f"Proposal không cùng một input_sha256: {sorted(input_hashes)}")
    require(provisional_count > 0, "Không có provisional nào; simulation recovery không còn ý nghĩa")
    return ProposalCorpus(
        by_id=proposals,
        source_states=source_states,
        input_sha256=next(iter(input_hashes)),
        provisional_count=provisional_count,
    )


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def normalize_exact(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).replace("_", " ")
    return " ".join(value.split())


def csv_safe(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def proposal_priority(proposal: Mapping[str, Any]) -> str:
    components = proposal["components"]
    if components["quality"]["duplicate_name_unresolved"]:
        return "P0_duplicate_name"
    if components["exact"]["conflict"]:
        return "P0_exact_conflict"
    risks = components["risk_gates"]
    if risks["ui"] or risks["tcm"]:
        return "P1_non_disease_or_tcm"
    if risks["low_info"]:
        return "P1_low_information"
    if proposal["proposal_tier"] == "A":
        return "P2_validate_tier_a"
    return "P1_manual_mapping"


def expected_review_queue_row(proposal: Mapping[str, Any]) -> dict[str, str]:
    components = proposal["components"]
    exact = components["exact"]
    risks = components["risk_gates"]
    evidence = proposal["evidence"]
    compact_candidates = [
        {
            "rank": item["rank"],
            "code": item["code"],
            "title_vi": item["title_vi"],
            "title_en": item["title_en"],
            "score": item["score"],
            "retrieved_by": item["retrieved_by"],
            "auto_single_code_eligible": item["eligibility"][
                "auto_single_code_eligible"
            ],
            "primary_eligibility": item["eligibility"]["primary_eligibility"],
            "restrictions": item["eligibility"]["restrictions"],
        }
        for item in proposal["candidates"]
    ]
    values: dict[str, Any] = {
        "priority": proposal_priority(proposal),
        "disease_id": proposal["disease_id"],
        "tên_bệnh": proposal["name"],
        "mô_tả_bệnh": evidence["description"],
        "lead": components["lead"]["raw"],
        "loại_bệnh": stable_json(evidence["types"]),
        "triệu_chứng": stable_json(evidence["symptoms"]),
        "kiểm_tra": stable_json(evidence["tests"]),
        "khoa_điều_trị": evidence["department"],
        "nguyên_nhân": evidence["cause"],
        "bệnh_đi_kèm": stable_json(evidence["comorbidities"]),
        "phương_pháp": evidence["method"],
        "proposal_tier": proposal["proposal_tier"] or "",
        "provisional_code": proposal["provisional_code"] or "",
        "decision_reasons": ";".join(proposal["decision_reasons"]),
        "exact_name_codes": stable_json(exact["name_codes"]),
        "exact_lead_codes": stable_json(exact["lead_codes"]),
        "top_candidates": stable_json(compact_candidates),
        "cờ_chất_lượng": ";".join(components["quality"]["flags"]),
        "ui_flag": str(risks["ui"]).lower(),
        "tcm_flag": str(risks["tcm"]).lower(),
        "low_info_flag": str(risks["low_info"]).lower(),
        "exact_conflict": str(exact["conflict"]).lower(),
        "current_icd10_code": proposal["source_state"]["icd10_code"],
        "current_review_status": proposal["source_state"][
            "trạng_thái_kiểm_duyệt"
        ],
        "decision": "",
        "icd10_code": "",
        "reviewer": "",
        "reviewed_at": "",
        "review_note": "",
        "row_sha256": proposal["row_sha256"],
        "row_fingerprint": proposal["row_fingerprint"],
    }
    return {column: csv_safe(values[column]) for column in REVIEW_COLUMNS}


def validate_review_queue(path: Path, proposals: ProposalCorpus) -> None:
    require(path.is_file(), f"Thiếu mapper review queue: {path}")
    expected_order = sorted(
        proposals.by_id.values(),
        key=lambda proposal: (
            proposal_priority(proposal),
            normalize_exact(str(proposal.get("name") or "")),
            str(proposal["disease_id"]),
        ),
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            list(reader.fieldnames or []) == REVIEW_COLUMNS,
            f"Review queue schema/order lệch: {reader.fieldnames!r}",
        )
        seen_ids: set[str] = set()
        rows_read = 0
        for rows_read, actual in enumerate(reader, 1):
            require(
                rows_read <= EXPECTED_CLEAN_ROWS,
                f"Review queue dư dòng ngoài {EXPECTED_CLEAN_ROWS}",
            )
            require(None not in actual, f"Review queue dòng {rows_read}: thừa field")
            proposal = expected_order[rows_read - 1]
            disease_id = str(proposal["disease_id"])
            require(disease_id not in seen_ids, f"Review queue disease_id trùng {disease_id}")
            seen_ids.add(disease_id)
            expected = expected_review_queue_row(proposal)
            require(
                actual == expected,
                f"Review queue dòng {rows_read}/{disease_id}: payload/full evidence lệch proposal",
            )
            require(
                all(actual[column] == "" for column in REVIEW_EDIT_COLUMNS),
                f"Review queue {disease_id}: review fields phải trống trước khi import",
            )
    require(
        rows_read == EXPECTED_CLEAN_ROWS,
        f"Review queue chờ {EXPECTED_CLEAN_ROWS} dòng, gặp {rows_read}",
    )
    require(
        seen_ids == set(proposals.by_id),
        "Review queue không one-to-one với proposal sidecar",
    )


def duplicate_group_size_distribution(clean: CleanCorpus) -> dict[str, int]:
    groups: Counter[str] = Counter(
        row["tên_bệnh"].casefold()
        for row in clean.data.rows
        if DUPLICATE_FLAG in clean.flags[row["disease_id"]]
    )
    distribution = Counter(groups.values())
    return {str(size): count for size, count in sorted(distribution.items())}


def proposal_report_counts(proposals: ProposalCorpus) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    candidate_distribution: Counter[str] = Counter()
    for proposal in proposals.by_id.values():
        exact = proposal["components"]["exact"]
        quality = proposal["components"]["quality"]
        risks = proposal["components"]["risk_gates"]
        stats["name_exact_any"] += bool(exact["name_codes"])
        stats["name_exact_unique"] += len(exact["name_codes"]) == 1
        stats["lead_exact_any"] += bool(exact["lead_codes"])
        stats["lead_exact_unique"] += len(exact["lead_codes"]) == 1
        stats["name_lead_same_unique_code"] += bool(exact["same_unique_code"])
        stats["exact_conflict"] += exact["conflict"]
        stats["quality_flagged"] += not quality["unflagged"]
        stats["duplicate_name_unresolved"] += quality["duplicate_name_unresolved"]
        stats["ui_gate"] += risks["ui"]
        stats["tcm_gate"] += risks["tcm"]
        stats["low_info_gate"] += risks["low_info"]
        stats["tier_a_provisional"] += proposal["proposal_tier"] == "A"
        stats["without_candidates"] += not proposal["candidates"]
        candidate_distribution[str(len(proposal["candidates"]))] += 1
        reasons.update(proposal["decision_reasons"])
    return {
        "rows_in_scope": EXPECTED_CLEAN_ROWS,
        "proposals": EXPECTED_CLEAN_ROWS,
        "review_queue_rows": EXPECTED_CLEAN_ROWS,
        **dict(sorted(stats.items())),
        "candidate_count_distribution": dict(
            sorted(candidate_distribution.items(), key=lambda item: int(item[0]))
        ),
        "decision_reasons": dict(sorted(reasons.items())),
    }


def validate_mapping_report(
    path: Path,
    clean: CleanCorpus,
    catalog: CatalogData,
    proposals: ProposalCorpus,
) -> None:
    require(path.is_file(), f"Thiếu mapping report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report = require_exact_keys(
        report,
        {
            "schema_version",
            "mapper_version",
            "mode",
            "scope",
            "inputs",
            "outputs",
            "counts",
            "invariants",
            "method",
        },
        "mapping report",
    )
    require(report["schema_version"] == "tayy-icd10-report-v1", "Mapping report schema lệch")
    require(report["mode"] == "write" and report["scope"] == "all", "Mapping report không phải full write")
    mapper_versions = {
        str(proposal["provenance"].get("mapper_version") or "")
        for proposal in proposals.by_id.values()
    }
    require(mapper_versions == {report["mapper_version"]}, "Mapping report mapper_version lệch proposals")

    inputs = require_exact_keys(report["inputs"], {"tayy_csv", "catalog"}, "mapping report.inputs")
    tayy_input = require_exact_keys(
        inputs["tayy_csv"],
        {
            "path",
            "sha256",
            "rows",
            "columns",
            "duplicate_name_flagged_rows",
            "duplicate_name_flagged_groups",
            "duplicate_name_group_size_distribution",
        },
        "mapping report.inputs.tayy_csv",
    )
    require(tayy_input["path"] == "data/TayY_clean.csv", "Mapping report TayY path lệch")
    require(
        tayy_input["sha256"] == proposals.input_sha256,
        "Mapping report phải giữ hash pre-materialization từ proposal provenance",
    )
    require(tayy_input["rows"] == EXPECTED_CLEAN_ROWS, "Mapping report TayY rows lệch")
    require(tayy_input["columns"] == EXPECTED_CLEAN_COLUMNS, "Mapping report TayY columns lệch")
    require(tayy_input["duplicate_name_flagged_rows"] == EXPECTED_DUPLICATE_ROWS, "Mapping report duplicate rows lệch")
    require(tayy_input["duplicate_name_flagged_groups"] == EXPECTED_DUPLICATE_GROUPS, "Mapping report duplicate groups lệch")
    require(
        tayy_input["duplicate_name_group_size_distribution"]
        == duplicate_group_size_distribution(clean),
        "Mapping report duplicate group-size distribution lệch clean",
    )

    catalog_input = require_exact_keys(
        inputs["catalog"],
        {"path", "meta_path", "system", "who_release", "sha256", "source_sha256", "rows"},
        "mapping report.inputs.catalog",
    )
    require(catalog_input["path"] == "data/icd10/vn_icd10_tt06_2026.jsonl", "Mapping report catalog path lệch")
    require(catalog_input["meta_path"] == "data/icd10/vn_icd10_tt06_2026.meta.json", "Mapping report meta path lệch")
    require(catalog_input["sha256"] == catalog.sha256, "Mapping report catalog hash lệch")
    require(catalog_input["source_sha256"] == catalog.meta.get("source_sha256"), "Mapping report source hash lệch")
    require(catalog_input["rows"] == len(catalog.records), "Mapping report catalog rows lệch")
    require(catalog_input["system"] == catalog.meta.get("system"), "Mapping report catalog system lệch")
    require(catalog_input["who_release"] == catalog.meta.get("who_release"), "Mapping report WHO release lệch")

    outputs = require_exact_keys(
        report["outputs"],
        {"written", "proposals", "review_queue", "report"},
        "mapping report.outputs",
    )
    require(outputs == {
        "written": True,
        "proposals": "data/icd10/tayy_icd10_proposals.jsonl",
        "review_queue": "data/icd10/tayy_icd10_review_queue.csv",
        "report": "data/tayy_icd10_report.json",
    }, "Mapping report output contract lệch")
    require(report["counts"] == proposal_report_counts(proposals), "Mapping report counts lệch proposals")

    expected_invariants = {
        "catalog_sha256_matches_meta",
        "catalog_codes_unique_and_structurally_valid",
        "catalog_eligibility_fields_recomputed",
        "tayy_ids_unique_contiguous_and_ordered",
        "tayy_json_array_columns_valid",
        "row_sha256_excludes_exactly_code_and_review_status",
        "one_proposal_per_scoped_row_in_input_order",
        "candidate_codes_unique_catalog_members_top5",
        "bm25_is_candidate_only",
        "provisional_is_tier_a_exact_consensus_only",
        "duplicate_name_flag_never_provisional",
        "all_proposals_require_human_review",
    }
    require(
        set(report["invariants"]) == expected_invariants
        and all(report["invariants"].values()),
        "Mapping report invariants thiếu/lạ/false",
    )
    method = report["method"]
    require(isinstance(method, dict), "Mapping report method không phải object")
    require(method.get("row_fingerprint") == EXPECTED_ROW_FINGERPRINT_ALGORITHM, "Mapping report fingerprint contract lệch")
    require("deterministic top-5" in str(method.get("candidate_retrieval")), "Mapping report thiếu top-5 method")


def expected_projection(
    disease_id: str,
    proposals: Mapping[str, Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, str]],
    llm_proposals: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    proposal = proposals[disease_id]
    review = reviews.get(disease_id)
    if review:
        decision = review["decision"]
        if decision == "approved":
            return review["icd10_code"], STATUS_REVIEWED
        if decision in UNMAPPABLE_DECISIONS:
            return "", STATUS_UNMAPPABLE
        if decision in ("rejected", "needs_more_information"):
            return "", STATUS_UNREVIEWED
    provisional = str(proposal.get("provisional_code") or "")
    if provisional:
        return provisional, STATUS_UNREVIEWED
    # Bậc máy đề xuất LLM (dưới Tier A, trên none) — trạng thái vẫn chưa kiểm duyệt.
    llm_code = str((llm_proposals.get(disease_id) or {}).get("_code") or "")
    return llm_code, STATUS_UNREVIEWED


def assert_immutable_rows(
    before: CleanData,
    after_rows: Sequence[Mapping[str, str]],
    label: str,
) -> None:
    require(len(after_rows) == len(before.rows), f"{label}: số dòng thay đổi")
    immutable = [field for field in before.fieldnames if field not in MUTABLE_COLUMNS]
    for index, (old, new) in enumerate(zip(before.rows, after_rows), 1):
        disease_id = old["disease_id"]
        require(new.get("disease_id") == disease_id, f"{label} dòng {index}: ID/order đổi")
        for field in immutable:
            require(
                (new.get(field) or "") == (old.get(field) or ""),
                f"{label} {disease_id}: cột immutable {field!r} bị đổi",
            )
        require(
            canonical_row_sha256(new, before.fieldnames) == before.row_hashes[disease_id],
            f"{label} {disease_id}: immutable fingerprint đổi",
        )


def validate_build_plan(
    current: CleanCorpus,
    catalog: CatalogData,
    proposals: ProposalCorpus,
    reviews: dict[str, dict[str, str]],
    llm_proposals: dict[str, dict[str, Any]],
) -> tuple[dict[str, tuple[str, str]], str]:
    desired = {
        disease_id: expected_projection(disease_id, proposals.by_id, reviews, llm_proposals)
        for disease_id in current.data.by_id
    }
    planned_rows, changes, _stats = build_plan(
        current.data,
        catalog,
        proposals.by_id,
        reviews,
        llm_proposals,
        reconcile_orphans=False,
    )
    assert_immutable_rows(current.data, planned_rows, "build_plan")
    expected_changed_ids: set[str] = set()
    for row, planned in zip(current.data.rows, planned_rows):
        disease_id = row["disease_id"]
        desired_code, desired_status = desired[disease_id]
        require(
            (planned["icd10_code"], planned["trạng_thái_kiểm_duyệt"])
            == (desired_code, desired_status),
            f"build_plan {disease_id}: projection sai",
        )
        if (row["icd10_code"], row["trạng_thái_kiểm_duyệt"]) != (
            desired_code,
            desired_status,
        ):
            expected_changed_ids.add(disease_id)
    require(
        {item["disease_id"] for item in changes} == expected_changed_ids,
        "build_plan changes không đúng tập dòng lệch desired projection",
    )

    current_projection = [
        (row["icd10_code"], row["trạng_thái_kiểm_duyệt"])
        for row in current.data.rows
    ]
    desired_projection = [desired[row["disease_id"]] for row in current.data.rows]
    reset_projection = [("", STATUS_UNREVIEWED)] * EXPECTED_CLEAN_ROWS
    if current_projection == reset_projection:
        materialized_state = "cleaner_reset"
    else:
        require(
            current_projection == desired_projection,
            "CSV hiện tại không phải cleaner-reset hoàn toàn cũng không khớp đầy đủ sidecars",
        )
        materialized_state = "materialized"
    return desired, materialized_state


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def rows_with_projection(
    rows: Sequence[Mapping[str, str]],
    projection: Mapping[str, tuple[str, str]],
) -> Iterable[dict[str, str]]:
    for row in rows:
        copied = dict(row)
        code, status = projection[row["disease_id"]]
        copied["icd10_code"] = code
        copied["trạng_thái_kiểm_duyệt"] = status
        yield copied


def assert_projection(
    data: CleanData,
    desired: Mapping[str, tuple[str, str]],
    label: str,
) -> None:
    for row in data.rows:
        disease_id = row["disease_id"]
        actual = (row["icd10_code"], row["trạng_thái_kiểm_duyệt"])
        require(actual == desired[disease_id], f"{label} {disease_id}: {actual} != {desired[disease_id]}")


def simulate_materialization(
    clean: CleanCorpus,
    catalog: CatalogData,
    proposals: ProposalCorpus,
    reviews: dict[str, dict[str, str]],
    llm_proposals: dict[str, dict[str, Any]],
    desired: dict[str, tuple[str, str]],
) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="tayy_icd10_regression_") as temp_name:
        temp_dir = Path(temp_name)
        temp_clean = temp_dir / "TayY_clean.csv"
        backup_dir = temp_dir / "backups"

        source_projection = {
            disease_id: (
                state["icd10_code"],
                state["trạng_thái_kiểm_duyệt"],
            )
            for disease_id, state in proposals.source_states.items()
        }
        write_csv(
            temp_clean,
            clean.data.fieldnames,
            rows_with_projection(clean.data.rows, source_projection),
        )
        require(
            sha256_file(temp_clean) == proposals.input_sha256,
            "Không tái tạo được provenance.input_sha256 từ full corpus + source_state",
        )

        reset_projection = {
            disease_id: ("", STATUS_UNREVIEWED) for disease_id in clean.data.by_id
        }
        write_csv(
            temp_clean,
            clean.data.fieldnames,
            rows_with_projection(clean.data.rows, reset_projection),
        )
        reset_data = load_clean(temp_clean)
        require(
            reset_data.row_hashes == clean.data.row_hashes,
            "Reset đúng hai cột nhưng canonical row hashes đã đổi",
        )

        def apply_once() -> tuple[int, str, int]:
            current = load_clean(temp_clean)
            planned, changes, _stats = build_plan(
                current,
                catalog,
                proposals.by_id,
                reviews,
                llm_proposals,
                reconcile_orphans=False,
            )
            assert_immutable_rows(current, planned, "temp apply")
            if changes:
                apply_module._atomic_apply(temp_clean, current, planned, backup_dir)
            backups = len(list(backup_dir.glob("TayY_clean.bak.*.csv")))
            return len(changes), sha256_file(temp_clean), backups

        first_changes, first_hash, first_backups = apply_once()
        require(first_changes > 0, "Lần apply đầu từ reset không có thay đổi")
        require(first_backups == 1, f"Lần apply đầu phải có 1 backup, gặp {first_backups}")
        first_data = load_clean(temp_clean)
        assert_projection(first_data, desired, "first apply")
        require(first_data.row_hashes == clean.data.row_hashes, "First apply đổi immutable hashes")

        second_changes, second_hash, second_backups = apply_once()
        require(second_changes == 0, f"Apply lần hai không idempotent: {second_changes} thay đổi")
        require(second_hash == first_hash, "Apply lần hai ghi lại CSV dù projection không đổi")
        require(second_backups == first_backups, "Apply lần hai tạo backup thừa")

        write_csv(
            temp_clean,
            clean.data.fieldnames,
            rows_with_projection(first_data.rows, reset_projection),
        )
        reset_again = load_clean(temp_clean)
        require(
            reset_again.row_hashes == clean.data.row_hashes,
            "Cleaner-reset simulation lần hai đổi immutable hashes",
        )
        recovery_changes, recovery_hash, recovery_backups = apply_once()
        require(
            recovery_changes == first_changes,
            f"Recovery đổi {recovery_changes} dòng, lần đầu đổi {first_changes}",
        )
        require(recovery_hash == first_hash, "Recovery không tái tạo byte-identical desired CSV")
        require(recovery_backups == first_backups + 1, "Recovery không tạo đúng một backup")
        recovered = load_clean(temp_clean)
        assert_projection(recovered, desired, "recovery apply")

        final_changes, final_hash, final_backups = apply_once()
        require(final_changes == 0, "Apply sau recovery không idempotent")
        require(final_hash == recovery_hash, "Apply sau recovery ghi lại file")
        require(final_backups == recovery_backups, "Apply sau recovery tạo backup thừa")
        return {
            "desired_rows_from_reset": first_changes,
            "backups_after_first": first_backups,
            "backups_after_recovery": recovery_backups,
        }


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise AssertionError(f"CSV rỗng: {path}") from exc
        rows = sum(1 for _ in reader)
    return rows, len(header)


def validate_clean_report(path: Path) -> None:
    if not path.exists():
        print(f"[SKIP] Không có clean report tùy chọn: {path}")
        return
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    require(report["nguồn"]["số_dòng"] == EXPECTED_SOURCE_ROWS, "Clean report source rows lệch")
    require(report["nguồn"]["số_cột"] == EXPECTED_SOURCE_COLUMNS, "Clean report source columns lệch")
    clean_result = report["kết_quả"]["TayY_clean.csv"]
    quarantine_result = report["kết_quả"]["TayY_quarantine.csv"]
    require(clean_result["số_dòng"] == EXPECTED_CLEAN_ROWS, "Clean report clean rows lệch")
    require(clean_result["số_cột"] == EXPECTED_CLEAN_COLUMNS, "Clean report clean columns lệch")
    require(quarantine_result["số_dòng"] == EXPECTED_QUARANTINE_ROWS, "Clean report quarantine rows lệch")
    require(
        report["thống_kê"][f"cờ::{DUPLICATE_FLAG}"] == EXPECTED_DUPLICATE_ROWS,
        "Clean report duplicate flag count lệch",
    )
    for column, expected in EXPECTED_JSON_NONEMPTY.items():
        require(
            report["thống_kê"][f"parse_ok::{column}"] == expected,
            f"Clean report parse_ok::{column} lệch",
        )
    require(report["thống_kê"]["loại_tên_rác"] == 9, "Clean report loại_tên_rác lệch")
    require(report["thống_kê"]["loại_bản_trùng"] == 167, "Clean report loại_bản_trùng lệch")
    require(
        report["thống_kê"]["trùng_tên_chưa_phân_xử_giữ"] == EXPECTED_DUPLICATE_ROWS,
        "Clean report trùng_tên_chưa_phân_xử_giữ lệch",
    )


def validate_optional_source_artifacts(source: Path, quarantine: Path) -> None:
    if source.exists():
        rows, columns = csv_shape(source)
        require((rows, columns) == (EXPECTED_SOURCE_ROWS, EXPECTED_SOURCE_COLUMNS), f"TayY.csv shape lệch: {(rows, columns)}")
        require(sha256_file(source) == EXPECTED_SOURCE_SHA256, "TayY.csv SHA-256 lệch corpus đã thẩm định")
    else:
        print(f"[SKIP] Không có source tùy chọn: {source}")
    if quarantine.exists():
        rows, columns = csv_shape(quarantine)
        require(rows == EXPECTED_QUARANTINE_ROWS, f"Quarantine chờ {EXPECTED_QUARANTINE_ROWS} dòng, gặp {rows}")
        require(columns == EXPECTED_SOURCE_COLUMNS + 2, f"Quarantine chờ {EXPECTED_SOURCE_COLUMNS + 2} cột, gặp {columns}")
        require(sha256_file(quarantine) == EXPECTED_QUARANTINE_SHA256, "Quarantine SHA-256 lệch corpus đã thẩm định")
    else:
        print(f"[SKIP] Không có quarantine tùy chọn: {quarantine}")


def validate_llm_proposals(
    path: Path,
    clean: CleanCorpus,
    catalog: CatalogData,
    proposals: ProposalCorpus,
) -> dict[str, dict[str, Any]]:
    """Sidecar đề xuất LLM (tùy chọn): loader production đã khóa schema/hash/eligibility/
    candidates/agreement; đây kiểm thêm bất biến mức pipeline."""
    llm = load_llm_proposals(path, clean.data, catalog, proposals.by_id)
    for disease_id, record in llm.items():
        base = proposals.by_id[disease_id]
        exact = (base.get("components") or {}).get("exact") or {}
        require(
            not exact.get("conflict"),
            f"LLM proposal {disease_id}: dòng exact-conflict phải do người xử tay, "
            "không được nhận mã LLM",
        )
        consensus = record.get("consensus") or {}
        require(
            consensus.get("kind") in LLM_CODE_KINDS,
            f"LLM proposal {disease_id}: loader trả record không mang mã ({consensus.get('kind')!r})",
        )
        require(
            record.get("_code") == catalog.canonical_code(consensus.get("code")),
            f"LLM proposal {disease_id}: _code không canonical",
        )
    return llm


def snapshot_existing(paths: Iterable[Path]) -> dict[Path, str]:
    return {path.resolve(): sha256_file(path) for path in set(paths) if path.is_file()}


def assert_snapshots_unchanged(before: Mapping[Path, str]) -> None:
    for path, digest in before.items():
        require(path.is_file(), f"Regression đã làm mất artifact thật: {path}")
        require(sha256_file(path) == digest, f"Regression đã thay đổi artifact thật: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-corpus regression cho mapping TayY -> VN ICD-10; không ghi artifact thật."
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument(
        "--materialized",
        type=Path,
        help="CSV materialized riêng; mặc định kiểm tra mapping ngay trên --clean",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--catalog-meta", type=Path, default=DEFAULT_CATALOG_META)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--llm-proposals", type=Path, default=DEFAULT_LLM_PROPOSALS)
    parser.add_argument("--clean-report", type=Path, default=DEFAULT_CLEAN_REPORT)
    parser.add_argument("--mapping-report", type=Path, default=DEFAULT_MAPPING_REPORT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    materialized_path = args.materialized or args.clean
    watched_paths = [
        args.clean,
        materialized_path,
        args.catalog,
        args.catalog_meta,
        args.proposals,
        args.review_queue,
        args.reviews,
        args.llm_proposals,
        args.clean_report,
        args.mapping_report,
        args.source,
        args.quarantine,
    ]
    real_snapshots = snapshot_existing(watched_paths)

    catalog = load_catalog_contract(args.catalog, args.catalog_meta)
    clean = load_clean_contract(args.clean, "clean CSV")
    duplicate_ids = validate_duplicate_name_contract(clean)
    current = clean
    if materialized_path.resolve() != args.clean.resolve():
        current = load_clean_contract(materialized_path, "materialized CSV")
        require(
            current.data.row_hashes == clean.data.row_hashes,
            "Materialized CSV khác immutable corpus của clean CSV",
        )
    status_counts = validate_status_code_pairs(current, catalog, "current CSV")

    proposals = validate_proposals(args.proposals, clean, catalog, duplicate_ids)
    validate_review_queue(args.review_queue, proposals)
    validate_mapping_report(args.mapping_report, clean, catalog, proposals)
    reviews = load_reviews(args.reviews, clean.data, catalog, required=False)
    llm_proposals = validate_llm_proposals(args.llm_proposals, clean, catalog, proposals)
    desired, materialized_state = validate_build_plan(
        current,
        catalog,
        proposals,
        reviews,
        llm_proposals,
    )

    validate_clean_report(args.clean_report)
    validate_optional_source_artifacts(args.source, args.quarantine)
    simulation = simulate_materialization(
        clean, catalog, proposals, reviews, llm_proposals, desired
    )
    assert_snapshots_unchanged(real_snapshots)

    print("[PASS] Full-corpus TayY ICD-10 regression")
    print(f"  clean rows / columns       : {EXPECTED_CLEAN_ROWS} / {EXPECTED_CLEAN_COLUMNS}")
    print(f"  JSON array cells validated : {EXPECTED_CLEAN_ROWS * len(JSON_ARRAY_COLUMNS)}")
    print(f"  duplicate rows / groups    : {len(duplicate_ids)} / {EXPECTED_DUPLICATE_GROUPS}")
    print(f"  catalog rows / SHA-256     : {len(catalog.records)} / {catalog.sha256}")
    print(f"  proposals / provisional    : {len(proposals.by_id)} / {proposals.provisional_count}")
    print(f"  review queue               : {EXPECTED_CLEAN_ROWS} rows, payload/order exact")
    print(f"  reviews                    : {len(reviews)}")
    print(f"  llm consensus codes        : {len(llm_proposals)}")
    print(f"  current CSV state          : {materialized_state}")
    print(f"  current status counts      : {dict(sorted(status_counts.items()))}")
    print(
        "  temp apply/recovery       : "
        f"{simulation['desired_rows_from_reset']} mapped rows; "
        "second applies were no-op"
    )
    print("  real source/quarantine/data: byte hashes unchanged")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ContractError, json.JSONDecodeError, csv.Error) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
