#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Đề xuất ánh xạ TayY_clean sang danh mục VN ICD-10 TT06/2026.

Script này chỉ tạo *proposal* để chuyên gia duyệt; không sửa ``TayY_clean.csv`` và
không tự coi kết quả truy hồi là mã chẩn đoán. Chỉ Tier A rất hẹp mới có
``provisional_code``: tên bệnh và lead của mô tả phải exact (có dấu) vào cùng một
mã duy nhất, mã đó được catalog đánh dấu ``auto_single_code_eligible``, và dòng
không có bất kỳ cờ/rủi ro UI/Đông-y/thiếu thông tin nào. BM25 chỉ xếp top-5 ứng
viên để duyệt, tuyệt đối không cấp mã provisional.

Ví dụ::

    .venv\\Scripts\\python.exe scripts/map_tayy_icd10.py --flagged-only
    .venv\\Scripts\\python.exe scripts/map_tayy_icd10.py --all --write

Mặc định là dry-run. Phải chọn đúng một scope ``--flagged-only`` hoặc ``--all``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

csv.field_size_limit(10**9)

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "TayY_clean.csv"
CATALOG_JSONL = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.jsonl"
CATALOG_META = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.meta.json"
PROPOSALS_JSONL = ROOT / "data" / "icd10" / "tayy_icd10_proposals.jsonl"
REVIEW_QUEUE_CSV = ROOT / "data" / "icd10" / "tayy_icd10_review_queue.csv"
REPORT_JSON = ROOT / "data" / "tayy_icd10_report.json"

MAPPER_VERSION = "1.0.0"
PROPOSAL_SCHEMA_VERSION = "tayy-icd10-proposal-v1"
REPORT_SCHEMA_VERSION = "tayy-icd10-report-v1"
ROW_FINGERPRINT_ALGORITHM = (
    "sha256(utf8(json-object-of-raw-row;exclude=icd10_code,"
    "trạng_thái_kiểm_duyệt;ensure_ascii=false;sort_keys=true;separators=(',',':')))"
)
FINGERPRINT_EXCLUDED_COLUMNS = frozenset(
    {"icd10_code", "trạng_thái_kiểm_duyệt"}
)
DUPLICATE_NAME_FLAG = "trung_ten_khac_noi_dung"
TOP_K = 5

REQUIRED_TAYY_COLUMNS = frozenset(
    {
        "disease_id",
        "tên_bệnh",
        "mô_tả_bệnh",
        "loại_bệnh",
        "nguyên_nhân",
        "triệu_chứng",
        "bệnh_đi_kèm",
        "phương_pháp",
        "khoa_điều_trị",
        "kiểm_tra",
        "icd10_code",
        "trạng_thái_kiểm_duyệt",
        "cờ_chất_lượng",
    }
)
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

_ICD_CODE_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,2})?$")
_DISEASE_ID_RE = re.compile(r"^TAYY-[0-9]{5}$")
_BRACKETED_CLARIFIER_RE = re.compile(
    r"\s*(?:\([^()]*\)|\[[^\[\]]*\]|（[^（）]*）|【[^【】]*】)"
)
_LEAD_BOUNDARY_RE = re.compile(
    r"\s*(?:[,;:]|\(|\b(?:là|đề\s+cập|còn\s+được|cũng\s+được|được\s+gọi|"
    r"thường|chủ\s+yếu|chiếm|bao\s+gồm|xảy\s+ra|có\s+thể|được\s+xem|"
    r"được\s+coi)\b)",
    re.IGNORECASE,
)
_UI_TEMPLATE_RE = re.compile(r"^(?:bản\s+mẫu|template)\s*:", re.IGNORECASE)

_UI_EXACT_TITLES_RAW = ("Trang chủ", "Việt", "Name", "Phong bì")
_TCM_EXPLICIT_PHRASES_RAW = (
    "y học cổ truyền",
    "đông y",
    "trung y",
    "y học trung quốc",
    "y học trung hoa",
)
_TCM_CLASSICAL_PHRASES_RAW = (
    "nội kinh",
    "kim quỹ",
    "thương hàn luận",
    "ôn bệnh",
    "cổ kim y",
    "sách cổ",
    "y thư cổ",
    "chứng trị",
    "danh y biệt lục",
    "hoàng đế nội kinh",
    "bản thảo cương mục",
    "thiên kim",
    "ngoại khoa chính tông",
    "y tông kim giám",
    "cảnh nhạc toàn thư",
    "tố vấn",
    "linh khu",
)
_GENERIC_LOW_INFO_TITLES_RAW = (
    "bệnh",
    "chứng",
    "hội chứng",
    "ung thư",
    "viêm",
    "nhiễm trùng",
    "khối u",
    "rối loạn",
    "tên bệnh",
)
_CERTIFICATE_META_PHRASES_RAW = (
    "tên giấy chứng nhận",
    "tên chứng nhận bệnh",
)

# Từ quá phổ biến chỉ tạo nhiễu cho truy hồi. Exact matching không dùng danh sách
# này và luôn giữ nguyên dấu tiếng Việt.
_SPARSE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "bi",
        "boi",
        "cac",
        "co",
        "cua",
        "da",
        "day",
        "de",
        "do",
        "duoc",
        "gay",
        "hoac",
        "la",
        "mot",
        "nay",
        "nhieu",
        "nhung",
        "nhu",
        "o",
        "ra",
        "su",
        "thuong",
        "trong",
        "tu",
        "va",
        "ve",
        "voi",
    }
)

CATALOG_SEARCH_FIELDS: tuple[tuple[str, float, int], ...] = (
    ("title_vi", 5.0, 80),
    ("category_vi", 2.5, 80),
    ("block_vi", 1.0, 80),
    ("subdivision1_vi", 0.8, 80),
    ("subdivision2_vi", 0.8, 80),
    ("guidance_vi", 0.7, 120),
    ("title_en", 1.7, 80),
    ("who_title_en", 1.2, 80),
    ("who_inclusions_en", 0.8, 120),
    ("guidance_en", 0.5, 120),
)
QUERY_SEARCH_FIELDS: tuple[tuple[str, float, int], ...] = (
    ("name", 5.0, 40),
    ("lead", 4.0, 60),
    ("description", 1.0, 80),
    ("types", 1.3, 60),
    ("symptoms", 1.2, 100),
    ("tests", 0.8, 80),
    ("comorbidities", 0.6, 70),
    ("department", 0.7, 50),
    ("cause", 0.5, 60),
    ("method", 0.4, 50),
)
QUERY_FIELD_FEATURE_LIMITS = {
    "name": 16,
    "lead": 16,
    "description": 10,
    "types": 8,
    "symptoms": 12,
    "tests": 8,
    "comorbidities": 6,
    "department": 6,
    "cause": 8,
    "method": 6,
}
QUERY_FEATURE_GLOBAL_LIMIT = 72
QUERY_FEATURE_MIN_PER_NONEMPTY_FIELD = 2
BM25_MAX_POSTING_LENGTH = 2_500
BM25_K1 = 1.35
BM25_B = 0.72
BM25_MIN_IDF = 0.25
EXACT_RANK_BONUS = 10_000.0


class InvariantError(RuntimeError):
    """Dữ liệu đầu vào/đầu ra vi phạm contract đã khóa."""


@dataclass(frozen=True)
class SourceRow:
    line_number: int
    values: dict[str, str]
    parsed_lists: dict[str, tuple[str, ...]]
    flags: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class CatalogData:
    records: tuple[dict[str, Any], ...]
    by_code: dict[str, int]
    meta: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class QueryPlanTerm:
    feature: str
    total_multiplier: float
    field_multipliers: tuple[tuple[str, float], ...]


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_exact(text: str) -> str:
    """Chuẩn hóa hình thức nhưng giữ dấu; không accent-fold cho exact."""

    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).replace("_", " ")
    return " ".join(value.split())


def _fold_for_sparse(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = value.replace("đ", "d")
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if unicodedata.category(char) != "Mn"
    )
    return value


def _lexical_tokens(text: str, max_tokens: int) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", _fold_for_sparse(text))
    kept = [token for token in raw if len(token) > 1 and token not in _SPARSE_STOPWORDS]
    return kept[:max_tokens]


def _feature_counts(text: str, max_tokens: int) -> Counter[str]:
    tokens = _lexical_tokens(text, max_tokens)
    features: Counter[str] = Counter(f"u:{token}" for token in tokens)
    features.update(f"b:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def _first_sentence(text: str) -> str:
    # Lead chỉ nằm ở dòng văn bản đầu tiên có nội dung; không nối heading/dòng sau
    # vào cùng một câu trước khi tìm dấu kết thúc câu.
    first_line = next(
        (line.strip() for line in (text or "").replace("\r\n", "\n").split("\n") if line.strip()),
        "",
    )
    compact = " ".join(first_line.split())
    if not compact:
        return ""
    match = re.search(r"[.!?。！？](?:[\"'”’\])}]*)?(?:\s|$)", compact)
    return (compact[: match.start()] if match else compact).strip()


def extract_lead(description: str) -> tuple[str, str]:
    sentence = _first_sentence(description)
    if not sentence:
        return "", ""
    sentence = re.sub(r"^\s*\d{1,3}[.)、]\s*", "", sentence)
    match = _LEAD_BOUNDARY_RE.search(sentence)
    lead = (sentence[: match.start()] if match else sentence).strip(" \t\"'“”‘’.,;:-")
    return sentence, lead


def _title_variants(title: str) -> tuple[tuple[str, str], ...]:
    variants: list[tuple[str, str]] = [("primary", title)]
    stripped = title
    while True:
        next_value = _BRACKETED_CLARIFIER_RE.sub(" ", stripped)
        if next_value == stripped:
            break
        stripped = next_value
    stripped = " ".join(stripped.split()).strip(" ,;:-")
    if stripped and normalize_exact(stripped) != normalize_exact(title):
        variants.append(("bracket_stripped", stripped))
    return tuple(variants)


def _canonical_row_fingerprint(row: Mapping[str, str], fieldnames: Sequence[str]) -> str:
    payload = {
        column: row.get(column, "")
        for column in fieldnames
        if column not in FINGERPRINT_EXCLUDED_COLUMNS
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _require_bool(record: Mapping[str, Any], field: str, code: str) -> bool:
    value = record.get(field)
    if type(value) is not bool:  # bool, không chấp nhận 0/1
        raise InvariantError(f"Catalog {code}: {field} phải là boolean")
    return value


def load_catalog(catalog_path: Path, meta_path: Path) -> CatalogData:
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Không thấy catalog: {catalog_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Không thấy metadata catalog: {meta_path}")
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise InvariantError("Metadata catalog phải là JSON object")

    digest = sha256_file(catalog_path)
    if meta.get("catalog_sha256") != digest:
        raise InvariantError(
            "SHA256 catalog không khớp metadata: "
            f"meta={meta.get('catalog_sha256')!r}, thực tế={digest}"
        )

    records: list[dict[str, Any]] = []
    by_code: dict[str, int] = {}
    auto_eligible_count = 0
    with catalog_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise InvariantError(f"Catalog có dòng rỗng tại {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvariantError(f"Catalog JSON lỗi tại dòng {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise InvariantError(f"Catalog dòng {line_number} không phải object")
            code = record.get("code")
            if not isinstance(code, str) or not _ICD_CODE_RE.fullmatch(code):
                raise InvariantError(f"Catalog dòng {line_number}: mã không hợp lệ {code!r}")
            if code in by_code:
                raise InvariantError(f"Catalog trùng mã {code}")
            if record.get("stt") != line_number:
                raise InvariantError(
                    f"Catalog {code}: stt={record.get('stt')!r}, chờ {line_number}"
                )
            if record.get("code_nodot") != code.replace(".", ""):
                raise InvariantError(f"Catalog {code}: code_nodot lệch")
            if not isinstance(record.get("title_vi"), str) or not record["title_vi"].strip():
                raise InvariantError(f"Catalog {code}: title_vi rỗng")
            restrictions = record.get("restrictions")
            if not isinstance(restrictions, list) or not all(
                isinstance(item, str) and item for item in restrictions
            ):
                raise InvariantError(f"Catalog {code}: restrictions không phải list[str]")
            if len(restrictions) != len(set(restrictions)):
                raise InvariantError(f"Catalog {code}: restrictions trùng")

            dagger = _require_bool(record, "dagger", code)
            asterisk = _require_bool(record, "asterisk", code)
            codable = _require_bool(record, "codable", code)
            morbidity = _require_bool(record, "morbidity_eligible", code)
            primary = _require_bool(record, "primary_eligible", code)
            auto = _require_bool(record, "auto_single_code_eligible", code)
            if dagger and asterisk:
                raise InvariantError(f"Catalog {code}: vừa dagger vừa asterisk")
            expected_dual_marker = "dagger" if dagger else "asterisk" if asterisk else None
            if record.get("dual_coding_marker") != expected_dual_marker:
                raise InvariantError(f"Catalog {code}: dual_coding_marker lệch dagger/asterisk")
            if not isinstance(record.get("who_references"), list):
                raise InvariantError(f"Catalog {code}: who_references không phải list")
            if codable != ("requires_more_specific" not in restrictions):
                raise InvariantError(f"Catalog {code}: codable lệch restrictions")
            expected_morbidity = not {
                "requires_more_specific",
                "mortality_only",
            }.intersection(restrictions)
            if morbidity != expected_morbidity:
                raise InvariantError(f"Catalog {code}: morbidity_eligible lệch")
            expected_primary = not (
                {"not_primary", "discouraged_primary", "requires_more_specific", "mortality_only"}
                .intersection(restrictions)
                or asterisk
            )
            if primary != expected_primary:
                raise InvariantError(f"Catalog {code}: primary_eligible lệch")
            expected_primary_label = (
                "not_usable"
                if not morbidity
                else "forbidden"
                if "not_primary" in restrictions or asterisk
                else "discouraged"
                if "discouraged_primary" in restrictions
                else "allowed"
            )
            if record.get("primary_eligibility") != expected_primary_label:
                raise InvariantError(f"Catalog {code}: primary_eligibility lệch")
            expected_auto = expected_primary_label == "allowed" and not dagger and not asterisk
            if auto != expected_auto:
                raise InvariantError(f"Catalog {code}: auto_single_code_eligible lệch")

            auto_eligible_count += int(auto)
            by_code[code] = len(records)
            records.append(record)

    if meta.get("rows") != len(records) or meta.get("unique_codes") != len(records):
        raise InvariantError(
            "Số dòng catalog lệch metadata: "
            f"rows={len(records)}, meta.rows={meta.get('rows')}, "
            f"meta.unique_codes={meta.get('unique_codes')}"
        )
    meta_auto = (meta.get("statistics") or {}).get("auto_single_code_eligible")
    if meta_auto != auto_eligible_count:
        raise InvariantError(
            "Số mã auto_single_code_eligible lệch metadata: "
            f"catalog={auto_eligible_count}, meta={meta_auto}"
        )
    if len(records) == 0:
        raise InvariantError("Catalog rỗng")
    return CatalogData(tuple(records), by_code, meta, digest)


def _parse_json_array(raw: str, column: str, disease_id: str) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvariantError(
            f"{disease_id}: {column} không phải JSON array hợp lệ: {exc}"
        ) from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvariantError(f"{disease_id}: {column} phải là JSON array chỉ chứa string")
    return tuple(value)


def load_tayy(path: Path) -> tuple[list[SourceRow], list[str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy input: {path}")
    source_sha = sha256_file(path)
    loaded: list[SourceRow] = []
    seen_ids: set[str] = set()
    duplicate_groups: defaultdict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InvariantError("TayY_clean.csv thiếu header")
        fieldnames = list(reader.fieldnames)
        if len(fieldnames) != len(set(fieldnames)):
            raise InvariantError("TayY_clean.csv có tên cột trùng")
        missing = sorted(REQUIRED_TAYY_COLUMNS - set(fieldnames))
        if missing:
            raise InvariantError(f"TayY_clean.csv thiếu cột: {missing}")
        if not FINGERPRINT_EXCLUDED_COLUMNS.issubset(fieldnames):
            raise InvariantError("Thiếu cột phải loại khỏi row fingerprint")

        present_array_columns = [c for c in JSON_ARRAY_COLUMNS if c in fieldnames]
        for row_index, row in enumerate(reader, 1):
            if None in row:
                raise InvariantError(f"CSV dòng dữ liệu {row_index}: thừa field ngoài header")
            values = {column: row.get(column) or "" for column in fieldnames}
            disease_id = values["disease_id"]
            expected_id = f"TAYY-{row_index:05d}"
            if not _DISEASE_ID_RE.fullmatch(disease_id):
                raise InvariantError(f"CSV dòng {row_index}: disease_id sai {disease_id!r}")
            if disease_id != expected_id:
                raise InvariantError(
                    f"CSV không còn thứ tự ID ổn định: dòng {row_index} là {disease_id}, "
                    f"chờ {expected_id}"
                )
            if disease_id in seen_ids:
                raise InvariantError(f"disease_id trùng: {disease_id}")
            if not values["tên_bệnh"].strip():
                raise InvariantError(f"{disease_id}: tên_bệnh rỗng")
            seen_ids.add(disease_id)

            flags_raw = values["cờ_chất_lượng"]
            flags = tuple(part.strip() for part in flags_raw.split(";") if part.strip())
            if len(flags) != len(set(flags)) or flags != tuple(sorted(flags)):
                raise InvariantError(f"{disease_id}: cờ_chất_lượng không unique/sorted")
            parsed_lists = {
                column: _parse_json_array(values[column], column, disease_id)
                for column in present_array_columns
            }
            fingerprint = _canonical_row_fingerprint(values, fieldnames)
            loaded.append(
                SourceRow(row_index + 1, values, parsed_lists, flags, fingerprint)
            )
            duplicate_groups[values["tên_bệnh"].casefold()].append(disease_id)

    if not loaded:
        raise InvariantError("TayY_clean.csv không có dòng dữ liệu")
    for row in loaded:
        if DUPLICATE_NAME_FLAG in row.flags:
            group = duplicate_groups[row.values["tên_bệnh"].casefold()]
            if len(group) < 2:
                raise InvariantError(
                    f"{row.values['disease_id']}: có cờ {DUPLICATE_NAME_FLAG} "
                    "nhưng không còn dòng trùng tên"
                )
    fingerprints = {row.fingerprint for row in loaded}
    if len(fingerprints) != len(loaded):
        raise InvariantError("Row fingerprint trùng (disease_id đáng lẽ làm mỗi dòng duy nhất)")
    return loaded, fieldnames, source_sha


class ExactTitleIndex:
    """Chỉ index title_vi và biến thể bỏ phần chú giải trong ngoặc."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        index: defaultdict[str, defaultdict[int, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for doc_index, record in enumerate(records):
            for variant_type, variant in _title_variants(record["title_vi"]):
                normalized = normalize_exact(variant)
                if normalized:
                    index[normalized][doc_index].add(variant_type)
        self._index = {
            term: {
                doc_index: tuple(sorted(variant_types))
                for doc_index, variant_types in doc_map.items()
            }
            for term, doc_map in index.items()
        }

    def lookup(self, normalized: str) -> dict[int, tuple[str, ...]]:
        return dict(self._index.get(normalized, {}))


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(item for item in value if isinstance(item, str))
    return ""


class SparseBM25Index:
    """Inverted BM25 đa trường; chỉ sinh/xếp ứng viên, không quyết định mã."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        raw_postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        doc_lengths: list[float] = []
        for doc_index, record in enumerate(records):
            weighted: defaultdict[str, float] = defaultdict(float)
            for field, boost, max_tokens in CATALOG_SEARCH_FIELDS:
                counts = _feature_counts(_text_value(record.get(field, "")), max_tokens)
                for feature, frequency in counts.items():
                    feature_boost = 1.35 if feature.startswith("b:") else 1.0
                    weighted[feature] += frequency * boost * feature_boost
            doc_length = sum(weighted.values()) or 1.0
            doc_lengths.append(doc_length)
            for feature, frequency in weighted.items():
                raw_postings[feature].append((doc_index, frequency))

        self.doc_lengths = tuple(doc_lengths)
        self.codes = tuple(str(record["code"]) for record in records)
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths)
        self.document_count = len(doc_lengths)
        self.idf = {
            feature: math.log(
                1.0 + (self.document_count - len(items) + 0.5) / (len(items) + 0.5)
            )
            for feature, items in raw_postings.items()
        }
        # Mọi đại lượng phụ thuộc document được tính đúng một lần lúc dựng index.
        # Query loop sau đó chỉ còn phép nhân/cộng sparse, thay vì tính BM25 lại cho
        # hàng triệu posting ở mỗi dòng TayY.
        self.postings: dict[str, tuple[tuple[int, float], ...]] = {}
        for feature, items in raw_postings.items():
            idf = self.idf[feature]
            impacts = []
            for doc_index, term_frequency in items:
                length_norm = BM25_K1 * (
                    1.0
                    - BM25_B
                    + BM25_B * self.doc_lengths[doc_index] / self.avg_doc_length
                )
                impact = (
                    idf
                    * (term_frequency * (BM25_K1 + 1.0))
                    / (term_frequency + length_norm)
                )
                impacts.append((doc_index, impact))
            self.postings[feature] = tuple(impacts)

    def score(
        self,
        query_fields: Mapping[str, str],
        retain_doc_indices: Iterable[int] = (),
    ) -> tuple[dict[int, float], tuple[QueryPlanTerm, ...]]:
        per_field: dict[str, list[tuple[float, str, float]]] = {}
        for field, query_boost, max_tokens in QUERY_SEARCH_FIELDS:
            counts = _feature_counts(query_fields.get(field, ""), max_tokens)
            ranked_features: list[tuple[float, str, float]] = []
            for feature, query_frequency in counts.items():
                posting = self.postings.get(feature)
                if (
                    not posting
                    or len(posting) > BM25_MAX_POSTING_LENGTH
                    or self.idf[feature] < BM25_MIN_IDF
                ):
                    continue
                multiplier = query_boost * (1.0 + math.log(query_frequency))
                if feature.startswith("b:"):
                    multiplier *= 1.25
                priority = multiplier * self.idf[feature]
                ranked_features.append((priority, feature, multiplier))
            ranked_features.sort(key=lambda item: (-item[0], item[1]))
            limit = QUERY_FIELD_FEATURE_LIMITS[field]
            if ranked_features:
                per_field[field] = ranked_features[:limit]

        # Giữ tối thiểu vài feature hiếm của mỗi trường rồi lấp ngân sách chung theo
        # độ phân biệt. Nhờ vậy retrieval vẫn thực sự đa trường mà runtime bị chặn.
        selected_pairs: list[tuple[str, float, str, float]] = []
        selected_keys: set[tuple[str, str]] = set()
        for field, entries in per_field.items():
            for priority, feature, multiplier in entries[:QUERY_FEATURE_MIN_PER_NONEMPTY_FIELD]:
                selected_pairs.append((field, priority, feature, multiplier))
                selected_keys.add((field, feature))
        remaining = [
            (field, priority, feature, multiplier)
            for field, entries in per_field.items()
            for priority, feature, multiplier in entries
            if (field, feature) not in selected_keys
        ]
        remaining.sort(key=lambda item: (-item[1], item[0], item[2]))
        room = max(0, QUERY_FEATURE_GLOBAL_LIMIT - len(selected_pairs))
        selected_pairs.extend(remaining[:room])

        combined: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
        for field, _priority, feature, multiplier in selected_pairs:
            combined[feature].append((field, multiplier))
        plan = tuple(
            QueryPlanTerm(
                feature=feature,
                total_multiplier=sum(multiplier for _field, multiplier in multipliers),
                field_multipliers=tuple(sorted(multipliers)),
            )
            for feature, multipliers in sorted(combined.items())
        )

        totals: defaultdict[int, float] = defaultdict(float)
        for term in plan:
            for doc_index, impact in self.postings[term.feature]:
                totals[doc_index] += term.total_multiplier * impact
        # Chỉ top-5 sparse có thể xuất hiện ở output. Giữ thêm exact docs để báo
        # đúng BM25 component của chúng; bỏ phần đuôi trước bước xếp hạng cuối giúp
        # tránh duyệt hàng nghìn doc lần thứ hai cho mỗi bệnh.
        best_sparse = heapq.nsmallest(
            TOP_K,
            totals,
            key=lambda doc_index: (-totals[doc_index], self.codes[doc_index]),
        )
        retained = set(best_sparse) | set(retain_doc_indices)
        return {doc_index: totals.get(doc_index, 0.0) for doc_index in retained}, plan

    def component_scores(
        self, plan: Sequence[QueryPlanTerm], doc_indices: Iterable[int]
    ) -> dict[str, dict[int, float]]:
        selected = set(doc_indices)
        by_field: defaultdict[str, defaultdict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        if not selected:
            return {}
        for term in plan:
            for doc_index, impact in self.postings[term.feature]:
                if doc_index not in selected:
                    continue
                for field, multiplier in term.field_multipliers:
                    by_field[field][doc_index] += multiplier * impact
        return {field: dict(scores) for field, scores in by_field.items()}


def _phrase_hits(normalized_text: str, phrases_raw: Iterable[str]) -> list[str]:
    padded = f" {normalized_text} "
    hits = []
    for phrase in phrases_raw:
        normalized_phrase = normalize_exact(phrase)
        if f" {normalized_phrase} " in padded:
            hits.append(phrase)
    return hits


def classify_risks(name: str, description: str, first_sentence: str) -> dict[str, Any]:
    name_norm = normalize_exact(name)
    sentence_norm = normalize_exact(first_sentence)
    scan_norm = normalize_exact(f"{name} {first_sentence}")

    ui_reasons: list[str] = []
    ui_exact = {normalize_exact(value) for value in _UI_EXACT_TITLES_RAW}
    if name_norm in ui_exact:
        ui_reasons.append("known_ui_title")
    if _UI_TEMPLATE_RE.match(name.strip()):
        ui_reasons.append("template_title")

    tcm_explicit = _phrase_hits(scan_norm, _TCM_EXPLICIT_PHRASES_RAW)
    tcm_classical = _phrase_hits(scan_norm, _TCM_CLASSICAL_PHRASES_RAW)

    low_info_reasons: list[str] = []
    meaningful_sentence_chars = sum(char.isalnum() for char in first_sentence)
    if not description.strip():
        low_info_reasons.append("description_empty")
    elif meaningful_sentence_chars < 30:
        low_info_reasons.append("first_sentence_under_30_alnum")
    if sentence_norm and sentence_norm == name_norm:
        low_info_reasons.append("first_sentence_equals_name")
    if name_norm in {normalize_exact(value) for value in _GENERIC_LOW_INFO_TITLES_RAW}:
        low_info_reasons.append("generic_title")
    meta_hits = _phrase_hits(scan_norm, _CERTIFICATE_META_PHRASES_RAW)
    if sentence_norm.endswith(normalize_exact("tên bệnh")):
        meta_hits.append("first_sentence_ends_with_tên_bệnh")
    if meta_hits:
        low_info_reasons.append("certificate_or_name_metadata")

    return {
        "ui": bool(ui_reasons),
        "ui_reasons": ui_reasons,
        "tcm": bool(tcm_explicit or tcm_classical),
        "tcm_explicit_hits": tcm_explicit,
        "tcm_classical_hits": tcm_classical,
        "low_info": bool(low_info_reasons),
        "low_info_reasons": low_info_reasons,
    }


def _join_list(row: SourceRow, column: str) -> str:
    return " ".join(row.parsed_lists.get(column, ()))


def _query_fields(row: SourceRow, lead: str) -> dict[str, str]:
    values = row.values
    return {
        "name": values["tên_bệnh"],
        "lead": lead,
        "description": values["mô_tả_bệnh"],
        "types": _join_list(row, "loại_bệnh"),
        "symptoms": _join_list(row, "triệu_chứng"),
        "tests": _join_list(row, "kiểm_tra"),
        "comorbidities": _join_list(row, "bệnh_đi_kèm"),
        "department": values["khoa_điều_trị"],
        "cause": values["nguyên_nhân"],
        "method": values["phương_pháp"],
    }


def _candidate_payload(
    rank: int,
    doc_index: int,
    record: Mapping[str, Any],
    bm25_score: float,
    field_scores: Mapping[str, Mapping[int, float]],
    name_hits: Mapping[int, tuple[str, ...]],
    lead_hits: Mapping[int, tuple[str, ...]],
) -> dict[str, Any]:
    exact_name = doc_index in name_hits
    exact_lead = doc_index in lead_hits
    exact_bonus = EXACT_RANK_BONUS * (int(exact_name) + int(exact_lead))
    retrieved_by: list[str] = []
    if exact_name:
        retrieved_by.extend(f"name_exact:{kind}" for kind in name_hits[doc_index])
    if exact_lead:
        retrieved_by.extend(f"lead_exact:{kind}" for kind in lead_hits[doc_index])
    bm25_by_field = {
        field: round(scores[doc_index], 6)
        for field, scores in field_scores.items()
        if scores.get(doc_index, 0.0) > 0.0
    }
    if bm25_score > 0.0:
        retrieved_by.append("sparse_bm25")
    dual_marker = record.get("dual_coding_marker")
    return {
        "rank": rank,
        "code": record["code"],
        "title_vi": record["title_vi"],
        "title_en": record.get("title_en", ""),
        "category_vi": record.get("category_vi", ""),
        "guidance_vi": record.get("guidance_vi", ""),
        "retrieved_by": retrieved_by,
        "score": round(bm25_score + exact_bonus, 6),
        "components": {
            "bm25": round(bm25_score, 6),
            "bm25_by_tayy_field": bm25_by_field,
            "name_exact": exact_name,
            "name_exact_variant_types": list(name_hits.get(doc_index, ())),
            "lead_exact": exact_lead,
            "lead_exact_variant_types": list(lead_hits.get(doc_index, ())),
            "exact_rank_bonus": exact_bonus,
        },
        "eligibility": {
            "auto_single_code_eligible": record["auto_single_code_eligible"],
            "codable": record["codable"],
            "morbidity_eligible": record["morbidity_eligible"],
            "primary_eligible": record["primary_eligible"],
            "primary_eligibility": record["primary_eligibility"],
            "dagger": record["dagger"],
            "asterisk": record["asterisk"],
            "dual_coding_marker": dual_marker,
            "pair_resolution_required": bool(dual_marker),
            "who_references": record.get("who_references", []) if dual_marker else [],
            "restrictions": record["restrictions"],
        },
        "source_audit": {
            "source_page": record.get("source_page"),
            "source_code_raw": record.get("source_code_raw"),
            "source_anomaly": record.get("source_anomaly"),
            "source_title_en_status": record.get("source_title_en_status"),
            "source_title_en_warning": record.get("source_title_en_warning"),
        },
    }


def build_proposal(
    row: SourceRow,
    catalog: CatalogData,
    exact_index: ExactTitleIndex,
    sparse_index: SparseBM25Index,
    source_sha: str,
) -> dict[str, Any]:
    values = row.values
    name = values["tên_bệnh"].strip()
    first_sentence, lead = extract_lead(values["mô_tả_bệnh"])
    name_norm = normalize_exact(name)
    lead_norm = normalize_exact(lead)
    name_hits = exact_index.lookup(name_norm)
    lead_hits = exact_index.lookup(lead_norm) if lead_norm else {}
    name_codes = sorted(catalog.records[index]["code"] for index in name_hits)
    lead_codes = sorted(catalog.records[index]["code"] for index in lead_hits)
    name_code_set = set(name_codes)
    lead_code_set = set(lead_codes)
    exact_conflict = bool(
        name_code_set and lead_code_set and name_code_set.isdisjoint(lead_code_set)
    )

    risks = classify_risks(name, values["mô_tả_bệnh"], first_sentence)
    has_duplicate_name_flag = DUPLICATE_NAME_FLAG in row.flags
    unflagged = not row.flags

    sparse_scores, query_plan = sparse_index.score(
        _query_fields(row, lead), set(name_hits) | set(lead_hits)
    )
    candidate_doc_indices = set(sparse_scores) | set(name_hits) | set(lead_hits)

    def rank_key(doc_index: int) -> tuple[float, str]:
        exact_count = int(doc_index in name_hits) + int(doc_index in lead_hits)
        score = sparse_scores.get(doc_index, 0.0) + EXACT_RANK_BONUS * exact_count
        return (-score, catalog.records[doc_index]["code"])

    selected_indices = heapq.nsmallest(TOP_K, candidate_doc_indices, key=rank_key)
    field_scores = sparse_index.component_scores(query_plan, selected_indices)
    candidates = [
        _candidate_payload(
            rank,
            doc_index,
            catalog.records[doc_index],
            sparse_scores.get(doc_index, 0.0),
            field_scores,
            name_hits,
            lead_hits,
        )
        for rank, doc_index in enumerate(selected_indices, 1)
    ]

    same_unique_code: str | None = None
    if len(name_codes) == 1 and len(lead_codes) == 1 and name_codes[0] == lead_codes[0]:
        same_unique_code = name_codes[0]
    matched_record = (
        catalog.records[catalog.by_code[same_unique_code]] if same_unique_code else None
    )
    eligible = bool(matched_record and matched_record["auto_single_code_eligible"])

    tier_a = bool(
        same_unique_code
        and eligible
        and not exact_conflict
        and unflagged
        and not has_duplicate_name_flag
        and not risks["ui"]
        and not risks["tcm"]
        and not risks["low_info"]
    )
    provisional_code = same_unique_code if tier_a else None

    decision_reasons: list[str] = []
    if exact_conflict:
        decision_reasons.append("exact_name_lead_conflict")
    if not name_codes:
        decision_reasons.append("name_not_exact")
    elif len(name_codes) != 1:
        decision_reasons.append("name_exact_not_unique")
    if not lead_codes:
        decision_reasons.append("lead_not_exact")
    elif len(lead_codes) != 1:
        decision_reasons.append("lead_exact_not_unique")
    if len(name_codes) == 1 and len(lead_codes) == 1 and name_codes[0] != lead_codes[0]:
        if "exact_name_lead_conflict" not in decision_reasons:
            decision_reasons.append("exact_name_lead_conflict")
    if same_unique_code and not eligible:
        decision_reasons.append("exact_code_not_auto_single_code_eligible")
    if row.flags:
        decision_reasons.append("quality_flags_present")
    if has_duplicate_name_flag:
        decision_reasons.append("duplicate_name_unresolved")
    if risks["ui"]:
        decision_reasons.append("ui_or_template_row")
    if risks["tcm"]:
        decision_reasons.append("tcm_or_classical_reference")
    if risks["low_info"]:
        decision_reasons.append("low_information")
    if tier_a:
        decision_reasons.append("tier_a_exact_consensus_requires_human_review")

    return {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "disease_id": values["disease_id"],
        "source_csv_line": row.line_number,
        "row_sha256": row.fingerprint,
        "row_fingerprint": row.fingerprint,
        "source_state": {
            "icd10_code": values["icd10_code"],
            "trạng_thái_kiểm_duyệt": values["trạng_thái_kiểm_duyệt"],
        },
        "name": name,
        "evidence": {
            "description": values["mô_tả_bệnh"],
            "types": list(row.parsed_lists.get("loại_bệnh", ())),
            "symptoms": list(row.parsed_lists.get("triệu_chứng", ())),
            "tests": list(row.parsed_lists.get("kiểm_tra", ())),
            "comorbidities": list(row.parsed_lists.get("bệnh_đi_kèm", ())),
            "department": values["khoa_điều_trị"],
            "cause": values["nguyên_nhân"],
            "method": values["phương_pháp"],
        },
        "proposal_tier": "A" if tier_a else None,
        "provisional_code": provisional_code,
        "requires_human_review": True,
        "decision_reasons": decision_reasons,
        "components": {
            "name": {"raw": name, "exact_normalized": name_norm},
            "lead": {"raw": lead, "exact_normalized": lead_norm},
            "first_sentence": first_sentence,
            "exact": {
                "accent_sensitive": True,
                "name_codes": name_codes,
                "lead_codes": lead_codes,
                "same_unique_code": same_unique_code,
                "conflict": exact_conflict,
            },
            "quality": {
                "flags": list(row.flags),
                "unflagged": unflagged,
                "duplicate_name_unresolved": has_duplicate_name_flag,
            },
            "risk_gates": risks,
            "retrieval": {
                "bm25_query_features_used": len(query_plan),
                "bm25_query_feature_cap": QUERY_FEATURE_GLOBAL_LIMIT,
            },
        },
        "candidates": candidates,
        "provenance": {
            "mapper": "scripts/map_tayy_icd10.py",
            "mapper_version": MAPPER_VERSION,
            "input_path": "data/TayY_clean.csv",
            "input_sha256": source_sha,
            "row_fingerprint_algorithm": ROW_FINGERPRINT_ALGORITHM,
            "row_fingerprint_excluded_columns": sorted(FINGERPRINT_EXCLUDED_COLUMNS),
            "catalog_path": "data/icd10/vn_icd10_tt06_2026.jsonl",
            "catalog_system": catalog.meta.get("system"),
            "catalog_who_release": catalog.meta.get("who_release"),
            "catalog_sha256": catalog.sha256,
            "catalog_source_sha256": catalog.meta.get("source_sha256"),
            "exact_title_variants": ["title_vi", "title_vi_without_bracketed_clarifier"],
            "sparse_retrieval": {
                "method": "inverted_bm25_multi_field",
                "accent_folded": True,
                "candidate_only": True,
                "top_k": TOP_K,
                "k1": BM25_K1,
                "b": BM25_B,
                "max_posting_length": BM25_MAX_POSTING_LENGTH,
                "query_feature_cap": QUERY_FEATURE_GLOBAL_LIMIT,
            },
        },
    }


def _proposal_priority(proposal: Mapping[str, Any]) -> str:
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


def _csv_safe(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


REVIEW_COLUMNS = (
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
)


def _review_row(proposal: Mapping[str, Any]) -> dict[str, str]:
    components = proposal["components"]
    exact = components["exact"]
    risk = components["risk_gates"]
    compact_candidates = [
        {
            "rank": item["rank"],
            "code": item["code"],
            "title_vi": item["title_vi"],
            "title_en": item["title_en"],
            "score": item["score"],
            "retrieved_by": item["retrieved_by"],
            "auto_single_code_eligible": item["eligibility"]["auto_single_code_eligible"],
            "primary_eligibility": item["eligibility"]["primary_eligibility"],
            "restrictions": item["eligibility"]["restrictions"],
        }
        for item in proposal["candidates"]
    ]
    values = {
        "priority": _proposal_priority(proposal),
        "disease_id": proposal["disease_id"],
        "tên_bệnh": proposal["name"],
        "mô_tả_bệnh": proposal["evidence"]["description"],
        "lead": components["lead"]["raw"],
        "loại_bệnh": _stable_json(proposal["evidence"]["types"]),
        "triệu_chứng": _stable_json(proposal["evidence"]["symptoms"]),
        "kiểm_tra": _stable_json(proposal["evidence"]["tests"]),
        "khoa_điều_trị": proposal["evidence"]["department"],
        "nguyên_nhân": proposal["evidence"]["cause"],
        "bệnh_đi_kèm": _stable_json(proposal["evidence"]["comorbidities"]),
        "phương_pháp": proposal["evidence"]["method"],
        "proposal_tier": proposal["proposal_tier"] or "",
        "provisional_code": proposal["provisional_code"] or "",
        "decision_reasons": ";".join(proposal["decision_reasons"]),
        "exact_name_codes": _stable_json(exact["name_codes"]),
        "exact_lead_codes": _stable_json(exact["lead_codes"]),
        "top_candidates": _stable_json(compact_candidates),
        "cờ_chất_lượng": ";".join(components["quality"]["flags"]),
        "ui_flag": str(risk["ui"]).lower(),
        "tcm_flag": str(risk["tcm"]).lower(),
        "low_info_flag": str(risk["low_info"]).lower(),
        "exact_conflict": str(exact["conflict"]).lower(),
        "current_icd10_code": proposal["source_state"]["icd10_code"],
        "current_review_status": proposal["source_state"]["trạng_thái_kiểm_duyệt"],
        "decision": "",
        "icd10_code": "",
        "reviewer": "",
        "reviewed_at": "",
        "review_note": "",
        "row_sha256": proposal["row_sha256"],
        "row_fingerprint": proposal["row_fingerprint"],
    }
    return {column: _csv_safe(values[column]) for column in REVIEW_COLUMNS}


def verify_proposals(
    proposals: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[SourceRow],
    catalog: CatalogData,
) -> dict[str, bool]:
    if len(proposals) != len(selected_rows):
        raise InvariantError("Số proposal không bằng số dòng trong scope")
    expected_ids = [row.values["disease_id"] for row in selected_rows]
    actual_ids = [proposal["disease_id"] for proposal in proposals]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise InvariantError("Proposal không one-to-one/đúng thứ tự input")

    catalog_codes = set(catalog.by_code)
    for proposal, source_row in zip(proposals, selected_rows):
        if (
            proposal["row_sha256"] != source_row.fingerprint
            or proposal["row_fingerprint"] != source_row.fingerprint
        ):
            raise InvariantError(f"{proposal['disease_id']}: row fingerprint lệch")
        candidates = proposal["candidates"]
        if len(candidates) > TOP_K:
            raise InvariantError(f"{proposal['disease_id']}: quá {TOP_K} candidates")
        candidate_codes = [candidate["code"] for candidate in candidates]
        if len(candidate_codes) != len(set(candidate_codes)):
            raise InvariantError(f"{proposal['disease_id']}: candidate code trùng")
        if not set(candidate_codes).issubset(catalog_codes):
            raise InvariantError(f"{proposal['disease_id']}: candidate ngoài catalog")
        if [candidate["rank"] for candidate in candidates] != list(
            range(1, len(candidates) + 1)
        ):
            raise InvariantError(f"{proposal['disease_id']}: rank candidate không liên tục")

        provisional = proposal["provisional_code"]
        if provisional is None:
            if proposal["proposal_tier"] is not None:
                raise InvariantError(f"{proposal['disease_id']}: tier không-null nhưng thiếu mã")
            continue
        exact = proposal["components"]["exact"]
        quality = proposal["components"]["quality"]
        risks = proposal["components"]["risk_gates"]
        if proposal["proposal_tier"] != "A":
            raise InvariantError(f"{proposal['disease_id']}: provisional không phải Tier A")
        if not (
            exact["same_unique_code"] == provisional
            and exact["name_codes"] == [provisional]
            and exact["lead_codes"] == [provisional]
            and not exact["conflict"]
        ):
            raise InvariantError(f"{proposal['disease_id']}: provisional thiếu exact consensus")
        record = catalog.records[catalog.by_code[provisional]]
        if not record["auto_single_code_eligible"]:
            raise InvariantError(f"{proposal['disease_id']}: provisional code không eligible")
        if (
            not quality["unflagged"]
            or quality["duplicate_name_unresolved"]
            or risks["ui"]
            or risks["tcm"]
            or risks["low_info"]
        ):
            raise InvariantError(f"{proposal['disease_id']}: provisional vượt hard gate")
        if provisional not in candidate_codes:
            raise InvariantError(f"{proposal['disease_id']}: provisional không nằm trong top-5")

    return {
        "catalog_sha256_matches_meta": True,
        "catalog_codes_unique_and_structurally_valid": True,
        "catalog_eligibility_fields_recomputed": True,
        "tayy_ids_unique_contiguous_and_ordered": True,
        "tayy_json_array_columns_valid": True,
        "row_sha256_excludes_exactly_code_and_review_status": True,
        "one_proposal_per_scoped_row_in_input_order": True,
        "candidate_codes_unique_catalog_members_top5": True,
        "bm25_is_candidate_only": True,
        "provisional_is_tier_a_exact_consensus_only": True,
        "duplicate_name_flag_never_provisional": True,
        "all_proposals_require_human_review": True,
    }


def build_report(
    *,
    scope: str,
    write_requested: bool,
    all_rows: Sequence[SourceRow],
    selected_rows: Sequence[SourceRow],
    proposals: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    source_sha: str,
    catalog: CatalogData,
    invariants: Mapping[str, bool],
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    candidate_count_distribution: Counter[str] = Counter()
    for proposal in proposals:
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
        candidate_count_distribution[str(len(proposal["candidates"]))] += 1
        reason_counts.update(proposal["decision_reasons"])

    duplicate_group_sizes = Counter(
        row.values["tên_bệnh"].casefold()
        for row in all_rows
        if DUPLICATE_NAME_FLAG in row.flags
    )
    total_flagged = sum(duplicate_group_sizes.values())
    duplicate_size_distribution = Counter(duplicate_group_sizes.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mapper_version": MAPPER_VERSION,
        "mode": "write" if write_requested else "dry-run",
        "scope": scope,
        "inputs": {
            "tayy_csv": {
                "path": "data/TayY_clean.csv",
                "sha256": source_sha,
                "rows": len(all_rows),
                "columns": len(fieldnames),
                "duplicate_name_flagged_rows": total_flagged,
                "duplicate_name_flagged_groups": len(duplicate_group_sizes),
                "duplicate_name_group_size_distribution": {
                    str(size): count
                    for size, count in sorted(duplicate_size_distribution.items())
                },
            },
            "catalog": {
                "path": "data/icd10/vn_icd10_tt06_2026.jsonl",
                "meta_path": "data/icd10/vn_icd10_tt06_2026.meta.json",
                "system": catalog.meta.get("system"),
                "who_release": catalog.meta.get("who_release"),
                "sha256": catalog.sha256,
                "source_sha256": catalog.meta.get("source_sha256"),
                "rows": len(catalog.records),
            },
        },
        "outputs": {
            "written": write_requested,
            "proposals": "data/icd10/tayy_icd10_proposals.jsonl",
            "review_queue": "data/icd10/tayy_icd10_review_queue.csv",
            "report": "data/tayy_icd10_report.json",
        },
        "counts": {
            "rows_in_scope": len(selected_rows),
            "proposals": len(proposals),
            "review_queue_rows": len(proposals),
            **dict(sorted(stats.items())),
            "candidate_count_distribution": dict(
                sorted(candidate_count_distribution.items(), key=lambda item: int(item[0]))
            ),
            "decision_reasons": dict(sorted(reason_counts.items())),
        },
        "invariants": dict(invariants),
        "method": {
            "row_fingerprint": ROW_FINGERPRINT_ALGORITHM,
            "exact": "NFKC + casefold + punctuation/whitespace normalization; accents retained",
            "catalog_exact_variants": [
                "title_vi",
                "title_vi with bracketed clarifier removed (bracket text is never an alias)",
            ],
            "candidate_retrieval": "accent-folded multi-field sparse BM25, deterministic top-5",
            "provisional_rule": (
                "Tier A only: name_exact == lead_exact == same unique code; "
                "catalog auto_single_code_eligible; no quality flag/UI/TCM/low-info; "
                "exact conflict is a hard gate"
            ),
            "warning": "Candidates and provisional codes still require expert review.",
            "review_queue_order": (
                "priority, then normalized disease name, then disease_id; "
                "all duplicate-name rows are P0_duplicate_name"
            ),
        },
    }


def _stage_text_file(path: Path, writer: Any, *, encoding: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _assert_review_queue_has_no_unimported_decisions(path: Path) -> None:
    """Refuse to overwrite reviewer edits that have not moved to the sidecar."""

    if not path.exists():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        protected = ("decision", "icd10_code", "reviewer", "reviewed_at", "review_note")
        if not set(protected).intersection(reader.fieldnames or []):
            return
        edited_ids = []
        for row in reader:
            if any((row.get(column) or "").strip() for column in protected):
                edited_ids.append((row.get("disease_id") or "?").strip())
    if edited_ids:
        preview = ", ".join(edited_ids[:12])
        suffix = f" ... (+{len(edited_ids) - 12})" if len(edited_ids) > 12 else ""
        raise InvariantError(
            f"Review queue hiện có {len(edited_ids)} dòng đã được điền ({preview}{suffix}); "
            "từ chối ghi đè. Hãy import bằng scripts/review_tayy_icd10.py hoặc lưu bản "
            "reviewed_queue riêng trước khi sinh lại queue."
        )


def write_outputs_atomically(
    proposals: Sequence[Mapping[str, Any]], report: Mapping[str, Any]
) -> None:
    _assert_review_queue_has_no_unimported_decisions(REVIEW_QUEUE_CSV)
    staged: list[tuple[Path, Path]] = []
    try:
        def write_proposals(handle: Any) -> None:
            for proposal in proposals:
                handle.write(_stable_json(proposal) + "\n")

        staged.append(
            (
                _stage_text_file(
                    PROPOSALS_JSONL,
                    write_proposals,
                    encoding="utf-8",
                ),
                PROPOSALS_JSONL,
            )
        )

        def write_review(handle: Any) -> None:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
            writer.writeheader()
            # Worklist ưu tiên xung đột/trùng tên và gom các dòng cùng tên cạnh
            # nhau để chuyên gia có thể phân xử theo nhóm. Proposal JSONL vẫn giữ
            # nguyên thứ tự disease_id nhằm làm sidecar deterministic.
            review_order = sorted(
                proposals,
                key=lambda proposal: (
                    _proposal_priority(proposal),
                    normalize_exact(str(proposal.get("name") or "")),
                    str(proposal["disease_id"]),
                ),
            )
            for proposal in review_order:
                writer.writerow(_review_row(proposal))

        staged.append(
            (
                _stage_text_file(REVIEW_QUEUE_CSV, write_review, encoding="utf-8-sig"),
                REVIEW_QUEUE_CSV,
            )
        )

        def write_report(handle: Any) -> None:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        staged.append(
            (
                _stage_text_file(REPORT_JSON, write_report, encoding="utf-8"),
                REPORT_JSON,
            )
        )
        for temp_path, destination in staged:
            os.replace(temp_path, destination)
    finally:
        for temp_path, _ in staged:
            if temp_path.exists():
                temp_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--flagged-only",
        action="store_true",
        help=f"Chỉ xử lý dòng có cờ {DUPLICATE_NAME_FLAG}",
    )
    scope.add_argument("--all", action="store_true", help="Xử lý toàn bộ TayY_clean")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Ghi atomic proposals/review queue/report; mặc định chỉ dry-run",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(CATALOG_JSONL, CATALOG_META)
    all_rows, fieldnames, source_sha = load_tayy(INPUT_CSV)
    if args.flagged_only:
        scope = "flagged-only"
        selected_rows = [
            row for row in all_rows if DUPLICATE_NAME_FLAG in row.flags
        ]
    else:
        scope = "all"
        selected_rows = all_rows
    if not selected_rows:
        raise InvariantError(f"Scope {scope} không có dòng nào")

    print(
        f"[INFO] catalog={len(catalog.records)} mã; TayY={len(all_rows)} dòng; "
        f"scope={scope}:{len(selected_rows)}"
    )
    exact_index = ExactTitleIndex(catalog.records)
    sparse_index = SparseBM25Index(catalog.records)
    proposals = [
        build_proposal(row, catalog, exact_index, sparse_index, source_sha)
        for row in selected_rows
    ]
    invariants = verify_proposals(proposals, selected_rows, catalog)
    report = build_report(
        scope=scope,
        write_requested=args.write,
        all_rows=all_rows,
        selected_rows=selected_rows,
        proposals=proposals,
        fieldnames=fieldnames,
        source_sha=source_sha,
        catalog=catalog,
        invariants=invariants,
    )
    if args.write:
        write_outputs_atomically(proposals, report)
        print(f"[OK] {PROPOSALS_JSONL}: {len(proposals)} proposals")
        print(f"[OK] {REVIEW_QUEUE_CSV}: {len(proposals)} dòng chờ duyệt")
        print(f"[OK] {REPORT_JSON}")
    else:
        print("[DRY-RUN] Không ghi file nào. Dùng --write sau khi đã xem report.")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
        return 0
    except (FileNotFoundError, InvariantError, json.JSONDecodeError) as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
