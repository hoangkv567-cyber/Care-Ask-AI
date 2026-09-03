#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kiểm định toàn bộ catalog VN-ICD10-TT06-2026 đã trích (không lấy mẫu)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.jsonl"
META = ROOT / "data" / "icd10" / "vn_icd10_tt06_2026.meta.json"
REPORT = ROOT / "data" / "icd10_vn_catalog_report.json"
CODE_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,2})?$")
EXPECTED_SCHEMA_VERSION = "vn-icd10-tt06-2026.catalog.v1"
EXPECTED_SYSTEM = "VN-ICD10-TT06-2026"
EXPECTED_SOURCE_SHA256 = "8639f5eeb77b571363dc841923095895d2498748f9bc6620f50710a6da9159e2"
EXPECTED_WHO_SHA256 = "344c571aa9aed3b9ad1c80261b7828c45cfb5fc0b1a5da64aecef354dff5b3d9"
EXPECTED_ROWS = 15844
EXPECTED_STATISTICS = {
    "asterisk": 807,
    "auto_single_code_eligible": 11266,
    "codable": 13725,
    "dagger": 111,
    "morbidity_eligible": 13715,
    "primary_eligible": 11373,
    "restriction::discouraged_primary": 221,
    "restriction::female_only": 931,
    "restriction::male_only": 143,
    "restriction::mortality_only": 12,
    "restriction::not_primary": 2427,
    "restriction::requires_more_specific": 2119,
    "source_title_en::known_pdf_text_anomaly": 1,
    "source_title_en::matches_who": 11236,
    "source_title_en::national_variant": 6,
    "source_title_en::tt06_expansion": 4601,
}
EXPECTED_RESTRICTIONS = {
    "not_primary": 2427,
    "discouraged_primary": 221,
    "requires_more_specific": 2119,
    "mortality_only": 12,
    "female_only": 931,
    "male_only": 143,
}
EXPECTED_RESTRICTION_OCCURRENCES = {
    "not_primary": 2427,
    "discouraged_primary": 221,
    "requires_more_specific": 2119,
    "mortality_only": 12,
    "female_only": 933,
    "male_only": 144,
}
EXPECTED_TITLE_STATUSES = {
    "matches_who": 11236,
    "national_variant": 6,
    "known_pdf_text_anomaly": 1,
    "tt06_expansion": 4601,
}
EXPECTED_WHO_USAGE = {"": 10804, "dagger": 66, "aster": 373}
EXPECTED_REFERENCE_RUBRICS = {
    "exclusion": 4409,
    "inclusion": 652,
    "modifierlink": 182,
    "preferred": 168,
    "coding-hint": 16,
    "note": 7,
}
EXPECTED_REFERENCE_USAGE = {"": 4580, "dagger": 538, "aster": 316}
EXPECTED_REFERENCE_CLASS = {"in brackets": 5246, "": 188}
EXPECTED_REFERENCE_COUNT = 5434
EXPECTED_REFERENCE_RESOURCE_CODES = 1656
EXPECTED_NATIONAL_VARIANTS = {
    "F38.8": ("Other specified mood [affective] disorder", "Other specified mood [affective] disorders"),
    "G51.0": ("Bell's palsy", "Bell palsy"),
    "Q77.5": ("Dystrophic [diastrophic] dysplasia", "Dystrophic dysplasia"),
    "T20.2": ("Burn of second degree [partial thickness] of head and neck", "Burn of second degree of head and neck"),
    "T20.3": ("Burn of third degree [full thickness] of head and neck", "Burn of third degree of head and neck"),
    "X88": ("Assault by carbon monoxide and other gases and vapours", "Assault by gases and vapours"),
}
EXPECTED_RECORD_KEYS = {
    "stt",
    "source_page",
    "system",
    "code",
    "code_nodot",
    "source_code_raw",
    "source_anomaly",
    "dagger",
    "asterisk",
    "dual_coding_marker",
    "chapter_number",
    "chapter_range",
    "chapter_en",
    "chapter_vi",
    "block_code",
    "block_en",
    "block_vi",
    "subdivision1_code",
    "subdivision1_en",
    "subdivision1_vi",
    "subdivision2_code",
    "subdivision2_en",
    "subdivision2_vi",
    "category_code",
    "category_en",
    "category_vi",
    "title_en",
    "source_title_en",
    "source_title_en_status",
    "source_title_en_warning",
    "who_title_en",
    "who_inclusions_en",
    "who_exclusions_en",
    "who_usage",
    "who_superclass",
    "who_references",
    "guidance_en",
    "title_vi",
    "guidance_vi",
    "restrictions",
    "codable",
    "morbidity_eligible",
    "primary_eligible",
    "primary_eligibility",
    "auto_single_code_eligible",
}
STRING_FIELDS = {
    "system",
    "code",
    "code_nodot",
    "source_code_raw",
    "chapter_number",
    "chapter_range",
    "chapter_en",
    "chapter_vi",
    "block_code",
    "block_en",
    "block_vi",
    "subdivision1_code",
    "subdivision1_en",
    "subdivision1_vi",
    "subdivision2_code",
    "subdivision2_en",
    "subdivision2_vi",
    "category_code",
    "category_en",
    "category_vi",
    "title_en",
    "source_title_en",
    "source_title_en_status",
    "who_title_en",
    "who_usage",
    "who_superclass",
    "guidance_en",
    "title_vi",
    "guidance_vi",
    "primary_eligibility",
}
BOOL_FIELDS = {
    "dagger",
    "asterisk",
    "codable",
    "morbidity_eligible",
    "primary_eligible",
    "auto_single_code_eligible",
}
EXPECTED_HIERARCHY_COUNTS = {
    "chapter": 22,
    "block": 211,
    "subdivision1": 26,
    "subdivision2": 37,
    "category": 2090,
}
EXPECTED_HIERARCHY_REPORT = {
    "chapter_en": (22, "WHO ClaML 2019", 0, 0, 0),
    "chapter_vi": (22, "TT06 complete-superstring, otherwise modal rendering", 0, 0, 0),
    "chapter_range": (22, "TT06 complete-superstring, otherwise modal rendering", 0, 0, 0),
    "block_en": (211, "WHO ClaML 2019", 30, 32, 730),
    "block_vi": (211, "TT06 complete-superstring, otherwise modal rendering", 10, 10, 71),
    "subdivision1_en": (26, "WHO ClaML 2019", 0, 0, 0),
    "subdivision1_vi": (26, "TT06 complete-superstring, otherwise modal rendering", 1, 1, 20),
    "subdivision2_en": (37, "WHO ClaML 2019", 0, 0, 0),
    "subdivision2_vi": (37, "TT06 complete-superstring, otherwise modal rendering", 1, 1, 26),
    "category_en": (2090, "WHO ClaML 2019", 41, 47, 79),
    "category_vi": (2090, "TT06 isolated three-character title_vi cell", 18, 82, 601),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent(code: str) -> str | None:
    compact = code.replace(".", "")
    if len(compact) == 3:
        return None
    if len(compact) == 4:
        return compact[:3]
    return f"{compact[:3]}.{compact[3]}"


def normalise_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("’", "'").replace("‐", "-").replace("‑", "-").replace("–", "-")
    return re.sub(r"\s+", " ", value).strip()


def source_title_without_preferred_reference(record: dict) -> tuple[str, bool]:
    """Bỏ suffix mã ghép chỉ khi nó khớp đúng các preferred Reference của ClaML."""
    title = record["source_title_en"]
    match = re.search(r"\s*\(([^()]*)\)\s*$", title)
    if match is None:
        return title, False
    preferred_targets = [
        reference["target"]
        for reference in record["who_references"]
        if reference["rubric_kind"] == "preferred"
    ]
    clean = lambda value: re.sub(r"\s+", "", value).rstrip("†*")
    suffix_targets = [clean(value) for value in match.group(1).split(",")]
    if suffix_targets != [clean(value) for value in preferred_targets]:
        return title, False
    return title[: match.start()].strip(), True


def has_reference(
    record: dict,
    *,
    rubric_kind: str,
    usage: str,
    target: str,
) -> bool:
    return any(
        reference["rubric_kind"] == rubric_kind
        and reference["usage"] == usage
        and reference["target"] == target
        for reference in record["who_references"]
    )


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    for path in (CATALOG, META, REPORT):
        if not path.exists():
            fail(f"Thiếu artifact: {path}")
    with META.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    with REPORT.open(encoding="utf-8") as handle:
        report = json.load(handle)

    report_from_meta = {key: value for key, value in meta.items() if key != "catalog_path"}
    if report_from_meta != report:
        fail("Meta không bằng report cộng duy nhất catalog_path")
    expected_metadata = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "system": EXPECTED_SYSTEM,
        "who_release": "ICD-10 2019",
        "source_page": "https://vanban.chinhphu.vn/?docid=217536&orggroupid=4&pageid=27160",
        "source_pdf": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/4/06-byt-kem.pdf",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "pages": 1271,
        "rows": EXPECTED_ROWS,
        "unique_codes": EXPECTED_ROWS,
        "who_claml_url": "https://icdcdn.who.int/icd10/claml/icd102019en.xml.zip",
        "who_claml_sha256": EXPECTED_WHO_SHA256,
    }
    for key, expected in expected_metadata.items():
        if report.get(key) != expected:
            fail(f"Metadata {key}: chờ {expected!r}, gặp {report.get(key)!r}")
    if meta.get("catalog_path") != "data/icd10/vn_icd10_tt06_2026.jsonl":
        fail(f"catalog_path không canonical: {meta.get('catalog_path')!r}")
    if not re.fullmatch(r"PyMuPDF \d+\.\d+\.\d+", str(report.get("extractor", ""))):
        fail(f"Extractor metadata lạ: {report.get('extractor')!r}")
    if report.get("statistics") != EXPECTED_STATISTICS:
        fail(f"Report statistics lệch: {report.get('statistics')!r}")
    if report.get("restriction_set_sizes") != EXPECTED_RESTRICTIONS:
        fail("Report restriction_set_sizes lệch")
    if report.get("restriction_occurrences") != EXPECTED_RESTRICTION_OCCURRENCES:
        fail("Report restriction_occurrences lệch")
    expected_crosscheck = {
        "explicit_who_category_codes": 11243,
        "explicit_who_codes_present_in_tt06": 11243,
        "who_codes_missing_from_tt06": 0,
        "tt06_expanded_modifier_codes_not_explicit_in_claml": 4601,
        "usage": {"aster": 373, "dagger": 66, "none": 10804},
        "references": EXPECTED_REFERENCE_COUNT,
        "reference_usage": {"aster": 316, "dagger": 538, "none": 4580},
        "reference_rubric_kind": EXPECTED_REFERENCE_RUBRICS,
        "references_with_resource_code": EXPECTED_REFERENCE_RESOURCE_CODES,
    }
    if report.get("who_claml_crosscheck") != expected_crosscheck:
        fail("WHO ClaML cross-check metadata lệch")
    expected_invariants = {
        "stt_contiguous_1_to_15844",
        "code_nodot_valid",
        "codes_unique",
        "english_title_nonempty",
        "vietnamese_title_nonempty",
        "restriction_codes_exist_in_catalog",
        "every_non_root_code_has_parent",
        "requires_more_specific_equals_has_children",
        "asterisk_codes_are_not_primary",
        "dagger_and_asterisk_disjoint",
        "female_and_male_sets_disjoint",
        "who_usage_matches_pdf_markers",
        "who_superclass_matches_pdf_hierarchy",
        "who_preferred_titles_crosschecked",
        "who_references_preserved",
        "hierarchy_code_titles_unique_after_canonicalization",
    }
    invariants = report.get("invariants") or {}
    if set(invariants) != expected_invariants or not all(invariants.values()):
        fail(f"Report invariants thiếu/lạ/false: {invariants!r}")
    if report.get("known_source_anomalies") != [
        {"stt": 15750, "printed": "U13/9", "canonical": "U13.9"}
    ]:
        fail("Metadata source anomaly U13/9 lệch")
    if report.get("known_pdf_text_title_anomalies") != [
        {
            "code": "Y83.2",
            "extracted": "Surgical operation with , bypass or graft",
            "canonical_who": "Surgical operation with anastomosis, bypass or graft",
        }
    ]:
        fail("Metadata PDF text anomaly Y83.2 lệch")

    hierarchy_report = report.get("hierarchy_canonicalization") or {}
    if set(hierarchy_report) != set(EXPECTED_HIERARCHY_REPORT):
        fail(
            "Hierarchy report field lệch: "
            f"{sorted(hierarchy_report)}/{sorted(EXPECTED_HIERARCHY_REPORT)}"
        )
    for field, expected in EXPECTED_HIERARCHY_REPORT.items():
        details = hierarchy_report[field]
        scalar = (
            details.get("codes"),
            details.get("canonical_source"),
            details.get("source_variant_codes"),
            details.get("source_differs_from_canonical_codes"),
            details.get("rows_normalized"),
        )
        if scalar != expected:
            fail(f"Hierarchy report {field}: chờ {expected!r}, gặp {scalar!r}")
        variants = details.get("source_variants")
        if not isinstance(variants, list):
            fail(f"Hierarchy report {field}.source_variants không phải list")
        variant_codes = []
        calculated_variant_codes = 0
        calculated_diff_codes = 0
        calculated_rows_normalized = 0
        for variant in variants:
            if not isinstance(variant, dict) or set(variant) != {
                "code",
                "canonical",
                "observed",
            }:
                fail(f"Hierarchy provenance schema lệch: {field}/{variant!r}")
            code = variant["code"]
            canonical = variant["canonical"]
            observed = variant["observed"]
            if not isinstance(code, str) or not code or not isinstance(canonical, str) or not canonical:
                fail(f"Hierarchy provenance code/canonical rỗng: {field}/{variant!r}")
            if not isinstance(observed, list) or not observed:
                fail(f"Hierarchy provenance observed rỗng: {field}/{code}")
            values = set()
            has_difference = False
            for item in observed:
                if not isinstance(item, dict) or set(item) != {"value", "rows"}:
                    fail(f"Hierarchy provenance observed schema lệch: {field}/{code}/{item!r}")
                value = item["value"]
                rows = item["rows"]
                if not isinstance(value, str) or not value or type(rows) is not int or rows <= 0:
                    fail(f"Hierarchy provenance observed value lỗi: {field}/{code}/{item!r}")
                if value in values:
                    fail(f"Hierarchy provenance observed trùng: {field}/{code}/{value!r}")
                values.add(value)
                if value != canonical:
                    has_difference = True
                    calculated_rows_normalized += rows
            variant_codes.append(code)
            calculated_variant_codes += int(len(observed) > 1)
            calculated_diff_codes += int(has_difference)
        if variant_codes != sorted(set(variant_codes)):
            fail(f"Hierarchy provenance code trùng/sai thứ tự: {field}")
        calculated = (
            calculated_variant_codes,
            calculated_diff_codes,
            calculated_rows_normalized,
        )
        if calculated != expected[2:]:
            fail(f"Hierarchy provenance summary tính lại lệch: {field}/{calculated!r}")

    records = []
    with CATALOG.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != EXPECTED_RECORD_KEYS:
                missing = EXPECTED_RECORD_KEYS - set(record)
                extra = set(record) - EXPECTED_RECORD_KEYS
                fail(f"Schema JSONL line {line_number}: thiếu={sorted(missing)}, thừa={sorted(extra)}")
            if record["stt"] != line_number:
                fail(f"STT lệch ở JSONL line {line_number}: {record['stt']}")
            records.append(record)

    if len(records) != EXPECTED_ROWS:
        fail(f"Số record: {len(records)}")
    codes = {record["code"] for record in records}
    if len(codes) != len(records):
        fail("Code không unique")

    restrictions: dict[str, set[str]] = {name: set() for name in EXPECTED_RESTRICTIONS}
    statistics = Counter()
    title_statuses = Counter()
    who_usages = Counter()
    reference_rubrics = Counter()
    reference_usages = Counter()
    reference_classes = Counter()
    reference_count = 0
    reference_resource_codes = 0
    preferred_suffix_count = 0
    who_explicit = 0
    national_variant_codes = set()
    warning_codes = set()
    hierarchy_titles: dict[str, dict[str, set[tuple[str, ...]]]] = {
        level: {} for level in EXPECTED_HIERARCHY_COUNTS
    }
    hierarchy_occurrences: dict[str, Counter[str]] = {
        level: Counter() for level in EXPECTED_HIERARCHY_COUNTS
    }
    previous_page = 0
    for record in records:
        code = record["code"]
        if type(record["stt"]) is not int or type(record["source_page"]) is not int:
            fail(f"Kiểu stt/source_page sai: {code}")
        for field in STRING_FIELDS:
            if not isinstance(record[field], str):
                fail(f"Field {field} không phải string: {code}")
        for field in BOOL_FIELDS:
            if type(record[field]) is not bool:
                fail(f"Field {field} không phải bool: {code}")
        for field in ("who_inclusions_en", "who_exclusions_en", "restrictions"):
            if not isinstance(record[field], list) or not all(
                isinstance(value, str) for value in record[field]
            ):
                fail(f"Field {field} không phải list[string]: {code}")
        if not isinstance(record["who_references"], list):
            fail(f"who_references không phải list: {code}")
        if record["system"] != EXPECTED_SYSTEM:
            fail(f"System lệch: {code}/{record['system']!r}")
        if not CODE_RE.fullmatch(code):
            fail(f"Code sai format: {code}")
        if record["code_nodot"] != code.replace(".", ""):
            fail(f"code_nodot lệch: {code}")
        raw_compact = re.sub(r"[./\s\u2020*]", "", record["source_code_raw"])
        if raw_compact != record["code_nodot"]:
            fail(f"source_code_raw lệch: {record['source_code_raw']}/{code}")
        if not record["title_en"] or not record["title_vi"]:
            fail(f"Thiếu title: {code}")
        category_plain = re.sub(r"[\s\u2020*]", "", record["category_code"])
        if category_plain != record["code_nodot"][:3]:
            fail(f"Category lệch: {code}/{record['category_code']}")
        for field in (
            "chapter_number",
            "chapter_range",
            "chapter_en",
            "chapter_vi",
            "block_code",
            "block_en",
            "block_vi",
            "category_code",
            "category_en",
            "category_vi",
            "source_title_en",
        ):
            if not record[field]:
                fail(f"Hierarchy/title field rỗng {field}: {code}")
        for prefix in ("subdivision1", "subdivision2"):
            present = [bool(record[f"{prefix}_{suffix}"]) for suffix in ("code", "en", "vi")]
            if any(present) and not all(present):
                fail(f"Triplet {prefix} không đầy đủ: {code}")
        hierarchy_items = (
            (
                "chapter",
                record["chapter_number"],
                (record["chapter_range"], record["chapter_en"], record["chapter_vi"]),
            ),
            ("block", record["block_code"], (record["block_en"], record["block_vi"])),
            (
                "subdivision1",
                record["subdivision1_code"],
                (record["subdivision1_en"], record["subdivision1_vi"]),
            ),
            (
                "subdivision2",
                record["subdivision2_code"],
                (record["subdivision2_en"], record["subdivision2_vi"]),
            ),
            ("category", category_plain, (record["category_en"], record["category_vi"])),
        )
        for level, hierarchy_code, titles in hierarchy_items:
            if not hierarchy_code:
                continue
            hierarchy_titles[level].setdefault(hierarchy_code, set()).add(titles)
            hierarchy_occurrences[level][hierarchy_code] += 1
        page = record["source_page"]
        if not 1 <= page <= 1271 or page < previous_page:
            fail(f"source_page lệch tại {code}: {page}")
        previous_page = page
        if record["dagger"] != ("†" in record["source_code_raw"]):
            fail(f"Dagger không khớp source_code_raw: {code}")
        if record["asterisk"] != ("*" in record["source_code_raw"]):
            fail(f"Asterisk không khớp source_code_raw: {code}")
        marker_count = int(record["dagger"]) + int(record["asterisk"])
        if marker_count > 1:
            fail(f"Hai dual markers: {code}")
        expected_marker = "dagger" if record["dagger"] else "asterisk" if record["asterisk"] else None
        if record["dual_coding_marker"] != expected_marker:
            fail(f"dual_coding_marker lệch: {code}")
        if len(record["restrictions"]) != len(set(record["restrictions"])):
            fail(f"Restriction trùng trong record: {code}")
        expected_restriction_order = [
            name for name in EXPECTED_RESTRICTIONS if name in record["restrictions"]
        ]
        if record["restrictions"] != expected_restriction_order:
            fail(f"Restriction lạ hoặc sai thứ tự: {code}/{record['restrictions']!r}")
        for restriction in expected_restriction_order:
            if restriction not in restrictions:
                fail(f"Restriction lạ: {restriction}")
            restrictions[restriction].add(code)

        requires_more_specific = "requires_more_specific" in record["restrictions"]
        mortality_only = "mortality_only" in record["restrictions"]
        not_primary = "not_primary" in record["restrictions"]
        discouraged = "discouraged_primary" in record["restrictions"]
        expected_codable = not requires_more_specific
        expected_morbidity = not (requires_more_specific or mortality_only)
        expected_primary = not (
            not_primary
            or discouraged
            or requires_more_specific
            or mortality_only
            or record["asterisk"]
        )
        expected_primary_status = (
            "not_usable"
            if not expected_morbidity
            else "forbidden"
            if not_primary or record["asterisk"]
            else "discouraged"
            if discouraged
            else "allowed"
        )
        expected_auto_single = bool(
            expected_primary_status == "allowed"
            and not record["dagger"]
            and not record["asterisk"]
        )
        expected_eligibility = {
            "codable": expected_codable,
            "morbidity_eligible": expected_morbidity,
            "primary_eligible": expected_primary,
            "primary_eligibility": expected_primary_status,
            "auto_single_code_eligible": expected_auto_single,
        }
        for field, expected in expected_eligibility.items():
            if record[field] != expected:
                fail(f"Eligibility {field} lệch tại {code}: {record[field]!r}/{expected!r}")

        status = record["source_title_en_status"]
        title_statuses[status] += 1
        statistics[f"source_title_en::{status}"] += 1
        if record["source_title_en_warning"] is not None:
            if not isinstance(record["source_title_en_warning"], dict):
                fail(f"source_title_en_warning sai kiểu: {code}")
            warning_codes.add(code)
        explicit = bool(record["who_title_en"])
        if explicit:
            who_explicit += 1
            if record["title_en"] != record["who_title_en"]:
                fail(f"title_en không canonical WHO: {code}")
            expected_usage = "dagger" if record["dagger"] else "aster" if record["asterisk"] else ""
            if record["who_usage"] != expected_usage:
                fail(f"marker/who_usage lệch: {code}/{record['who_usage']!r}/{expected_usage!r}")
            who_usages[record["who_usage"]] += 1
            expected_superclass = (
                record["subdivision2_code"]
                or record["subdivision1_code"]
                or record["block_code"]
                if len(record["code_nodot"]) == 3
                else parent(code)
            )
            if record["who_superclass"] != expected_superclass:
                fail(
                    f"WHO superclass lệch: {code}/{record['who_superclass']!r}/{expected_superclass!r}"
                )
        else:
            if record["title_en"] != record["source_title_en"]:
                fail(f"TT06 expansion không giữ source title: {code}")
            if any(
                (
                    record["who_usage"],
                    record["who_superclass"],
                    record["who_inclusions_en"],
                    record["who_exclusions_en"],
                    record["who_references"],
                )
            ):
                fail(f"TT06 expansion có dữ liệu WHO giả: {code}")

        source_for_compare, stripped_reference = source_title_without_preferred_reference(record)
        preferred_suffix_count += int(stripped_reference)
        title_pair = (source_for_compare, record["who_title_en"])
        if status == "matches_who":
            if not explicit or normalise_title(source_for_compare) != normalise_title(
                record["who_title_en"]
            ):
                fail(f"Status matches_who không đúng: {code}/{title_pair!r}")
        elif status == "national_variant":
            national_variant_codes.add(code)
            if title_pair != EXPECTED_NATIONAL_VARIANTS.get(code):
                fail(f"National title variant ngoài allowlist: {code}/{title_pair!r}")
        elif status == "known_pdf_text_anomaly":
            if code != "Y83.2" or title_pair != (
                "Surgical operation with , bypass or graft",
                "Surgical operation with anastomosis, bypass or graft",
            ):
                fail(f"PDF title anomaly ngoài allowlist: {code}/{title_pair!r}")
        elif status == "tt06_expansion":
            if explicit:
                fail(f"Mã WHO bị gắn nhầm tt06_expansion: {code}")
        else:
            fail(f"source_title_en_status lạ: {code}/{status!r}")

        for reference in record["who_references"]:
            if not isinstance(reference, dict) or set(reference) != {
                "rubric_kind",
                "class",
                "usage",
                "target",
                "resource_code",
            }:
                fail(f"Schema WHO Reference lệch: {code}/{reference!r}")
            if not all(isinstance(value, str) for value in reference.values()):
                fail(f"WHO Reference có field không phải string: {code}/{reference!r}")
            if not reference["target"]:
                fail(f"WHO Reference mất target: {code}/{reference!r}")
            reference_count += 1
            reference_rubrics[reference["rubric_kind"]] += 1
            reference_usages[reference["usage"]] += 1
            reference_classes[reference["class"]] += 1
            reference_resource_codes += int(bool(reference["resource_code"]))

        statistics["codable"] += record["codable"]
        statistics["morbidity_eligible"] += record["morbidity_eligible"]
        statistics["primary_eligible"] += record["primary_eligible"]
        statistics["auto_single_code_eligible"] += record["auto_single_code_eligible"]
        statistics["dagger"] += record["dagger"]
        statistics["asterisk"] += record["asterisk"]
        for restriction in expected_restriction_order:
            statistics[f"restriction::{restriction}"] += 1
        pcode = parent(code)
        if pcode is not None and pcode not in codes:
            fail(f"Thiếu parent {pcode} của {code}")

    for level, expected_count in EXPECTED_HIERARCHY_COUNTS.items():
        mappings = hierarchy_titles[level]
        if len(mappings) != expected_count:
            fail(f"Hierarchy {level} code count: chờ {expected_count}, gặp {len(mappings)}")
        conflicts = {
            hierarchy_code: titles
            for hierarchy_code, titles in mappings.items()
            if len(titles) != 1
        }
        if conflicts:
            preview = {code: sorted(titles) for code, titles in list(conflicts.items())[:10]}
            fail(f"Hierarchy {level} có code -> nhiều title: {preview!r}")
        if set(mappings) != set(hierarchy_occurrences[level]):
            fail(f"Hierarchy {level} occurrence/code set lệch")
    block_levels = (
        set(hierarchy_titles["block"]),
        set(hierarchy_titles["subdivision1"]),
        set(hierarchy_titles["subdivision2"]),
    )
    if any(block_levels[a] & block_levels[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        fail("Các level block/subdivision giao code")
    if len(set().union(*block_levels)) != 274:
        fail("Ba level block không hợp thành đúng 274 WHO block")
    category_roots = {code for code in codes if len(code) == 3}
    if set(hierarchy_titles["category"]) != category_roots:
        fail("Category hierarchy không bằng tập 2.090 mã gốc ba ký tự")

    report_field_location = {
        "chapter_range": ("chapter", 0),
        "chapter_en": ("chapter", 1),
        "chapter_vi": ("chapter", 2),
        "block_en": ("block", 0),
        "block_vi": ("block", 1),
        "subdivision1_en": ("subdivision1", 0),
        "subdivision1_vi": ("subdivision1", 1),
        "subdivision2_en": ("subdivision2", 0),
        "subdivision2_vi": ("subdivision2", 1),
        "category_en": ("category", 0),
        "category_vi": ("category", 1),
    }
    for field, details in hierarchy_report.items():
        level, tuple_index = report_field_location[field]
        if details["codes"] != len(hierarchy_titles[level]):
            fail(f"Hierarchy report/catalog code count lệch: {field}")
        for variant in details["source_variants"]:
            hierarchy_code = variant["code"]
            titles = hierarchy_titles[level].get(hierarchy_code)
            if titles is None:
                fail(f"Hierarchy provenance trỏ code ngoài catalog: {field}/{hierarchy_code}")
            canonical = next(iter(titles))[tuple_index]
            if variant["canonical"] != canonical:
                fail(
                    f"Hierarchy provenance canonical lệch catalog: "
                    f"{field}/{hierarchy_code}/{variant['canonical']!r}/{canonical!r}"
                )
            observed_rows = sum(item["rows"] for item in variant["observed"])
            if observed_rows != hierarchy_occurrences[level][hierarchy_code]:
                fail(
                    f"Hierarchy provenance không phủ đủ rows: "
                    f"{field}/{hierarchy_code}/{observed_rows}/"
                    f"{hierarchy_occurrences[level][hierarchy_code]}"
                )

    parents = {parent(code) for code in codes if parent(code)}
    if parents != restrictions["requires_more_specific"]:
        fail("requires_more_specific không bằng tập code có child")
    if not {record["code"] for record in records if record["asterisk"]} <= restrictions["not_primary"]:
        fail("Có mã asterisk không nằm trong not_primary")
    if restrictions["female_only"] & restrictions["male_only"]:
        fail("female_only giao male_only")
    if dict(statistics) != EXPECTED_STATISTICS:
        fail(f"Thống kê tính lại lệch: {dict(statistics)!r}")
    if dict(title_statuses) != EXPECTED_TITLE_STATUSES:
        fail(f"Title statuses lệch: {dict(title_statuses)!r}")
    if who_explicit != 11243 or dict(who_usages) != EXPECTED_WHO_USAGE:
        fail(f"WHO usage distribution lệch: explicit={who_explicit}, usage={dict(who_usages)!r}")
    if reference_count != EXPECTED_REFERENCE_COUNT:
        fail(f"Số WHO Reference: {reference_count}")
    if dict(reference_rubrics) != EXPECTED_REFERENCE_RUBRICS:
        fail(f"WHO Reference rubric distribution lệch: {dict(reference_rubrics)!r}")
    if dict(reference_usages) != EXPECTED_REFERENCE_USAGE:
        fail(f"WHO Reference usage distribution lệch: {dict(reference_usages)!r}")
    if dict(reference_classes) != EXPECTED_REFERENCE_CLASS:
        fail(f"WHO Reference class distribution lệch: {dict(reference_classes)!r}")
    if reference_resource_codes != EXPECTED_REFERENCE_RESOURCE_CODES:
        fail(f"WHO Reference resource_code count lệch: {reference_resource_codes}")
    if preferred_suffix_count != 158:
        fail(f"Số source title có preferred-code suffix lệch: {preferred_suffix_count}")
    if national_variant_codes != set(EXPECTED_NATIONAL_VARIANTS):
        fail(f"National variant code set lệch: {sorted(national_variant_codes)}")
    if warning_codes != {"Y83.2"}:
        fail(f"Source-title warning code set lệch: {sorted(warning_codes)}")
    for key, expected in EXPECTED_RESTRICTIONS.items():
        if len(restrictions[key]) != expected:
            fail(f"Restriction {key}: chờ {expected}, gặp {len(restrictions[key])}")

    by_code = {record["code"]: record for record in records}
    sentinels = {
        "A00": {
            "title_en": "Cholera",
            "source_title_en": "Cholera",
            "title_vi": "Bệnh tả",
            "who_superclass": "A00-A09",
        },
        "A17.0": {
            "title_en": "Tuberculous meningitis",
            "source_title_en": "Tuberculous meningitis (G01*)",
            "title_vi": "Viêm màng não do bệnh lao (G01*)",
            "dagger": True,
            "who_usage": "dagger",
            "who_superclass": "A17",
            "auto_single_code_eligible": False,
        },
        "A18.1": {
            "title_en": "Tuberculosis of genitourinary system",
            "dagger": True,
            "who_usage": "dagger",
            "who_superclass": "A18",
            "auto_single_code_eligible": False,
        },
        "A20": {
            "block_code": "A20-A28",
            "block_en": "Certain zoonotic bacterial diseases",
            "block_vi": "Bệnh nhiễm khuẩn do động vật truyền sang người",
        },
        "A50": {
            "block_code": "A50-A64",
            "block_en": "Infections with a predominantly sexual mode of transmission",
            "block_vi": "Bệnh nhiễm trùng lây truyền chủ yếu qua đường tình dục",
        },
        "A92": {
            "block_code": "A92-A99",
            "block_en": "Arthropod-borne viral fevers and viral haemorrhagic fevers",
            "block_vi": "Bệnh sốt virus và sốt xuất huyết virus do tiết túc truyền [virus arbo]",
        },
        "B90": {
            "block_code": "B90-B94",
            "block_en": "Sequelae of infectious and parasitic diseases",
            "block_vi": "Di chứng của bệnh truyền nhiễm và ký sinh trùng",
        },
        "D51": {
            "title_en": "Vitamin B12 deficiency anaemia",
            "source_title_en": "Vitamin B12 deficiency anaemia",
            "title_vi": "Thiếu máu do thiếu vitamin B12",
            "who_superclass": "D50-D53",
        },
        "D51.1": {
            "title_en": "Vitamin B12 deficiency anaemia due to selective vitamin B12 malabsorption with proteinuria",
            "source_title_en": "Vitamin B12 deficiency anaemia due to selective vitamin B12 malabsorption with proteinuria",
            "title_vi": "Thiếu vitamin B12 do giảm hấp thu chọn lọc vitamin B12 kèm protein niệu",
            "who_superclass": "D51",
        },
        "D80": {
            "block_code": "D80-D89",
            "block_en": "Certain disorders involving the immune mechanism",
            "block_vi": "Một số rối loạn liên quan đến cơ chế miễn dịch",
        },
        "F91": {
            "title_en": "Conduct disorders",
            "source_title_en": "Conduct disorders",
            "title_vi": "Rối loạn ứng xử",
            "who_superclass": "F90-F98",
        },
        "G01": {
            "title_en": "Meningitis in bacterial diseases classified elsewhere",
            "asterisk": True,
            "who_usage": "aster",
            "who_superclass": "G00-G09",
            "primary_eligibility": "forbidden",
            "auto_single_code_eligible": False,
        },
        "N74.0": {
            "title_en": "Tuberculous infection of cervix uteri",
            "source_title_en": "Tuberculous infection of cervix uteri (A18.1†)",
            "asterisk": True,
            "who_usage": "aster",
            "who_superclass": "N74",
            "primary_eligibility": "forbidden",
            "auto_single_code_eligible": False,
        },
        "N14": {
            "title_en": "Drug- and heavy-metal-induced tubulo-interstitial and tubular conditions",
            "source_title_en": "Drug- and heavy-metal-induced tubulo-interstitial and tubular conditions",
            "category_en": "Drug- and heavy-metal-induced tubulo-interstitial and tubular conditions",
        },
        "O10": {
            "block_code": "O10-O16",
            "block_en": "Oedema, proteinuria and hypertensive disorders in pregnancy, childbirth and the puerperium",
            "block_vi": "Rối loạn phù, protein niệu và tăng huyết áp trong thai kỳ, sinh đẻ và thời kỳ sau đẻ",
        },
        "V02": {
            "title_en": "Pedestrian injured in collision with two- or three-wheeled motor vehicle",
            "source_title_en": "Pedestrian injured in collision with two- or three-wheeled motor vehicle",
            "category_en": "Pedestrian injured in collision with two- or three-wheeled motor vehicle",
        },
        "V80": {
            "subdivision2_code": "V80-V89",
            "subdivision2_en": "Other land transport accidents",
            "subdivision2_vi": "Tai nạn giao thông khác trên mặt đất",
        },
        "W00": {
            "subdivision1_code": "W00-X59",
            "subdivision1_en": "Other external causes of accidental injury",
            "subdivision1_vi": "Thương tích vì tai nạn do nguyên nhân bên ngoài khác",
        },
        "I79": {
            "category_code": "I79*",
            "category_en": "Disorders of arteries, arterioles and capillaries in diseases classified elsewhere",
            "category_vi": "Rối loạn động mạch, tiểu động mạch và/hoặc mao mạch do bệnh phân loại mục khác",
        },
        "Y81": {
            "title_en": "General- and plastic-surgery devices associated with adverse incidents",
            "source_title_en": "General- and plastic-surgery devices associated with adverse incidents",
            "category_en": "General- and plastic-surgery devices associated with adverse incidents",
        },
        "Y83.2": {
            "title_en": "Surgical operation with anastomosis, bypass or graft",
            "source_title_en": "Surgical operation with , bypass or graft",
            "source_title_en_status": "known_pdf_text_anomaly",
            "title_vi": "Tai biến do phẫu thuật với khâu nối, bắc cầu hoặc ghép",
            "who_superclass": "Y83",
        },
        "Z30": {
            "block_code": "Z30-Z39",
            "block_en": "Persons encountering health services in circumstances related to reproduction",
            "block_vi": "Những người tiếp cận dịch vụ y tế trong hoàn cảnh liên quan đến sinh sản",
        },
    }
    for code, expected_fields in sentinels.items():
        record = by_code.get(code)
        if record is None:
            fail(f"Thiếu sentinel: {code}")
        for field, expected in expected_fields.items():
            if record[field] != expected:
                fail(f"Sentinel {code}.{field}: {record[field]!r}/{expected!r}")
    if by_code["Y83.2"]["source_title_en_warning"] != {
        "field": "source_title_en",
        "extracted": "Surgical operation with , bypass or graft",
        "canonical_who": "Surgical operation with anastomosis, bypass or graft",
    }:
        fail("Không lưu đúng warning Y83.2")
    reference_sentinels = (
        ("A17.0", "preferred", "aster", "G01"),
        ("G01", "inclusion", "dagger", "A17.0"),
        ("N74.0", "preferred", "dagger", "A18.1"),
        ("A18.1", "inclusion", "aster", "N74.0"),
    )
    for code, rubric_kind, usage, target in reference_sentinels:
        if not has_reference(
            by_code[code], rubric_kind=rubric_kind, usage=usage, target=target
        ):
            fail(f"Thiếu reference sentinel {code} -> {target} ({rubric_kind}/{usage})")
    if by_code["U13.9"]["source_anomaly"] != {
        "printed": "U13/9",
        "canonical": "U13.9",
    }:
        fail("Không lưu đúng anomaly U13/9")

    catalog_hash = sha256(CATALOG)
    if meta.get("catalog_sha256") != catalog_hash or report.get("catalog_sha256") != catalog_hash:
        fail("Catalog SHA không khớp meta/report")

    print(
        "[OK] VN-ICD10-TT06-2026: "
        f"{len(records)} mã; {statistics['codable']} leaf; "
        f"{who_explicit} category đối chiếu WHO; {reference_count} references; "
        "full-corpus PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
