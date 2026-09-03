#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dựng catalog ICD-10 Việt Nam từ phụ lục chính thức của TT 06/2026/TT-BYT.

Nguồn đích được khóa phiên bản:
  * Thông tư 06/2026/TT-BYT, ban hành 2026-04-02, hiệu lực 2026-07-01.
  * Phụ lục 1.271 trang, 15.844 dòng, dựa trên WHO ICD-10 2019.

PDF là bảng 29 cột có lớp chữ. Script dùng đường kẻ của bảng trên trang đầu để
xác định biên cột, sau đó đọc mọi từ theo tọa độ. Cách này nhanh hơn gọi bộ nhận
dạng bảng cho 1.271 trang và không dựa vào khoảng trắng của ``extract_text()``.

PDF/catalog lớn được giữ ngoài git trong ``data/icd10/``; report nhỏ có checksum,
thống kê và invariant được ghi ra ``data/icd10_vn_catalog_report.json``.

Chạy:
    python scripts/build_icd10_vn_catalog.py --download
    python scripts/build_icd10_vn_catalog.py --source-pdf path/to/06-byt-kem.pdf
    python scripts/build_icd10_vn_catalog.py --verify-only
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - lỗi môi trường, không phải logic
    raise SystemExit(
        "Thiếu PyMuPDF. Cài dependencies bằng: pip install -r requirements.txt"
    ) from exc


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "icd10"
DEFAULT_PDF = DATA_DIR / "06-byt-kem.pdf"
DEFAULT_WHO_CLAML = DATA_DIR / "icd102019en.xml.zip"
DEFAULT_CATALOG = DATA_DIR / "vn_icd10_tt06_2026.jsonl"
DEFAULT_META = DATA_DIR / "vn_icd10_tt06_2026.meta.json"
DEFAULT_REPORT = ROOT / "data" / "icd10_vn_catalog_report.json"

SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/4/06-byt-kem.pdf"
SOURCE_PAGE = "https://vanban.chinhphu.vn/?docid=217536&orggroupid=4&pageid=27160"
SOURCE_SHA256 = "8639f5eeb77b571363dc841923095895d2498748f9bc6620f50710a6da9159e2"
WHO_CLAML_URL = "https://icdcdn.who.int/icd10/claml/icd102019en.xml.zip"
WHO_CLAML_SHA256 = "344c571aa9aed3b9ad1c80261b7828c45cfb5fc0b1a5da64aecef354dff5b3d9"
EXPECTED_PAGES = 1271
EXPECTED_ROWS = 15844
SYSTEM_ID = "VN-ICD10-TT06-2026"
SCHEMA_VERSION = "vn-icd10-tt06-2026.catalog.v1"
WHO_RELEASE = "ICD-10 2019"
# Lỗi in đã xác nhận trực tiếp trong PDF: STT 15750 ghi U13/9 ở cột có dấu,
# trong khi cột không dấu, tên Anh và tên Việt đều xác nhận U13.9.
KNOWN_SOURCE_CODE_ANOMALIES = {15750: {"printed": "U13/9", "canonical": "U13.9"}}
NATIONAL_EN_TITLE_VARIANTS = {
    "F38.8": ("Other specified mood [affective] disorder", "Other specified mood [affective] disorders"),
    "G51.0": ("Bell's palsy", "Bell palsy"),
    "Q77.5": ("Dystrophic [diastrophic] dysplasia", "Dystrophic dysplasia"),
    "T20.2": ("Burn of second degree [partial thickness] of head and neck", "Burn of second degree of head and neck"),
    "T20.3": ("Burn of third degree [full thickness] of head and neck", "Burn of third degree of head and neck"),
    "X88": ("Assault by carbon monoxide and other gases and vapours", "Assault by gases and vapours"),
}
KNOWN_PDF_TEXT_TITLE_ANOMALIES = {
    "Y83.2": (
        "Surgical operation with , bypass or graft",
        "Surgical operation with anastomosis, bypass or graft",
    )
}

FIELD_NAMES = [
    "stt",
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
    "code_pdf",
    "code_nodot",
    "title_en",
    "guidance_en",
    "title_vi",
    "guidance_vi",
    "not_primary",
    "discouraged_primary",
    "requires_more_specific",
    "mortality_only",
    "female_only",
    "male_only",
]
FIELD_INDEX = {name: index for index, name in enumerate(FIELD_NAMES)}

FLAG_NAMES = FIELD_NAMES[23:]
CODE_NODOT_RE = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]{0,2}$")
PLAIN_CODE_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,2})?$")
FLAG_CODE_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,2})?[\u2020*]?$")
ROW_NUMBER_RE = re.compile(r"^[0-9]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_cell(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalise_title_compare(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = value.replace("’", "'").replace("‐", "-").replace("‑", "-").replace("–", "-")
    # ClaML represents a few chemical/vitamin subscripts as inline elements.
    value = re.sub(r"\b([a-z]{1,3})\s+(\d+)\b", r"\1\2", value)
    value = re.sub(r"(?<=\d)\s+-\s*(?=[a-z])", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalise_title_layout(value: str) -> str:
    """Khóa bảo toàn ký tự khi lớp text PDF bẻ một từ qua hai dòng."""
    return re.sub(r"\s+", "", _normalise_title_compare(value))


def _source_title_without_trailing_reference(title: str, who_record: dict) -> str:
    preferred_references = [
        reference
        for reference in who_record.get("references", [])
        if reference.get("rubric_kind") == "preferred"
    ]
    if not preferred_references:
        return title
    match = re.search(r"\s*\(([^)]*[A-Z][0-9]{2}[^)]*)\)\s*$", title)
    if not match:
        return title
    group = _normalise_title_compare(match.group(1).replace("†", "").replace("*", ""))
    targets = {
        _normalise_title_compare(reference.get("target", ""))
        for reference in preferred_references
        if reference.get("target")
    }
    if not any(target in group for target in targets):
        return title
    return title[: match.start()].strip()


def _plain_code(code_nodot: str) -> str:
    if not CODE_NODOT_RE.fullmatch(code_nodot):
        raise ValueError(f"Mã không dấu không hợp lệ: {code_nodot!r}")
    return code_nodot if len(code_nodot) == 3 else f"{code_nodot[:3]}.{code_nodot[3:]}"


def _parent_code(code: str) -> str | None:
    compact = code.replace(".", "")
    if len(compact) == 3:
        return None
    if len(compact) == 4:
        return compact[:3]
    return f"{compact[:3]}.{compact[3]}"


def _label_without_references(label: ET.Element) -> str:
    """Lấy prose của Label, bỏ Reference nhưng giữ đúng khoảng cách inline.

    ClaML dùng phần tử inline cho chỉ số dưới (``B<Term>12</Term>``) và lồng
    ``Reference`` bên trong ``Fragment``. Nối mỗi node bằng một dấu cách sẽ làm
    hỏng ``B12``; dùng ``itertext`` trực tiếp lại làm mã tham chiếu dính vào tên.
    """

    def walk(node: ET.Element) -> str:
        chunks = [node.text or ""]
        for child in node:
            if child.tag != "Reference":
                chunks.append(walk(child))
            chunks.append(child.tail or "")
        return "".join(chunks)

    return _normalise_cell(walk(label))


def _label_references(label: ET.Element) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for reference in label.iter("Reference"):
        target = _normalise_cell("".join(reference.itertext()))
        usage = reference.get("usage") or ""
        if not target or usage not in {"", "dagger", "aster"}:
            raise RuntimeError(
                f"WHO ClaML Reference lỗi: target={target!r}, usage={usage!r}"
            )
        references.append(
            {
                "class": reference.get("class") or "",
                "usage": usage,
                "target": target,
                "resource_code": reference.get("code") or "",
            }
        )
    return references


def parse_who_claml(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    with zipfile.ZipFile(path) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if len(xml_names) != 1:
            raise RuntimeError(f"WHO ClaML zip cần đúng 1 XML, thấy {xml_names}")
        root = ET.fromstring(archive.read(xml_names[0]))
    records: dict[str, dict] = {}
    hierarchy: dict[str, dict] = {}
    for node in root.findall("Class"):
        kind = node.get("kind") or ""
        code = node.get("code") or ""
        if kind in {"chapter", "block"}:
            preferred = ""
            for rubric in node.findall("Rubric"):
                if rubric.get("kind") != "preferred":
                    continue
                label = rubric.find("Label")
                if label is not None:
                    preferred = _label_without_references(label)
            if not code or not preferred or code in hierarchy:
                raise RuntimeError(
                    f"WHO ClaML hierarchy lỗi: kind={kind!r}, code={code!r}, "
                    f"title={preferred!r}"
                )
            hierarchy[code] = {"kind": kind, "title": preferred}
            continue
        if kind != "category":
            continue
        usage = node.get("usage") or ""
        if usage not in {"", "dagger", "aster"}:
            raise RuntimeError(f"WHO ClaML usage lạ tại {code}: {usage!r}")
        superclasses = [parent.get("code") or "" for parent in node.findall("SuperClass")]
        if len(superclasses) != 1 or not superclasses[0]:
            raise RuntimeError(f"WHO ClaML superclass lỗi tại {code}: {superclasses!r}")
        preferred = ""
        inclusions: list[str] = []
        exclusions: list[str] = []
        references: list[dict[str, str]] = []
        for rubric in node.findall("Rubric"):
            kind = rubric.get("kind")
            label = rubric.find("Label")
            if label is None:
                continue
            for reference in _label_references(label):
                references.append({"rubric_kind": kind or "", **reference})
            text = _label_without_references(label)
            if not text:
                continue
            if kind == "preferred":
                preferred = text
            elif kind == "inclusion":
                inclusions.append(text)
            elif kind == "exclusion":
                exclusions.append(text)
        if not PLAIN_CODE_RE.fullmatch(code) or not preferred:
            raise RuntimeError(f"WHO ClaML category lỗi: code={code!r}, title={preferred!r}")
        if code in records:
            raise RuntimeError(f"WHO ClaML trùng code {code}")
        records[code] = {
            "title": preferred,
            "inclusions": inclusions,
            "exclusions": exclusions,
            "usage": usage,
            "superclass": superclasses[0],
            "references": references,
        }
    if len(records) != 11243:
        raise RuntimeError(f"WHO ClaML cần 11.243 category tường minh, gặp {len(records)}")
    hierarchy_counts = Counter(record["kind"] for record in hierarchy.values())
    if hierarchy_counts != Counter({"block": 274, "chapter": 22}):
        raise RuntimeError(f"WHO ClaML hierarchy count lệch: {dict(hierarchy_counts)}")
    return records, hierarchy


def _column_edges(first_page: fitz.Page) -> list[float]:
    """Đọc 30 đường biên x của bảng 29 cột từ trang đầu."""
    tables = first_page.find_tables().tables
    matches = [table for table in tables if table.col_count == 29]
    if len(matches) != 1:
        raise RuntimeError(f"Cần đúng 1 bảng 29 cột ở trang đầu, thấy {len(matches)}")
    table = matches[0]
    data = table.extract()
    row_index = next(
        (i for i, row in enumerate(data) if row and row[0] == "1" and row[17] == "A00"),
        None,
    )
    if row_index is None:
        raise RuntimeError("Không tìm thấy dòng ICD đầu tiên (STT=1, code=A00)")
    cells = table.rows[row_index].cells
    if len(cells) != 29 or any(cell is None for cell in cells):
        raise RuntimeError("Biên bảng trang đầu không đầy đủ 29 cột")
    edges = [float(cells[0][0])] + [float(cell[2]) for cell in cells]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise RuntimeError("Biên cột ICD không tăng nghiêm ngặt")
    return edges


Char = tuple[float, float, float, float, str, tuple[int, int]]


def _page_chars(page: fitz.Page) -> list[Char]:
    chars: list[Char] = []
    raw = page.get_text("rawdict", sort=True)
    for block_index, block in enumerate(raw.get("blocks", [])):
        for line_index, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    x0, y0, x1, y1 = map(float, char["bbox"])
                    chars.append(
                        (x0, y0, x1, y1, str(char["c"]), (block_index, line_index))
                    )
    return chars


def _join_chars(
    chars: list[Char],
    _cell_left: float,
    _cell_right: float,
) -> str:
    """Ghép ký tự trong một cell mà không để từ tràn sang cell kế bên."""
    if not chars:
        return ""
    # Superscript/subscript glyphs have a different y0 although PyMuPDF correctly
    # assigns them to the same rawdict line. Grouping on rounded y0 reorders text
    # such as ``B12`` into ``B deficiency 12``; retain the source line identity.
    lines: dict[tuple[int, int], list[Char]] = {}
    for char in chars:
        lines.setdefault(char[5], []).append(char)
    pieces: list[tuple[str, float, float, float, tuple[int, int], bool, bool]] = []
    for line_id, line_chars in lines.items():
        line = sorted(line_chars, key=lambda char: char[0])
        raw_text = "".join(char[4] for char in line)
        text = raw_text.strip()
        if text:
            pieces.append(
                (
                    text,
                    min(char[0] for char in line),
                    max(char[2] for char in line),
                    min((char[1] + char[3]) / 2.0 for char in line),
                    line_id,
                    bool(raw_text[:1].isspace()),
                    bool(raw_text[-1:].isspace()),
                )
            )
    if not pieces:
        return ""
    pieces.sort(key=lambda piece: (piece[3], piece[4]))
    # Lớp chữ PDF giữ một ký tự trắng cuối dòng khi wrap xảy ra giữa hai từ, nhưng
    # không có ký tự trắng khi một từ bị bẻ đôi. Không được ``strip`` mất tín hiệu
    # này: A04.3 là ``Enterohaemorrhagi`` + ``c ...``, còn A04.8 là
    # ``bacterial intestinal `` + ``infections``.
    merged = pieces[0][0]
    for previous, current in zip(pieces, pieces[1:]):
        separator = " " if previous[6] or current[5] else ""
        merged += separator + current[0]
    return _normalise_cell(merged)


def _page_rows(
    page: fitz.Page,
    edges: list[float],
    expected_stt: int,
) -> tuple[list[list[str]], int]:
    words = page.get_text("words", sort=True)
    # Dòng dữ liệu căn trái sát đường biên đầu (x ~= 19.2). Dòng header số thứ tự
    # ở trang 1 được căn giữa (x ~= 24.2), nên không lọt điều kiện này.
    marker_max_x = edges[0] + 4.0
    markers = [
        word
        for word in words
        if edges[0] <= float(word[0]) <= marker_max_x
        and ROW_NUMBER_RE.fullmatch(str(word[4]))
    ]
    markers.sort(key=lambda word: float(word[1]))
    chars = _page_chars(page)

    rows: list[list[str]] = []
    for index, marker in enumerate(markers):
        stt = int(marker[4])
        if stt != expected_stt:
            raise RuntimeError(
                f"STT đứt chuỗi ở trang {page.number + 1}: "
                f"chờ {expected_stt}, gặp {stt}"
            )
        top = float(marker[1]) - 0.75
        bottom = (
            float(markers[index + 1][1]) - 0.75
            if index + 1 < len(markers)
            else float(page.rect.height) - 1.0
        )
        cells: list[list[Char]] = [[] for _ in range(29)]
        for char in chars:
            center_y = (char[1] + char[3]) / 2.0
            if center_y < top or center_y >= bottom:
                continue
            center_x = (char[0] + char[2]) / 2.0
            col = bisect.bisect_right(edges, center_x) - 1
            if 0 <= col < 29:
                cells[col].append(char)
        row = [
            _join_chars(cell, edges[col], edges[col + 1])
            for col, cell in enumerate(cells)
        ]
        if row[0] != str(stt):
            raise RuntimeError(
                f"Không tái tạo được STT {stt} ở trang {page.number + 1}: {row[0]!r}"
            )
        rows.append(row)
        expected_stt += 1
    return rows, expected_stt


def _flag_code(raw: str) -> str:
    """Chuẩn hóa một mã trong sáu cột danh sách hạn chế.

    Sáu cột 24-29 là sáu *danh sách mã độc lập*, không phải sáu boolean cùng
    hàng. Một số ô còn được gộp dọc trong PDF. Vì vậy phải gom toàn cột thành
    tập mã rồi mới join theo code; không được gắn cờ theo vị trí hàng.
    """
    compact = re.sub(r"\s+", "", raw or "")
    if not FLAG_CODE_RE.fullmatch(compact):
        raise ValueError(f"Mã trong cột hạn chế không hợp lệ: {raw!r}")
    plain = compact.rstrip("\u2020*")
    if not PLAIN_CODE_RE.fullmatch(plain):
        raise ValueError(f"Mã hạn chế sau chuẩn hóa không hợp lệ: {raw!r}")
    return plain


def _most_frequent_complete(values: Counter[str], *, label: str) -> str:
    nonempty = [value for value in values if value]
    if not nonempty:
        raise RuntimeError(f"Không có giá trị hierarchy để canonicalize: {label}")
    # If one observed rendering contains every other rendering, it is the unique
    # complete form even when a page break makes the truncated prefix slightly
    # more frequent (A20-A28 is 28 truncated vs 27 complete rows). Only fall back
    # to the modal rendering for genuine non-containment wording conflicts.
    complete = [
        candidate
        for candidate in nonempty
        if all(other in candidate for other in nonempty)
    ]
    if complete:
        return max(complete, key=lambda value: (len(value), values[value], value))
    return max(nonempty, key=lambda value: (values[value], len(value), value))


def canonicalize_hierarchy(
    raw_rows: list[tuple[int, list[str]]],
    who_records: dict[str, dict],
    who_hierarchy: dict[str, dict],
) -> dict[str, dict]:
    """Canonicalize merged hierarchy cells before creating catalog records.

    The PDF vertically merges chapter/block/category cells. Row y-slicing can
    therefore capture a truncated label or, at a boundary, a neighbour's label.
    Disease titles are isolated cells and are unaffected. English hierarchy
    labels come from pinned WHO ClaML; Vietnamese range labels use the modal
    official PDF rendering, while category VI comes from its own three-character
    category row (the isolated disease-title cell).
    """

    def value(row: list[str], field: str) -> str:
        return row[FIELD_INDEX[field]]

    def category_key(row: list[str]) -> str:
        return re.sub(r"[\s†*]", "", value(row, "category_code"))

    def direct_key(field: str):
        return lambda row: value(row, field)

    def counters(key_fn, field: str) -> dict[str, Counter[str]]:
        grouped: dict[str, Counter[str]] = {}
        for _page, row in raw_rows:
            key = key_fn(row)
            if key:
                grouped.setdefault(key, Counter())[value(row, field)] += 1
        return grouped

    report: dict[str, dict] = {}

    def apply_field(
        report_key: str,
        key_fn,
        field: str,
        canonical: dict[str, str],
        canonical_source: str,
    ) -> None:
        observed = counters(key_fn, field)
        if set(observed) != set(canonical):
            missing = sorted(set(observed) - set(canonical))
            extra = sorted(set(canonical) - set(observed))
            raise RuntimeError(
                f"Hierarchy {report_key} key lệch: thiếu canonical={missing[:20]}, "
                f"thừa canonical={extra[:20]}"
            )
        changed = 0
        for _page, row in raw_rows:
            key = key_fn(row)
            if not key:
                continue
            index = FIELD_INDEX[field]
            if row[index] != canonical[key]:
                row[index] = canonical[key]
                changed += 1
        variants = []
        for key in sorted(observed):
            source_values = observed[key]
            if len(source_values) == 1 and next(iter(source_values)) == canonical[key]:
                continue
            variants.append(
                {
                    "code": key,
                    "canonical": canonical[key],
                    "observed": [
                        {"value": text, "rows": count}
                        for text, count in sorted(
                            source_values.items(),
                            key=lambda item: (-item[1], -len(item[0]), item[0]),
                        )
                    ],
                }
            )
        report[report_key] = {
            "field": field,
            "codes": len(canonical),
            "canonical_source": canonical_source,
            "source_variant_codes": sum(len(items) > 1 for items in observed.values()),
            "source_differs_from_canonical_codes": len(variants),
            "rows_normalized": changed,
            "source_variants": variants,
        }

    chapter_keys = {
        value(row, "chapter_number") for _page, row in raw_rows if value(row, "chapter_number")
    }
    chapter_en = {
        code: who_hierarchy[code]["title"]
        for code in chapter_keys
        if code in who_hierarchy and who_hierarchy[code]["kind"] == "chapter"
    }
    if set(chapter_en) != chapter_keys:
        raise RuntimeError(f"WHO thiếu chapter: {sorted(chapter_keys - set(chapter_en))}")
    chapter_vi_observed = counters(direct_key("chapter_number"), "chapter_vi")
    chapter_vi = {
        code: _most_frequent_complete(items, label=f"chapter_vi/{code}")
        for code, items in chapter_vi_observed.items()
    }
    chapter_range_observed = counters(direct_key("chapter_number"), "chapter_range")
    chapter_range = {
        code: _most_frequent_complete(items, label=f"chapter_range/{code}")
        for code, items in chapter_range_observed.items()
    }
    apply_field(
        "chapter_en", direct_key("chapter_number"), "chapter_en", chapter_en, "WHO ClaML 2019"
    )
    apply_field(
        "chapter_vi",
        direct_key("chapter_number"),
        "chapter_vi",
        chapter_vi,
        "TT06 complete-superstring, otherwise modal rendering",
    )
    apply_field(
        "chapter_range",
        direct_key("chapter_number"),
        "chapter_range",
        chapter_range,
        "TT06 complete-superstring, otherwise modal rendering",
    )

    for level in ("block", "subdivision1", "subdivision2"):
        code_field = f"{level}_code"
        en_field = f"{level}_en"
        vi_field = f"{level}_vi"
        keys = {value(row, code_field) for _page, row in raw_rows if value(row, code_field)}
        canonical_en = {
            code: who_hierarchy[code]["title"]
            for code in keys
            if code in who_hierarchy and who_hierarchy[code]["kind"] == "block"
        }
        if set(canonical_en) != keys:
            raise RuntimeError(
                f"WHO thiếu {level}: {sorted(keys - set(canonical_en))[:20]}"
            )
        observed_vi = counters(direct_key(code_field), vi_field)
        canonical_vi = {
            code: _most_frequent_complete(items, label=f"{vi_field}/{code}")
            for code, items in observed_vi.items()
        }
        apply_field(
            en_field, direct_key(code_field), en_field, canonical_en, "WHO ClaML 2019"
        )
        apply_field(
            vi_field,
            direct_key(code_field),
            vi_field,
            canonical_vi,
            "TT06 complete-superstring, otherwise modal rendering",
        )

    category_keys = {category_key(row) for _page, row in raw_rows}
    if "" in category_keys:
        raise RuntimeError("Có dòng thiếu category_code")
    category_en = {
        code: who_records[code]["title"] for code in category_keys if code in who_records
    }
    if set(category_en) != category_keys:
        raise RuntimeError(
            f"WHO thiếu category hierarchy: {sorted(category_keys - set(category_en))[:20]}"
        )
    category_vi: dict[str, str] = {}
    for _page, row in raw_rows:
        code_nodot = re.sub(r"\s+", "", value(row, "code_nodot"))
        if len(code_nodot) != 3:
            continue
        code = _plain_code(code_nodot)
        title_vi = value(row, "title_vi")
        if code in category_vi or not title_vi:
            raise RuntimeError(f"Category VI source row lỗi/trùng: {code}")
        category_vi[code] = title_vi
    if set(category_vi) != category_keys:
        raise RuntimeError(
            f"Thiếu category VI source row: {sorted(category_keys - set(category_vi))[:20]}"
        )
    apply_field("category_en", category_key, "category_en", category_en, "WHO ClaML 2019")
    apply_field(
        "category_vi",
        category_key,
        "category_vi",
        category_vi,
        "TT06 isolated three-character title_vi cell",
    )

    # Hard postcondition: one code has exactly one canonical EN/VI label at every
    # hierarchy level. This is checked over all 15,844 output rows, not sampled.
    for level in ("block", "subdivision1", "subdivision2"):
        code_field = f"{level}_code"
        for language in ("en", "vi"):
            grouped = counters(direct_key(code_field), f"{level}_{language}")
            if any(len(items) != 1 or "" in items for items in grouped.values()):
                raise RuntimeError(f"Hierarchy {level}_{language} chưa canonical duy nhất")
    for language in ("en", "vi"):
        grouped = counters(category_key, f"category_{language}")
        if any(len(items) != 1 or "" in items for items in grouped.values()):
            raise RuntimeError(f"Hierarchy category_{language} chưa canonical duy nhất")
    return report


def parse_pdf(
    path: Path,
    who_records: dict[str, dict],
    who_hierarchy: dict[str, dict],
) -> tuple[list[dict], dict]:
    document = fitz.open(path)
    if len(document) != EXPECTED_PAGES:
        raise RuntimeError(f"Số trang sai: chờ {EXPECTED_PAGES}, gặp {len(document)}")
    edges = _column_edges(document[0])

    raw_rows: list[tuple[int, list[str]]] = []
    expected_stt = 1
    for page in document:
        page_rows, expected_stt = _page_rows(page, edges, expected_stt)
        raw_rows.extend((page.number + 1, row) for row in page_rows)
    document.close()

    if len(raw_rows) != EXPECTED_ROWS or expected_stt != EXPECTED_ROWS + 1:
        raise RuntimeError(
            f"Số dòng sai: chờ {EXPECTED_ROWS}, gặp {len(raw_rows)}; "
            f"STT kế tiếp={expected_stt}"
        )

    hierarchy_report = canonicalize_hierarchy(raw_rows, who_records, who_hierarchy)

    all_codes = {
        _plain_code(re.sub(r"\s+", "", row[18])) for _page_number, row in raw_rows
    }
    restriction_sets: dict[str, set[str]] = {}
    restriction_occurrences: dict[str, int] = {}
    for column, name in enumerate(FLAG_NAMES, 23):
        restriction_sets[name] = {
            _flag_code(row[column])
            for _page_number, row in raw_rows
            if row[column].strip()
        }
        restriction_occurrences[name] = sum(
            1 for _page_number, row in raw_rows if row[column].strip()
        )
        missing = restriction_sets[name] - all_codes
        if missing:
            raise RuntimeError(
                f"Cột hạn chế {name} chứa mã ngoài catalog: {sorted(missing)[:20]}"
            )

    records: list[dict] = []
    seen_codes: set[str] = set()
    marker_codes = {"dagger": set(), "asterisk": set()}
    anomalies: list[str] = []
    stats = Counter()
    for source_page, raw in raw_rows:
        values = dict(zip(FIELD_NAMES, raw))
        stt = int(values["stt"])
        code_nodot = re.sub(r"\s+", "", values["code_nodot"])
        code = _plain_code(code_nodot)
        code_pdf = re.sub(r"\s+", "", values["code_pdf"])
        if code in seen_codes:
            anomalies.append(f"code trùng {code} tại STT {stt}")
        seen_codes.add(code)
        raw_nodot = re.sub(r"[./\s\u2020*]", "", code_pdf)
        if raw_nodot != code_nodot:
            anomalies.append(
                f"code raw/no-dot lệch tại STT {stt}: {code_pdf!r}/{code_nodot!r}"
            )
        if code not in code_pdf:
            known = KNOWN_SOURCE_CODE_ANOMALIES.get(stt)
            if known != {"printed": code_pdf, "canonical": code}:
                anomalies.append(f"code PDF lệch tại STT {stt}: {code_pdf!r} != {code!r}")
        if not values["title_en"] or not values["title_vi"]:
            anomalies.append(f"thiếu title tại STT {stt}/{code}")
        dagger = "\u2020" in code_pdf
        asterisk = "*" in code_pdf
        category_plain = re.sub(r"[\s\u2020*]", "", values["category_code"])
        if category_plain != code_nodot[:3]:
            anomalies.append(
                f"category lệch tại STT {stt}: {category_plain!r}/{code_nodot!r}"
            )
        if int(dagger) + int(asterisk) > 1:
            anomalies.append(f"code có cả dagger và asterisk tại STT {stt}: {code_pdf!r}")

        who_record = who_records.get(code, {})
        who_usage = who_record.get("usage", "")
        expected_usage = "dagger" if dagger else "aster" if asterisk else ""
        if who_record and who_usage != expected_usage:
            anomalies.append(
                f"marker WHO/PDF lệch tại {code}: {who_usage!r}/{expected_usage!r}"
            )
        if who_record:
            if len(code_nodot) == 3:
                expected_superclass = (
                    values["subdivision2_code"]
                    or values["subdivision1_code"]
                    or values["block_code"]
                )
            else:
                expected_superclass = _parent_code(code) or ""
            if who_record.get("superclass") != expected_superclass:
                anomalies.append(
                    "superclass WHO/PDF lệch tại "
                    f"{code}: {who_record.get('superclass')!r}/{expected_superclass!r}"
                )

        source_title_status = "tt06_expansion"
        source_title_warning = None
        if who_record:
            source_for_compare = _source_title_without_trailing_reference(
                values["title_en"], who_record
            )
            title_pair = (source_for_compare, who_record["title"])
            if _normalise_title_compare(source_for_compare) == _normalise_title_compare(
                who_record["title"]
            ):
                source_title_status = "matches_who"
            elif _normalise_title_layout(source_for_compare) == _normalise_title_layout(
                who_record["title"]
            ):
                source_title_status = "pdf_linewrap_spacing"
            elif title_pair == NATIONAL_EN_TITLE_VARIANTS.get(code):
                source_title_status = "national_variant"
            elif title_pair == KNOWN_PDF_TEXT_TITLE_ANOMALIES.get(code):
                source_title_status = "known_pdf_text_anomaly"
                source_title_warning = {
                    "field": "source_title_en",
                    "extracted": source_for_compare,
                    "canonical_who": who_record["title"],
                }
            else:
                anomalies.append(
                    f"title EN WHO/PDF lệch tại {code}: "
                    f"{source_for_compare!r}/{who_record['title']!r}"
                )

        flag_values = {name: code in restriction_sets[name] for name in FLAG_NAMES}
        if dagger:
            marker_codes["dagger"].add(code)
        if asterisk:
            marker_codes["asterisk"].add(code)
        restrictions = [name for name, enabled in flag_values.items() if enabled]
        record = {
            "stt": stt,
            "source_page": source_page,
            "system": SYSTEM_ID,
            "code": code,
            "code_nodot": code_nodot,
            "source_code_raw": code_pdf,
            "source_anomaly": KNOWN_SOURCE_CODE_ANOMALIES.get(stt),
            "dagger": dagger,
            "asterisk": asterisk,
            "dual_coding_marker": "dagger" if dagger else "asterisk" if asterisk else None,
            "chapter_number": values["chapter_number"],
            "chapter_range": values["chapter_range"],
            "chapter_en": values["chapter_en"],
            "chapter_vi": values["chapter_vi"],
            "block_code": values["block_code"],
            "block_en": values["block_en"],
            "block_vi": values["block_vi"],
            "subdivision1_code": values["subdivision1_code"],
            "subdivision1_en": values["subdivision1_en"],
            "subdivision1_vi": values["subdivision1_vi"],
            "subdivision2_code": values["subdivision2_code"],
            "subdivision2_en": values["subdivision2_en"],
            "subdivision2_vi": values["subdivision2_vi"],
            "category_code": values["category_code"],
            "category_en": values["category_en"],
            "category_vi": values["category_vi"],
            # Với 11.243 category WHO tường minh, nhãn ClaML là nhãn Anh máy đọc
            # canonical. Vẫn giữ nguyên nhãn trích từ phụ lục để audit các biến thể
            # quốc gia và lỗi lớp chữ của PDF (ví dụ Y83.2).
            "title_en": who_record.get("title") or values["title_en"],
            "source_title_en": values["title_en"],
            "source_title_en_status": source_title_status,
            "source_title_en_warning": source_title_warning,
            "who_title_en": who_record.get("title", ""),
            "who_inclusions_en": who_record.get("inclusions", []),
            "who_exclusions_en": who_record.get("exclusions", []),
            "who_usage": who_usage,
            "who_superclass": who_record.get("superclass", ""),
            "who_references": who_record.get("references", []),
            "guidance_en": values["guidance_en"],
            "title_vi": values["title_vi"],
            "guidance_vi": values["guidance_vi"],
            "restrictions": restrictions,
            "codable": not flag_values["requires_more_specific"],
            "morbidity_eligible": not (
                flag_values["requires_more_specific"] or flag_values["mortality_only"]
            ),
            "primary_eligible": not (
                flag_values["not_primary"]
                or flag_values["discouraged_primary"]
                or flag_values["requires_more_specific"]
                or flag_values["mortality_only"]
                or asterisk
            ),
        }
        record["primary_eligibility"] = (
            "not_usable"
            if not record["morbidity_eligible"]
            else "forbidden"
            if flag_values["not_primary"] or asterisk
            else "discouraged"
            if flag_values["discouraged_primary"]
            else "allowed"
        )
        record["auto_single_code_eligible"] = bool(
            record["primary_eligibility"] == "allowed" and not dagger and not asterisk
        )
        records.append(record)
        stats["dagger"] += dagger
        stats["asterisk"] += asterisk
        stats["codable"] += record["codable"]
        stats["morbidity_eligible"] += record["morbidity_eligible"]
        stats["primary_eligible"] += record["primary_eligible"]
        stats["auto_single_code_eligible"] += record["auto_single_code_eligible"]
        stats[f"source_title_en::{source_title_status}"] += 1
        for restriction in restrictions:
            stats[f"restriction::{restriction}"] += 1

    if anomalies:
        preview = "\n".join(anomalies[:20])
        raise RuntimeError(f"Catalog có {len(anomalies)} lỗi:\n{preview}")

    missing_parents = {
        (code, _parent_code(code))
        for code in seen_codes
        if _parent_code(code) is not None and _parent_code(code) not in seen_codes
    }
    if missing_parents:
        raise RuntimeError(f"Có mã thiếu parent: {sorted(missing_parents)[:20]}")
    codes_with_children = {_parent_code(code) for code in seen_codes if _parent_code(code)}
    if codes_with_children != restriction_sets["requires_more_specific"]:
        missing_flags = sorted(codes_with_children - restriction_sets["requires_more_specific"])
        extra_flags = sorted(restriction_sets["requires_more_specific"] - codes_with_children)
        raise RuntimeError(
            "Cột requires_more_specific lệch cây mã: "
            f"thiếu={missing_flags[:20]}, thừa={extra_flags[:20]}"
        )
    if not marker_codes["asterisk"] <= restriction_sets["not_primary"]:
        raise RuntimeError("Có mã * không thuộc danh sách cấm dùng làm bệnh chính")
    if marker_codes["dagger"] & marker_codes["asterisk"]:
        raise RuntimeError("Có code đồng thời mang cả dấu dagger và asterisk")
    if restriction_sets["female_only"] & restriction_sets["male_only"]:
        raise RuntimeError("Danh sách mã chủ yếu nữ và chủ yếu nam giao nhau")

    who_crosscheck = None
    if who_records:
        who_codes = set(who_records)
        who_references = [
            reference
            for who_record in who_records.values()
            for reference in who_record["references"]
        ]
        missing_from_tt06 = sorted(who_codes - seen_codes)
        if missing_from_tt06:
            raise RuntimeError(
                f"TT06 thiếu {len(missing_from_tt06)} category WHO: {missing_from_tt06[:20]}"
            )
        who_crosscheck = {
            "explicit_who_category_codes": len(who_codes),
            "explicit_who_codes_present_in_tt06": len(who_codes & seen_codes),
            "who_codes_missing_from_tt06": 0,
            "tt06_expanded_modifier_codes_not_explicit_in_claml": len(seen_codes - who_codes),
            "usage": dict(
                sorted(Counter(record["usage"] or "none" for record in who_records.values()).items())
            ),
            "references": len(who_references),
            "reference_usage": dict(
                sorted(Counter(reference["usage"] or "none" for reference in who_references).items())
            ),
            "reference_rubric_kind": dict(
                sorted(Counter(reference["rubric_kind"] for reference in who_references).items())
            ),
            "references_with_resource_code": sum(
                bool(reference["resource_code"]) for reference in who_references
            ),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "system": SYSTEM_ID,
        "who_release": WHO_RELEASE,
        "source_page": SOURCE_PAGE,
        "source_pdf": SOURCE_URL,
        "source_sha256": sha256_file(path),
        "pages": EXPECTED_PAGES,
        "rows": len(records),
        "unique_codes": len(seen_codes),
        "extractor": f"PyMuPDF {fitz.VersionBind}",
        "statistics": dict(sorted(stats.items())),
        "restriction_set_sizes": {
            name: len(codes) for name, codes in restriction_sets.items()
        },
        "restriction_occurrences": restriction_occurrences,
        "known_source_anomalies": [
            {"stt": stt, **details}
            for stt, details in sorted(KNOWN_SOURCE_CODE_ANOMALIES.items())
        ],
        "known_pdf_text_title_anomalies": [
            {"code": code, "extracted": titles[0], "canonical_who": titles[1]}
            for code, titles in sorted(KNOWN_PDF_TEXT_TITLE_ANOMALIES.items())
        ],
        "who_claml_crosscheck": who_crosscheck,
        "hierarchy_canonicalization": hierarchy_report,
        "invariants": {
            "stt_contiguous_1_to_15844": True,
            "code_nodot_valid": True,
            "codes_unique": True,
            "english_title_nonempty": True,
            "vietnamese_title_nonempty": True,
            "restriction_codes_exist_in_catalog": True,
            "every_non_root_code_has_parent": True,
            "requires_more_specific_equals_has_children": True,
            "asterisk_codes_are_not_primary": True,
            "dagger_and_asterisk_disjoint": True,
            "female_and_male_sets_disjoint": True,
            "who_usage_matches_pdf_markers": who_records is not None,
            "who_superclass_matches_pdf_hierarchy": who_records is not None,
            "who_preferred_titles_crosschecked": who_records is not None,
            "who_references_preserved": bool(
                who_crosscheck and who_crosscheck["references"] == 5434
            ),
            "hierarchy_code_titles_unique_after_canonicalization": True,
        },
        "note": (
            "Catalog trích từ phụ lục chính thức. Không dùng riêng metadata này như phần mềm "
            "mã hóa lâm sàng; quyết định cuối cần xét hướng dẫn bao gồm/loại trừ và chuyên gia duyệt."
        ),
    }
    return records, report


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp_path, path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def download_file(path: Path, url: str, label: str) -> None:
    if path.exists():
        print(f"[GIỮ] PDF đã tồn tại: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "TCM-System/1.0"})
    fd, temp_name = tempfile.mkstemp(prefix="icd10_tt06_", suffix=".pdf", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    print(f"[OK] Đã tải {label}: {path}")


def load_and_verify_catalog(catalog_path: Path, meta_path: Path) -> tuple[int, str]:
    if not catalog_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Thiếu catalog/meta; hãy chạy builder không có --verify-only trước")
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    count = 0
    codes = set()
    with catalog_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            count += 1
            if record["stt"] != line_number:
                raise RuntimeError(f"STT catalog lệch ở dòng {line_number}")
            if record["code"] in codes:
                raise RuntimeError(f"Code trùng trong catalog: {record['code']}")
            codes.add(record["code"])
    if count != EXPECTED_ROWS or meta.get("rows") != EXPECTED_ROWS:
        raise RuntimeError(f"Catalog cần {EXPECTED_ROWS} dòng, gặp {count}")
    digest = sha256_file(catalog_path)
    if meta.get("catalog_sha256") != digest:
        raise RuntimeError("SHA256 catalog không khớp meta")
    return count, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--who-claml", type=Path, default=DEFAULT_WHO_CLAML)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--download", action="store_true", help="Tải PDF nếu chưa có")
    parser.add_argument(
        "--download-who", action="store_true", help="Tải WHO ICD-10 2019 ClaML nếu chưa có"
    )
    parser.add_argument("--verify-only", action="store_true", help="Chỉ kiểm tra output đã dựng")
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="Cho phép PDF có checksum khác snapshot đã thẩm định (report vẫn ghi checksum mới)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        count, digest = load_and_verify_catalog(args.catalog, args.meta)
        print(f"[OK] Catalog hợp lệ: {count} mã, sha256={digest}")
        return 0
    if args.download:
        download_file(args.source_pdf, SOURCE_URL, "PDF TT06")
    if args.download_who:
        download_file(args.who_claml, WHO_CLAML_URL, "WHO ClaML")
    if not args.source_pdf.exists():
        raise FileNotFoundError(
            f"Không thấy {args.source_pdf}. Dùng --download hoặc --source-pdf PATH."
        )
    source_digest = sha256_file(args.source_pdf)
    if source_digest != SOURCE_SHA256 and not args.allow_source_drift:
        raise RuntimeError(
            "PDF không đúng snapshot TT06 đã khóa. "
            f"chờ={SOURCE_SHA256}, gặp={source_digest}. "
            "Chỉ dùng --allow-source-drift sau khi đã thẩm định bản mới."
        )

    if not args.who_claml.exists():
        raise FileNotFoundError(
            f"Không thấy {args.who_claml}. Dùng --download-who hoặc --who-claml PATH."
        )
    who_digest = sha256_file(args.who_claml)
    if who_digest != WHO_CLAML_SHA256 and not args.allow_source_drift:
        raise RuntimeError(
            "WHO ClaML không đúng snapshot đã khóa. "
            f"chờ={WHO_CLAML_SHA256}, gặp={who_digest}."
        )
    who_records, who_hierarchy = parse_who_claml(args.who_claml)
    records, report = parse_pdf(args.source_pdf, who_records, who_hierarchy)
    report["who_claml_url"] = WHO_CLAML_URL
    report["who_claml_sha256"] = who_digest
    write_jsonl(args.catalog, records)
    report["catalog_sha256"] = sha256_file(args.catalog)
    meta = dict(report)
    try:
        catalog_display_path = args.catalog.resolve().relative_to(ROOT)
    except ValueError:
        catalog_display_path = args.catalog.resolve()
    meta["catalog_path"] = str(catalog_display_path).replace("\\", "/")
    write_json(args.meta, meta)
    write_json(args.report, report)
    count, digest = load_and_verify_catalog(args.catalog, args.meta)
    print(f"[OK] {args.catalog}: {count} mã")
    print(f"[OK] {args.meta}")
    print(f"[OK] {args.report}")
    print(f"[OK] catalog sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        raise SystemExit(1)
