#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/clean_tayy_csv.py — Làm sạch data/TayY.csv (8.900 bệnh Tây y dịch máy) thành bộ
truy xuất sơ bộ, làm nền cho bước ánh xạ ICD. KHÔNG BAO GIỜ sửa file nguồn TayY.csv.

v2 — sau vòng thẩm định đối kháng 5 chiều (2026-08-14) vá 4 lỗi lớn ở v1:
  * TRÙNG TÊN ≠ CÙNG BỆNH: ~75% cặp trùng tên là các bệnh KHÁC NHAU bị dịch máy phá tên
    ('Ung thư' = 11 bệnh, 'Bệnh Alzheimer' có 1 dòng nội dung nhiễm trùng huyết...).
    Chỉ gộp khi NỘI DUNG tương đồng; nhóm trùng tên chưa phân xử GIỮ HẾT + cờ
    trung_ten_khac_noi_dung -> tên_bệnh trong clean KHÔNG còn duy nhất tuyệt đối.
  * Chọn dòng giữ theo CHẤT LƯỢNG LÕI (triệu chứng, mô_tả, nguyên_nhân) thay vì đếm ô.
  * Parser danh sách tokenizer nháy-phẩy (chịu apostrophe Job's, phần tử không nháy,
    chuỗi dính 'A,''B, danh sách lồng); tách phẩy tôn trọng ngoặc đơn.
  * Placeholder đối_tượng: CẮT tiền tố, GIỮ đuôi dịch tễ thật.
  * Quarantine ghi từ BẢN GỐC chưa biến đổi.
  * Tách điều trị đòi liều + NGỮ CẢNH THUỐC (dinh dưỡng/vệ sinh không bị tách).

v3 — sau vòng thẩm định 2 (2026-08-14), vá phần còn sót:
  * Tokenizer: token chỉ tính "được nháy bảo vệ" khi MỞ ĐẦU bằng nháy; văn thường có
    ranh giới nháy-phẩy được tách dù không mở đầu bằng nháy; nháy-space-nháy và
    nháy-'và/hoặc'-nháy là ranh giới; nhận cả nháy 「」“”; văn thường không nháy có
    >=2 phẩy top-level (trừ triệu_chứng dễ là văn xuôi) tách phẩy tôn trọng ngoặc;
    _clean_item strip lặp đến ổn định (hết nháy sót mép); cờ danh_sach_chua_tach cho
    ô còn phần tử dính >=2 phẩy — downstream biết ô chưa cấu trúc hóa.
  * Dedup: thêm Jaccard TỪ trên mô_tả (bắt 2 bản dịch cùng bài mà SequenceMatcher ký
    tự bỏ lỡ), autojunk=False + max 2 chiều (hết phụ thuộc thứ tự dòng); bonus
    tên-trong-mô_tả so theo TẬP TỪ (chịu dịch máy đảo trật tự); giữ_lại_cho trỏ đúng
    disease_id CỦA CỤM gộp (v2 trỏ nhầm cụm đầu tiên cùng tên cho 11/130 dòng).
  * Tách điều trị: 'thuốc' không tính khi là 'hút/bỏ/cai thuốc (lá)', 'thuốc nhuộm/
    trừ sâu/diệt...'; 'hóa trị' không tính khi là '(cộng|đa) hóa trị', 'hóa trị ba'
    (valence); dòng nói VẮC-XIN/tiêm chủng/miễn dịch dự phòng LUÔN ở lại phòng tránh;
    mở rộng bắt sót: thêm ~25 tên thuốc, liều %, 'liều cao/thấp', 'hai lần một ngày'
    (số bằng chữ), 'để điều trị'.
  * Placeholder: thêm biến thể đối_tượng ('quần thể', 'người đặc biệt', 'Cá nhân...',
    'Không có gì', 'Mọi người đều có nguy cơ cao'); tiền tố chỉ bị cắt khi có ranh
    giới câu (dấu câu/và/nhưng) — 'Không có người nào miễn nhiễm...' không bị cắt sai
    nghĩa; dòng placeholder NẰM GIỮA văn xuôi nguyên_nhân/cách_phòng_tránh bị lọc.

v4 — sau vòng thẩm định 3, thu hồi 2 trigger v3 quá tay + dọn mép:
  * Tách điều trị: BỎ 'để điều trị' và '%' trần (kéo oan ~500 dòng tiên lượng/thống
    kê/ăn uống); '%' chỉ tính khi đi với động từ bôi/thoa/rửa/ngâm (thuốc dùng ngoài);
    BỎ 'liều cao/thấp/nhỏ'; lá chắn vắc-xin thu hẹp — 'miễn dịch' trần không còn che
    ('ức chế miễn dịch', 'bệnh tự miễn dịch' là điều trị/bệnh học, chỉ che 'miễn dịch
    dự phòng/trước phơi nhiễm', 'tăng cường miễn dịch'...).
  * Tokenizer: tách cả 'A'và'B' không space, phẩy ideographic '、', mẫu ', và "X"';
    _clean_item bỏ '(' mồ côi cuối item; giữ triệu chứng 1 ký tự thật ('ợ'); văn
    thường tách phẩy gom lại mảnh <5 ký tự ('Chụp CT tai, mũi, họng' không bị băm);
    _top_level_commas đếm cả '、'.
  * row_score: bonus tên-trong-mô_tả tách từ bằng \\w+ (hết trượt vì dấu câu dính).
  * Vá chốt sau vòng 4: thêm cyclophosphamide/penicillamine/NSAID/glucocorticoid...
    vào danh mục thuốc, đơn vị 'triệu U', 'liệu pháp sốc'; nháy đơn cong ‘’ vào bộ nháy.

Cột mới: disease_id (TAYY-#####, theo thứ tự file nguồn, ổn định khi chạy lại),
icd10_code (rỗng — bước sau), trạng_thái_kiểm_duyệt ("chua_kiem_duyet"),
điều_trị_tách_từ_phòng_tránh, tỉ_lệ_chữa_khỏi_min/max, cờ_chất_lượng.

Đầu ra (ghi đè nếu đã có; nguồn chỉ đọc):
    data/TayY_clean.csv         — 1 dòng/bệnh (tên có thể trùng nếu cắm cờ)
    data/TayY_quarantine.csv    — dòng bị loại + lý_do_loại + giữ_lại_cho
    data/tayy_clean_report.json — thống kê trước/sau

Chạy:  python scripts/clean_tayy_csv.py
"""
import ast
import csv
import difflib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

csv.field_size_limit(10 ** 9)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(_ROOT, "data", "TayY.csv")
OUT_CLEAN = os.path.join(_ROOT, "data", "TayY_clean.csv")
OUT_QUAR = os.path.join(_ROOT, "data", "TayY_quarantine.csv")
OUT_REPORT = os.path.join(_ROOT, "data", "tayy_clean_report.json")

LIST_COLS = [
    "loại_bệnh", "triệu_chứng", "bệnh_đi_kèm", "kiểm_tra",
    "nên_ăn_thực_phẩm_chứa", "không_nên_ăn_thực_phẩm_chứa",
    "đề_xuất_món_ăn", "đề_xuất_thuốc", "thuốc_phổ_biến",
]
DRUG_LIST_COLS = {"đề_xuất_thuốc", "thuốc_phổ_biến"}
CORE_COLS = ["mô_tả_bệnh", "nguyên_nhân", "triệu_chứng", "kiểm_tra",
             "cách_phòng_tránh", "khoa_điều_trị", "đối_tượng_dễ_mắc_bệnh"]

# ---------------------------------------------------------------- placeholder --
# Ô/dòng coi là RỖNG: "không có dữ liệu" đúng nghĩa. KHÔNG bắt phủ định y khoa có
# nghĩa ("Không có phương pháp điều trị hiệu quả" là thông tin thật).
_PLACEHOLDER_ANY = re.compile(
    r"^(hiện tại\s*,?\s*)?(không có|chưa có|không rõ)\s*(thông tin|dữ liệu|số liệu"
    r"|thống kê)?( thống kê)?( liên quan| tham khảo| đáng tin cậy| cụ thể| chi tiết)*[.\s]*$",
    re.IGNORECASE,
)
# Riêng đối_tượng_dễ_mắc_bệnh: tiền tố "không có nhóm người cụ thể" (kể cả 'Dân số
# không vô căn' — rác dịch máy của 无特定人群). Chỉ CẮT tiền tố; đuôi có nội dung thì GIỮ.
_DT_PREFIX = re.compile(
    r"^(cá nhân\s+)?(nhóm không có đặc điểm cụ thể|dân số không vô căn|không có gì"
    r"|mọi người đều (?:dễ bị bệnh|có nguy cơ cao)"
    r"|không có (?:một )?(?:dân số|đám đông|nhóm|quần thể|người)( người)?( có)?( nguy cơ cao)?"
    r"( đặc biệt| cụ thể| đặc thù| nào| rõ ràng)*( bị ảnh hưởng bởi căn bệnh này)?)"
    r"(?P<sep>[\s.,;]*)",
    re.IGNORECASE,
)
_DT_TAIL_TRIM = re.compile(r"^(và|nhưng|,|;)\s+", re.IGNORECASE)

# ---------------------------------------------------- breadcrumb -> chuyên khoa --
_BREADCRUMB_MAP = [
    (re.compile(r"ophthalmolog", re.IGNORECASE), "nhãn khoa"),
    (re.compile(r"oral pentagenolog", re.IGNORECASE), "răng hàm mặt"),
    (re.compile(r"\bENT\b|pentagenolog", re.IGNORECASE), "tai mũi họng"),
    (re.compile(r"nutrition", re.IGNORECASE), "dinh dưỡng"),
    (re.compile(r"internal medicine", re.IGNORECASE), "nội khoa"),
    (re.compile(r"sức khỏe sinh sản", re.IGNORECASE), "sức khỏe sinh sản"),
    (re.compile(r"bệnh bỏng", re.IGNORECASE), "bỏng"),
]
_BREADCRUMB_DROP = re.compile(r"bách khoa toàn thư|encyclopedia|đổi hướng từ", re.IGNORECASE)
_EN_SUFFIX = re.compile(r"\(bằng tiếng anh\)\.?", re.IGNORECASE)

# ------------------------------------------------------------- list tokenizer --
_QCHARS = "'\"「」“”‘’"
_Q = r"['\"「」“”‘’]"
# Ranh giới phần tử: phẩy (cả '、') kề nháy (chịu apostrophe trong item),
# nháy-cách-nháy (chuỗi dính "A" "B"), nháy-'và/hoặc'-nháy (kể cả không space),
# mẫu ', và "X"'.
_SEP = re.compile(
    rf"(?<={_Q})\s*[,、]\s*|[,、]\s*(?:(?:và|hoặc)\s+)?(?={_Q})"
    rf"|(?<={_Q})(?:\s*(?:và|hoặc)\s*|\s+)(?={_Q})"
)
_ADJACENT_QUOTES = re.compile(rf"{_Q}\s*{_Q}")      # bẫy ast nối chuỗi liền kề
_NESTED_LIST_ITEM = re.compile(rf"{_Q}\s*,\s*{_Q}")

# ------------------------------------------- tách điều trị khỏi cách_phòng_tránh --
# BẢO THỦ: chỉ tách khi liều lượng ĐI KÈM ngữ cảnh thuốc. Dinh dưỡng ('500g ngũ cốc',
# '2000 ml nước'), vệ sinh ('đánh răng 3 lần/ngày'), khử trùng môi trường, ngưỡng
# chẩn đoán, 'X-quang liều lượng nhỏ', và MỌI dòng vắc-xin/tiêm chủng -> GIỮ.
_DOSE = re.compile(
    r"\d+\s*(?:mg|ml|g|mcg|µg|iu)\b|\d+(?:[.,]\d+)?\s*(?:triệu|vạn|nghìn)?\s*đơn vị\b"
    r"|\d+(?:[.,]\d+)?\s*(?:triệu|vạn|nghìn)\s*U\b"
    r"|\d+\s*viên\b|\d+\s*giọt\b|\d+\s*lần\s*(?:/|một|mỗi)\s*ngày"
    r"|ngày\s+\d+\s*lần|(?:một|hai|ba|bốn|năm|sáu)\s+lần\s+(?:/|một|mỗi)\s*ngày"
    r"|liều\s+(?:lượng|dùng|khởi đầu|duy trì)"
    r"|mỗi ngày\s+\d+\s*(?:lần|viên|mg|ml|giọt)",
    re.IGNORECASE,
)
# Liều nồng độ % chỉ tính khi là thuốc DÙNG NGOÀI (bôi/rửa...) — '%' trần bắt oan
# thống kê tiên lượng ('tỷ lệ sống 90%').
_PCT_DOSE = re.compile(r"\d[\d.,]*\s*%")
_TOPICAL = re.compile(r"bôi|thoa|đắp|ngâm|nhỏ (?:mắt|mũi|tai|giọt)|rửa", re.IGNORECASE)
# 'thuốc' KHÔNG tính khi là hành vi hút/bỏ/cai thuốc (lá) hay hóa chất (thuốc nhuộm,
# thuốc trừ sâu...). 'hóa trị' KHÔNG tính khi là valence (cộng/đa hóa trị, hóa trị ba).
_DRUG_CTX = re.compile(
    r"(?<!hút )(?<!bỏ )(?<!cai )thuốc(?! lá| nhuộm| trừ sâu| diệt| súng| nổ)"
    r"|viên nang|viên nén|kháng sinh|tiêm bắp|tiêm tĩnh mạch|truyền tĩnh mạch"
    r"|penicillin|erythromycin|amoxicillin|ampicillin|gentamicin|aspirin|corticoid"
    r"|quinolone|metronidazole|azithromycin|fluconazole|itraconazole|ketoconazole"
    r"|nystatin|phenobarbital|rifampicin|isoniazid|streptomycin|chloramphenicol"
    r"|tetracycline|doxycycline|ciprofloxacin|ofloxacin|levofloxacin|acyclovir"
    r"|ribavirin|interferon|dexamethasone|prednisolone|prednisone|hydrocortisone"
    r"|thyroxine|sulfadiazine|trimethoprim|mebendazole|albendazole|praziquantel"
    r"|chloroquine|primaquine|cyclophosphamide|penicillamine|naproxen|ibuprofen"
    r"|nifedipine|glucocorticoid|methylprednisolone|amphotericin|levodopa",
    re.IGNORECASE,
)
_TREAT_ACTION = re.compile(
    r"(?<!cộng )(?<!đa )hóa trị(?! ba| hai| bốn| năm)|xạ trị|phẫu thuật cắt"
    r"|tiêm bắp|tiêm tĩnh mạch|truyền tĩnh mạch|liệu pháp sốc"
    r"|^\s*(?:\d+[.、)-]\s*)?(?:uống|tiêm|truyền|bôi|dùng)\s+(?:thuốc|kháng sinh)",
    re.IGNORECASE,
)
# Dòng vắc-xin/tiêm chủng/miễn dịch DỰ PHÒNG là phòng ngừa điển hình -> luôn giữ.
# KHÔNG che 'miễn dịch' trần: 'ức chế miễn dịch' là thuốc điều trị, 'bệnh tự miễn
# dịch' là bệnh học.
_VACCINE = re.compile(
    r"vắc[- ]?xin|vaccine|tiêm phòng|tiêm chủng|chủng ngừa"
    r"|miễn dịch (?:dự phòng|trước)|tăng cường (?:hệ )?miễn dịch"
    r"|nâng cao (?:sức )?(?:đề kháng|miễn dịch)|khả năng miễn dịch",
    re.IGNORECASE,
)

_PCT = r"(\d{1,3}(?:[.,]\d+)?)"
_RATE_PATTERNS = [
    (re.compile(rf"^{_PCT}\s*%?\s*[-–~đến]+\s*{_PCT}\s*%$", re.IGNORECASE), "range"),
    (re.compile(rf"^{_PCT}\s*%$"), "exact"),
    (re.compile(rf"^(?:khoảng|xấp xỉ|gần|tầm)\s*{_PCT}\s*%$", re.IGNORECASE), "approx"),
    (re.compile(rf"^(?:hơn|trên|lớn hơn|cao hơn|≥|>)\s*{_PCT}\s*%$", re.IGNORECASE), "min_only"),
    (re.compile(rf"^(?:dưới|thấp hơn|nhỏ hơn|≤|<)\s*{_PCT}\s*%$", re.IGNORECASE), "max_only"),
]


def _norm_ws(s):
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", s).strip()


def _fold(s):
    """Bỏ dấu + casefold để so khớp chịu biến thể dịch máy (Amíp ~ amip)."""
    s = unicodedata.normalize("NFD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d").replace("Đ", "d")


def _num(s):
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def parse_rate(raw):
    """'85%-90%' / 'Khoảng 40%' / 'hơn 90%' -> (min, max); không parse được -> (None, None)."""
    v = _norm_ws(raw)
    for rx, kind in _RATE_PATTERNS:
        m = rx.match(v)
        if not m:
            continue
        a = _num(m.group(1))
        if a is None or a > 100:
            return None, None
        if kind == "range":
            b = _num(m.group(2))
            if b is None or b > 100:
                return None, None
            return min(a, b), max(a, b)
        if kind == "min_only":
            return a, None
        if kind == "max_only":
            return None, a
        return a, a
    return None, None


def _top_level_commas(s):
    n = depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch in ",、" and depth == 0:
            n += 1
    return n


def _clean_item(x):
    """Strip lặp đến ổn định — hết nháy/ngoặc/phẩy sót ở mép dù xếp lớp lẫn nhau."""
    x = _norm_ws(str(x))
    prev = None
    while prev != x:
        prev = x
        x = x.strip(" ,;")
        x = re.sub(rf"^[\s\[\]{_QCHARS}]+|[\s\[\]{_QCHARS}]+$", "", x)
        x = re.sub(r"[.…]{2,}\s*$", "", x)
        if x.endswith(")") and x.count(")") > x.count("("):
            x = x[:-1]
        if x.endswith("(") and x.count("(") > x.count(")"):
            x = x[:-1]
    return x


def _split_paren_aware(s):
    """Tách phẩy nhưng KHÔNG cắt trong ngoặc đơn — giữ nguyên '(OT, PPD)'."""
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _tokenize(body, flags):
    """Tách danh sách bẩn theo ranh giới nháy-phẩy. Chịu: apostrophe trong item
    (Job's), phần tử không nháy xen kẽ, chuỗi dính 'A,''B, đuôi cắt cụt."""
    if sum(body.count(q) for q in _QCHARS) % 2 == 1 or body.rstrip().endswith("..."):
        flags.add("list_cat_cut")
    tokens = _SEP.split(body)
    items = []
    for tk in tokens:
        t = tk.strip()
        if not t:
            continue
        # chỉ MỞ ĐẦU bằng nháy mới coi là được-nháy-bảo-vệ (đuôi dính 1 nháy thì không)
        if t[0] not in _QCHARS and "," in t:
            items.extend(_split_paren_aware(t))
        else:
            items.append(t)
    return items


def parse_list(raw, col, flags, depth=0):
    """Parser khoan dung cho cột danh sách. Trả về list[str] đã làm sạch."""
    v = _norm_ws(raw)
    if not v:
        return []
    items = None
    n_quotes = sum(v.count(q) for q in _QCHARS)
    if v.startswith("["):
        try:
            got = json.loads(v)
            items = got if isinstance(got, list) else [got]
        except Exception:
            pass
        # ast chỉ khi KHÔNG có bẫy nối chuỗi liền kề ('A,''B' -> ast dính thành 1)
        if items is None and not _ADJACENT_QUOTES.search(v):
            try:
                got = ast.literal_eval(v)
                items = list(got) if isinstance(got, (list, tuple)) else [got]
            except Exception:
                pass
        if items is None:
            body = v[1:-1] if v.endswith("]") else v[1:]
            items = _tokenize(body, flags) if n_quotes >= 2 else _split_paren_aware(body)
    elif n_quotes >= 2 and _SEP.search(v):
        items = _tokenize(v, flags)               # văn thường chứa danh sách trong nháy
    elif col != "triệu_chứng" and _top_level_commas(v) >= 2:
        # văn thường 'A, B, C' (trừ triệu_chứng — dễ là văn xuôi có phẩy); gom lại
        # mảnh <5 ký tự để 'Chụp CT tai, mũi, họng' không bị băm thành 3 item
        parts, items = _split_paren_aware(v), []
        for p in parts:
            t = p.strip()
            if items and len(t) < 5:
                items[-1] = items[-1].rstrip() + ", " + t
            else:
                items.append(p)
    else:                                         # văn xuôi/đơn mục
        if col in DRUG_LIST_COLS and len(v) > 80:
            flags.add("thuoc_khong_tach")
        items = [v]
    # phần tử là nguyên một danh-sách-con -> bung đệ quy
    if depth < 2:
        expanded = []
        for it in items:
            s = str(it).strip()
            if s.startswith("[") or (_NESTED_LIST_ITEM.search(s)
                                     and sum(s.count(q) for q in _QCHARS) >= 4):
                expanded.extend(parse_list(s, col, set(), depth + 1))
            else:
                expanded.append(it)
        items = expanded
    out = _finalize_items(items, col)
    if any(_top_level_commas(it) >= 2 and len(it) >= 60 for it in out):
        flags.add("danh_sach_chua_tach")          # tín hiệu: ô còn phần tử chưa nguyên tử
    return out


def _map_loai_benh_item(it):
    for rx, spec in _BREADCRUMB_MAP:
        if rx.search(it):
            return spec
    if _BREADCRUMB_DROP.search(it):
        return None
    it = _EN_SUFFIX.sub("", it).strip(" .")
    return it or None


def _finalize_items(items, col):
    out, seen = [], set()
    for it in items:
        it = _clean_item(it)
        # 'ợ' (ợ hơi) là triệu chứng thật 1 ký tự — không vứt cùng rác 'I'/'Ⅱ'
        if (len(it) < 2 and it != "ợ") or _PLACEHOLDER_ANY.match(it):
            continue
        if col == "loại_bệnh":
            it = _map_loai_benh_item(it)
            if not it:
                continue
        k = _fold(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _drop_placeholder_lines(text):
    """Lọc dòng placeholder nằm giữa văn xuôi ('(1) Nguyên nhân bệnh / Hiện tại
    không có thông tin liên quan / (2) Bệnh sinh ...')."""
    kept, dropped = [], 0
    for ln in (text or "").split("\n"):
        t = ln.strip()
        if t and len(t) < 80 and _PLACEHOLDER_ANY.match(t):
            dropped += 1
            continue
        kept.append(ln)
    return "\n".join(kept).strip(), dropped


def split_prevention(raw, flags):
    """Tách dòng điều trị (liều + ngữ cảnh thuốc) khỏi cách_phòng_tránh."""
    v = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not v:
        return "", "", 0
    keep, moved, dropped = [], [], 0
    for line in v.split("\n"):
        t = line.strip()
        if not t:
            continue
        if len(t) < 80 and _PLACEHOLDER_ANY.match(t):
            dropped += 1
            continue
        if _VACCINE.search(t):                    # tiêm chủng/miễn dịch dự phòng = phòng ngừa
            keep.append(t)
            continue
        dose = _DOSE.search(t) or (_PCT_DOSE.search(t) and _TOPICAL.search(t))
        is_treat = _TREAT_ACTION.search(t) or (dose and _DRUG_CTX.search(t))
        (moved if is_treat else keep).append(t)
    if moved:
        flags.add("phong_tranh_lan_dieu_tri")
    return "\n".join(keep), "\n".join(moved), dropped


# ----------------------------------------------------------- dedup theo nội dung --
def _sym_set(r, cache):
    key = id(r)
    if key not in cache:
        cache[key] = {_fold(x) for x in parse_list(r.get("triệu_chứng"), "triệu_chứng", set())}
    return cache[key]


def _same_disease(r1, r2, sym_cache):
    """Trùng tên chưa chắc cùng bệnh (dịch máy phá tên). Chỉ coi là bản trùng khi
    NỘI DUNG tương đồng; nghi ngờ -> KHÁC bệnh (giữ cả hai, an toàn hơn mất bệnh)."""
    if all((r1.get(c) or "").strip() == (r2.get(c) or "").strip() for c in r1):
        return True
    d1, d2 = _fold(r1.get("mô_tả_bệnh")), _fold(r2.get("mô_tả_bệnh"))
    if len(d1) >= 80 and len(d2) >= 80:
        a, b = d1[:600], d2[:600]
        # autojunk=False: mặc định coi ký tự phổ biến là junk -> tỷ số nhiễu trên
        # tiếng Việt bỏ dấu; max 2 chiều: SequenceMatcher không đối xứng theo thứ tự.
        ratio = max(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(),
                    difflib.SequenceMatcher(None, b, a, autojunk=False).ratio())
        if ratio >= 0.5:
            return True
        w1, w2 = set(d1.split()), set(d2.split())
        if len(w1) >= 20 and len(w2) >= 20 and len(w1 & w2) / len(w1 | w2) >= 0.55:
            return True                            # 2 bản dịch cùng bài, đảo câu chữ
    s1, s2 = _sym_set(r1, sym_cache), _sym_set(r2, sym_cache)
    if len(s1) >= 3 and len(s2) >= 3 and len(s1 & s2) / len(s1 | s2) >= 0.6:
        return True
    return False


def _cluster(grp, sym_cache):
    """Union-find trong nhóm trùng tên: mỗi cụm = một bệnh thật."""
    n = len(grp)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _same_disease(grp[i][1], grp[j][1], sym_cache):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(grp[i])
    return list(clusters.values())


def row_score(r):
    """Điểm CHẤT LƯỢNG LÕI để chọn dòng giữ trong cụm bản trùng: triệu chứng parse
    được, mô_tả/nguyên_nhân dày, tên bệnh xuất hiện trong mô_tả (chống giữ nhầm dòng
    mô tả bệnh khác; so theo TẬP TỪ vì dịch máy hay đảo trật tự). Cột món ăn/thuốc
    filler KHÔNG được tính."""
    n_sym = len(parse_list(r.get("triệu_chứng"), "triệu_chứng", set()))
    score = min(n_sym, 20) * 50
    score += min(len(r.get("mô_tả_bệnh") or ""), 3000) / 6.0
    score += min(len(r.get("nguyên_nhân") or ""), 3000) / 10.0
    score += 100 * sum(1 for c in CORE_COLS if (r.get(c) or "").strip())
    name_words = [w for w in re.findall(r"\w+", _fold(re.sub(r"^(bệnh|chứng|hội chứng)\s+", "",
                                                             r.get("tên_bệnh") or ""))) if len(w) >= 2]
    desc_words = set(re.findall(r"\w+", _fold(r.get("mô_tả_bệnh") or "")[:400]))
    if name_words and all(w in desc_words for w in name_words):
        score += 800
    return score


def main():
    if not os.path.exists(SRC):
        print(f"[LỖI] Không thấy {SRC}")
        sys.exit(1)

    with open(SRC, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        src_cols = list(reader.fieldnames)
        rows = list(reader)
    raw_rows = [dict(r) for r in rows]        # bản gốc nguyên vẹn cho quarantine

    report = {"nguồn": {"file": "data/TayY.csv", "số_dòng": len(rows), "số_cột": len(src_cols)}}
    stats = Counter()

    # ---- Bước 0: chuẩn hóa whitespace tên + placeholder ----
    for r in rows:
        r["tên_bệnh"] = re.sub(r"\s+", " ", (r.get("tên_bệnh") or "")).strip()
        for c in src_cols:
            v = (r.get(c) or "").strip()
            if v and len(v) < 80 and _PLACEHOLDER_ANY.match(v):
                r[c] = ""
                stats["ô_placeholder_xóa"] += 1
        dt = (r.get("đối_tượng_dễ_mắc_bệnh") or "").strip()
        m = _DT_PREFIX.match(dt) if dt else None
        if m:
            tail = dt[m.end():].strip()
            sep = m.group("sep") or ""
            # chỉ cắt khi có ranh giới câu — tránh cắt sai nghĩa câu tiếp diễn
            # ('Không có người nào miễn nhiễm với bệnh này' phải giữ nguyên)
            boundary = (not tail) or any(ch in sep for ch in ".,;") or _DT_TAIL_TRIM.match(tail)
            if boundary:
                tail = _DT_TAIL_TRIM.sub("", tail)
                if len(tail) >= 8:
                    r["đối_tượng_dễ_mắc_bệnh"] = tail
                    stats["đối_tượng_cắt_tiền_tố_giữ_đuôi"] += 1
                else:
                    r["đối_tượng_dễ_mắc_bệnh"] = ""
                    stats["đối_tượng_placeholder_xóa"] += 1

    # ---- Bước 1: quarantine tên rác ----
    quarantine = []          # (pos, lý_do)
    body = []                # (pos, row)
    for pos, r in enumerate(rows):
        if r["tên_bệnh"].lower().startswith("lời bài hát"):
            quarantine.append((pos, "ten_rac_dich_may"))
            stats["loại_tên_rác"] += 1
        elif not r["tên_bệnh"]:
            quarantine.append((pos, "ten_rong"))
            stats["loại_tên_rỗng"] += 1
        else:
            body.append((pos, r))

    # ---- Bước 2: nhóm trùng tên -> cụm nội dung -> giữ dòng tốt nhất mỗi cụm ----
    by_name = defaultdict(list)
    for pos, r in body:
        by_name[r["tên_bệnh"].casefold()].append((pos, r))

    sym_cache = {}
    kept = []                # (pos, row)
    extra_flags = defaultdict(set)             # pos -> cờ
    dup_losers = []          # (loser_pos, winner_pos) — lineage đúng CỤM
    for _, grp in by_name.items():
        if len(grp) == 1:
            kept.append(grp[0])
            continue
        clusters = _cluster(grp, sym_cache)
        for cl in clusters:
            best_pos, best = max(cl, key=lambda pr: (row_score(pr[1]), -pr[0]))
            kept.append((best_pos, best))
            if len(clusters) > 1:
                # cùng tên, nội dung chưa đủ tương đồng để gộp -> giữ hết, chưa phân
                # xử (có thể là bệnh khác HOẶC 2 bản dịch quá lệch của cùng bệnh)
                extra_flags[best_pos].add("trung_ten_khac_noi_dung")
                stats["trùng_tên_chưa_phân_xử_giữ"] += 1
            for pos, r in cl:
                if r is not best:
                    dup_losers.append((pos, best_pos))
                    stats["loại_bản_trùng"] += 1
    kept.sort()

    # ---- Bước 3: biến đổi từng dòng giữ lại ----
    out_rows = []
    pos2id = {}
    for i, (pos, r) in enumerate(kept, 1):
        flags = set(extra_flags.get(pos, ()))
        o = {"disease_id": f"TAYY-{i:05d}"}
        pos2id[pos] = o["disease_id"]
        for c in src_cols:
            if c in LIST_COLS:
                items = parse_list(r.get(c), c, flags)
                o[c] = json.dumps(items, ensure_ascii=False) if items else ""
                if items:
                    stats[f"parse_ok::{c}"] += 1
            elif c == "cách_phòng_tránh":
                keep_txt, moved_txt, dropped = split_prevention(r.get(c), flags)
                o[c] = keep_txt
                o["điều_trị_tách_từ_phòng_tránh"] = moved_txt
                stats["dòng_placeholder_giữa_văn_xóa"] += dropped
            elif c == "nguyên_nhân":
                o[c], dropped = _drop_placeholder_lines(r.get(c))
                stats["dòng_placeholder_giữa_văn_xóa"] += dropped
            elif c == "tỉ_lệ_chữa_khỏi":
                o[c] = _norm_ws(r.get(c))
                lo, hi = parse_rate(r.get(c))
                o["tỉ_lệ_chữa_khỏi_min"] = "" if lo is None else f"{lo:g}"
                o["tỉ_lệ_chữa_khỏi_max"] = "" if hi is None else f"{hi:g}"
                if o[c] and lo is None and hi is None:
                    stats["tỉ_lệ_không_parse_được"] += 1
            else:
                o[c] = (r.get(c) or "").strip()
        o["icd10_code"] = ""
        o["trạng_thái_kiểm_duyệt"] = "chua_kiem_duyet"
        o["cờ_chất_lượng"] = ";".join(sorted(flags))
        for fl in flags:
            stats[f"cờ::{fl}"] += 1
        out_rows.append(o)

    out_cols = (["disease_id"] + src_cols
                + ["điều_trị_tách_từ_phòng_tránh", "tỉ_lệ_chữa_khỏi_min", "tỉ_lệ_chữa_khỏi_max",
                   "icd10_code", "trạng_thái_kiểm_duyệt", "cờ_chất_lượng"])
    # đặt cột mới liền sau cột gốc liên quan cho dễ đọc
    out_cols.remove("điều_trị_tách_từ_phòng_tránh")
    out_cols.insert(out_cols.index("cách_phòng_tránh") + 1, "điều_trị_tách_từ_phòng_tránh")
    out_cols.remove("tỉ_lệ_chữa_khỏi_min"); out_cols.remove("tỉ_lệ_chữa_khỏi_max")
    idx = out_cols.index("tỉ_lệ_chữa_khỏi")
    out_cols[idx + 1:idx + 1] = ["tỉ_lệ_chữa_khỏi_min", "tỉ_lệ_chữa_khỏi_max"]

    with open(OUT_CLEAN, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)

    # Quarantine ghi BẢN GỐC (raw_rows) — cam kết không mất dữ liệu.
    quar_cols = ["lý_do_loại", "giữ_lại_cho"] + src_cols
    with open(OUT_QUAR, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=quar_cols)
        w.writeheader()
        for pos, reason in quarantine:
            w.writerow({"lý_do_loại": reason, "giữ_lại_cho": "",
                        **{c: raw_rows[pos].get(c) or "" for c in src_cols}})
        for loser_pos, winner_pos in dup_losers:
            w.writerow({"lý_do_loại": "trung_ten_ban_trung",
                        "giữ_lại_cho": pos2id.get(winner_pos, ""),
                        **{c: raw_rows[loser_pos].get(c) or "" for c in src_cols}})

    report["kết_quả"] = {
        "TayY_clean.csv": {"số_dòng": len(out_rows), "số_cột": len(out_cols)},
        "TayY_quarantine.csv": {"số_dòng": len(quarantine) + len(dup_losers)},
    }
    report["thống_kê"] = dict(sorted(stats.items()))
    report["ghi_chú"] = [
        "tỉ_lệ_chữa_khỏi/đề_xuất_thuốc/thuốc_phổ_biến/thông_tin_thuốc: CHỈ tham khảo nội bộ, "
        "CẤM đưa vào câu trả lời bệnh nhân (chưa kiểm duyệt, nguồn dịch máy).",
        "tên_bệnh KHÔNG duy nhất tuyệt đối: dòng cờ trung_ten_khac_noi_dung là nhóm trùng tên "
        "CHƯA PHÂN XỬ — có thể là các bệnh khác nhau bị dịch máy phá tên, hoặc 2 bản dịch quá "
        "lệch của cùng một bài; ưu tiên phân xử khi ánh xạ ICD.",
        "cờ danh_sach_chua_tach: ô danh sách còn phần tử dính nhiều mục (nguồn quá bẩn để tách "
        "an toàn) — downstream không nên coi phần tử đó là nguyên tử.",
        "disease_id gán theo thứ tự file nguồn — chạy lại trên cùng TayY.csv cho cùng id.",
        "Bước tiếp theo: ánh xạ icd10_code + trạng_thái_kiểm_duyệt bởi chuyên gia.",
    ]
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] {OUT_CLEAN}: {len(out_rows)} dòng, {len(out_cols)} cột")
    print(f"[OK] {OUT_QUAR}: {len(quarantine) + len(dup_losers)} dòng")
    print(f"[OK] {OUT_REPORT}")
    for k, v in sorted(stats.items()):
        print(f"     {k}: {v}")


if __name__ == "__main__":
    main()
