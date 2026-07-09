# src/interview.py
"""Canonicalize phần VẤN CHẨN CÓ CẤU TRÚC (thập vấn + tuổi/giới/thời gian mắc) thành một chuỗi
text tiếng Việt CHUẨN, để ghép vào lời khai tự do trước khi đưa vào pipeline chẩn đoán.

Vì sao cần: hệ chỉ có Vấn + Vọng (không bắt mạch), nên phần Vấn phải gánh việc phân định
hàn/nhiệt/hư/thực. Ô text tự do thu được quá ít; form thập vấn có cấu trúc bổ sung các dấu chỉ
điểm quan trọng. Mỗi lựa chọn được quy về đúng TỪ KHÓA mà các cổng hàn-nhiệt (_kw_hit_clean) và
bộ khớp triệu chứng của DB nhận diện (vd 'sợ lạnh', 'táo bón', 'miệng nhạt không khát', 'lâu ngày').

Hàm THUẦN (không phụ thuộc Neo4j/LLM) nên test được độc lập.
"""

# Mỗi nhóm: mã lựa chọn -> cụm text CHUẨN. '' hoặc mã lạ -> bỏ qua.
_INTERVIEW_MAP = {
    "han_nhiet": {
        "so_lanh": "sợ lạnh",
        "so_nong": "sợ nóng, thích mát",
        "binh_thuong": "",
    },
    "mo_hoi": {
        "tu_han": "ra mồ hôi nhiều ban ngày",   # tự hãn -> khí/dương hư
        "dao_han": "mồ hôi trộm",                # đạo hãn -> âm hư
        "it": "",
    },
    "dai_tien": {
        "tao": "táo bón",
        "long": "đại tiện lỏng",
        "binh_thuong": "",
    },
    "tieu_tien": {
        "vang": "nước tiểu vàng",                # nhiệt
        "trong_dai": "tiểu tiện trong dài",      # hàn
        "binh_thuong": "",
    },
    "khat": {
        "khat_nuoc": "khát nước",                # nhiệt
        "mieng_nhat": "miệng nhạt không khát",   # hàn/thấp
        "binh_thuong": "",
    },
    "an_uong": {
        "an_kem": "ăn kém",
        "mau_doi": "ăn nhiều mau đói",
        "binh_thuong": "",
    },
    "ngu": {
        "kho_ngu": "mất ngủ",
        "binh_thuong": "",
    },
}

_SEX_MAP = {"nam": "nam giới", "nu": "nữ giới"}

_ONSET_MAP = {
    "moi": "bệnh mới mắc",
    "vai_ngay": "bệnh vài ngày",
    "vai_tuan": "bệnh vài tuần",
    "man": "bệnh mạn tính lâu ngày",   # 'lâu ngày' -> kích hoạt nhận diện Hư mạn của pipeline
}


def compose_interview_text(age=None, sex=None, onset=None, answers=None) -> str:
    """Trả về chuỗi text chuẩn từ các lựa chọn thập vấn + nhân khẩu. Rỗng nếu không có gì.

    age: số tuổi (int/str) — bỏ qua nếu không hợp lệ.
    sex: 'nam' | 'nu'.
    onset: khóa trong _ONSET_MAP.
    answers: dict {nhóm: mã lựa chọn} cho các nhóm thập vấn trong _INTERVIEW_MAP.
    """
    parts = []

    # Tuổi
    if age not in (None, ""):
        try:
            a = int(str(age).strip())
            if 0 < a < 130:
                parts.append(f"{a} tuổi")
        except (ValueError, TypeError):
            pass

    # Giới
    if sex:
        s = _SEX_MAP.get(str(sex).strip().lower())
        if s:
            parts.append(s)

    # Thời gian mắc
    if onset:
        o = _ONSET_MAP.get(str(onset).strip().lower())
        if o:
            parts.append(o)

    # Thập vấn
    if answers:
        for group, choices in _INTERVIEW_MAP.items():
            code = answers.get(group)
            if not code:
                continue
            phrase = choices.get(str(code).strip().lower())
            if phrase:
                parts.append(phrase)

    return ", ".join(parts)


def merge_symptoms(free_text: str, interview_text: str) -> str:
    """Ghép lời khai tự do với text thập vấn chuẩn (bỏ phần rỗng, tránh dấu phẩy mồ côi)."""
    free_text = (free_text or "").strip().rstrip(",").strip()
    interview_text = (interview_text or "").strip()
    if free_text and interview_text:
        return f"{free_text}, {interview_text}"
    return free_text or interview_text
