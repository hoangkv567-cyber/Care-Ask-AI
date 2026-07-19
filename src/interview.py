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
    # TÍNH CHẤT khát là dấu phân cực hàn/nhiệt cốt tử, không phải chỉ "có khát hay không":
    #   渴喜冷飲 khát thích uống LẠNH      -> NHIỆT
    #   渴喜熱飲 khát thích uống ẤM/NÓNG   -> HÀN
    #   渴不欲飲 khát mà KHÔNG muốn uống   -> thấp/đàm/ứ huyết/dương hư không hóa tân (KHÔNG phải nhiệt)
    # Trước đây form chỉ có "Khát, thích uống" -> quy về "khát nước" (đánh dấu là NHIỆT). Hệ quả đo
    # được ở ca thật (nữ 34t dương hư): dữ liệu không đủ phân cực nên LLM TỰ BỊA tính chất uống để
    # khớp chẩn đoán — ba lần chạy ra ba kiểu khác nhau ("không thể uống nhiều", "uống nhiều vẫn
    # không giải được", "khát mà không uống được đủ"), đều TRÁI lời khai "thích uống".
    # Cụm chuẩn bám ĐÚNG chữ KB đang dùng ("uống nước lạnh" 6 dòng, "thích uống nóng" 2 dòng) để
    # tầng khớp triệu chứng hưởng luôn, không phải tự chế từ mới.
    "khat": {
        "khat_lanh": "khát, thích uống nước lạnh",      # nhiệt
        "khat_am": "khát, thích uống nóng",             # hàn
        "khat_khong_uong": "khát nhưng không muốn uống",  # thấp/đàm/dương hư — KHÔNG phải nhiệt
        "khat_nuoc": "khát nước",                # mơ hồ (giữ cho tương thích ngược ca cũ)
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
