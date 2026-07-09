# src/vision_schema.py
"""Vọng chẩn CÓ CẤU TRÚC: VLM trả JSON theo schema cố định thay vì văn xuôi, rồi map DETERMINISTIC
sang tên triệu chứng chuẩn — thay cho bước LLM đọc prose + hàng loạt regex vá ảo giác phía sau.

Gồm:
  - parse_vlm_json(text): bóc JSON từ output VLM (chịu được rào ```json và prose thừa).
  - tongue_json_to_symptoms / face_json_to_symptoms: JSON -> danh sách triệu chứng chuẩn (bỏ 'không rõ').
  - tongue_json_to_prose / face_json_to_prose: JSON -> câu mô tả tiếng Việt (hiển thị + tương thích ngược).

Giá trị 'không rõ' (hoặc thiếu) -> KHÔNG sinh triệu chứng (không bịa). Tên đầu ra bám sát mapping/DB;
fusion_pipeline._resolve_symptom_conflicts/_normalize_symptoms sẽ chuẩn hóa nốt biến thể còn lại.
"""
import json
import re

# ── Schema: khóa cố định + tập giá trị hợp lệ (để đưa vào prompt & validate) ──
TONGUE_SCHEMA = {
    "than_luoi": ["nhợt", "hồng nhạt", "đỏ", "đỏ sẫm", "tím", "không rõ"],
    "reu_mau": ["trắng", "vàng", "xám đen", "không rêu", "không rõ"],
    "reu_day": ["mỏng", "dày", "không rõ"],
    "reu_chat": ["nhuận", "nhớt", "khô", "bong tróc", "không rõ"],
    "dau_rang": ["có", "không", "không rõ"],
    "vet_nut": ["có", "không", "không rõ"],
    "luoi_beu": ["có", "không", "không rõ"],
}
FACE_SCHEMA = {
    "sac_mat": ["trắng nhợt", "vàng úa", "đỏ bừng", "xanh xao", "sạm tối", "hồng hào bình thường", "không rõ"],
    "go_ma_do": ["có", "không", "không rõ"],
    "phu": ["có", "không", "không rõ"],
    "ban_do": ["có", "không", "không rõ"],
    "quang_tham": ["có", "không", "không rõ"],
    "trang_diem": ["có", "không", "không rõ"],
}

_SKIP = {"", "không rõ", "không", "hồng hào bình thường"}   # giá trị KHÔNG sinh triệu chứng


def parse_vlm_json(text: str):
    """Bóc object JSON đầu tiên từ output VLM. Trả dict hoặc None nếu không parse được."""
    if not text:
        return None
    t = text.strip()
    # Gỡ rào ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        t = m.group(1)
    else:
        # Lấy từ '{' đầu tới '}' cuối (chịu prose bao quanh)
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            t = t[i:j + 1]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _norm(v):
    return (v or "").strip().lower() if isinstance(v, str) else ""


def tongue_json_to_symptoms(data: dict) -> list:
    """JSON lưỡi -> danh sách triệu chứng chuẩn (deterministic)."""
    if not isinstance(data, dict):
        return []
    out = []

    than = _norm(data.get("than_luoi"))
    body_map = {
        "nhợt": "lưỡi nhợt",
        "đỏ": "lưỡi đỏ",
        "đỏ sẫm": "lưỡi đỏ sẫm",
        "tím": "lưỡi có vết bầm tím",
        # 'hồng nhạt' = sinh lý -> bỏ (không sinh triệu chứng)
    }
    if than in body_map:
        out.append(body_map[than])

    if _norm(data.get("luoi_beu")) == "có":
        out.append("lưỡi bệu")
    if _norm(data.get("dau_rang")) == "có":
        out.append("rìa lưỡi có hằn răng")
    if _norm(data.get("vet_nut")) == "có":
        out.append("lưỡi có vết nứt")

    # Rêu: gộp màu + độ dày + tính chất thành MỘT tên chuẩn
    reu_mau = _norm(data.get("reu_mau"))
    reu_day = _norm(data.get("reu_day"))
    reu_chat = _norm(data.get("reu_chat"))
    if reu_mau == "không rêu" or reu_chat == "bong tróc":
        out.append("rêu bong tróc" if reu_chat == "bong tróc" else "lưỡi không có rêu")
    elif reu_mau in ("trắng", "vàng", "xám đen"):
        color = {"trắng": "trắng", "vàng": "vàng", "xám đen": "xám"}[reu_mau]
        if reu_chat == "nhớt":
            out.append(f"rêu {color} nhớt")
        elif reu_day == "dày":
            out.append(f"rêu {color} dày")
        elif reu_day == "mỏng":
            out.append(f"rêu {color} mỏng")
        elif reu_chat == "khô":
            out.append(f"rêu {color} khô")
        else:
            out.append(f"rêu {color}")

    # Khử trùng, giữ thứ tự
    return list(dict.fromkeys(out))


def face_json_to_symptoms(data: dict) -> list:
    """JSON mặt -> danh sách triệu chứng chuẩn (deterministic)."""
    if not isinstance(data, dict):
        return []
    out = []
    sac_map = {
        "trắng nhợt": "mặt nhợt nhạt",
        "vàng úa": "mặt vàng",
        "đỏ bừng": "mặt đỏ",
        "xanh xao": "mặt xanh",
        "sạm tối": "sắc mặt ám tối",
    }
    sac = _norm(data.get("sac_mat"))
    if sac in sac_map:
        out.append(sac_map[sac])
    if _norm(data.get("go_ma_do")) == "có":
        out.append("hai gò má đỏ")
    if _norm(data.get("phu")) == "có":
        out.append("mặt phù")
    if _norm(data.get("ban_do")) == "có":
        out.append("mặt có ban")
    if _norm(data.get("quang_tham")) == "có":
        out.append("quầng đen dưới mắt")
    return list(dict.fromkeys(out))


def is_makeup(data: dict) -> bool:
    return isinstance(data, dict) and _norm(data.get("trang_diem")) == "có"


def tongue_json_to_prose(data: dict) -> str:
    """JSON lưỡi -> câu mô tả tiếng Việt (hiển thị)."""
    if not isinstance(data, dict):
        return ""
    parts = []
    than = _norm(data.get("than_luoi"))
    if than and than not in _SKIP:
        parts.append(f"thân lưỡi {than}")
    reu_mau = _norm(data.get("reu_mau"))
    if reu_mau == "không rêu":
        parts.append("không có rêu")
    elif reu_mau and reu_mau not in _SKIP:
        seg = f"rêu {reu_mau}"
        reu_day = _norm(data.get("reu_day"))
        reu_chat = _norm(data.get("reu_chat"))
        extra = [x for x in (reu_day, reu_chat) if x and x not in _SKIP]
        if extra:
            seg += " " + " ".join(extra)
        parts.append(seg)
    if _norm(data.get("luoi_beu")) == "có":
        parts.append("lưỡi bệu")
    if _norm(data.get("dau_rang")) == "có":
        parts.append("rìa lưỡi có dấu răng")
    if _norm(data.get("vet_nut")) == "có":
        parts.append("bề mặt có vết nứt")
    return ("Lưỡi: " + ", ".join(parts) + ".") if parts else ""


def face_json_to_prose(data: dict) -> str:
    """JSON mặt -> câu mô tả tiếng Việt (hiển thị)."""
    if not isinstance(data, dict):
        return ""
    parts = []
    sac = _norm(data.get("sac_mat"))
    if sac and sac not in _SKIP:
        parts.append(f"sắc mặt {sac}")
    if _norm(data.get("go_ma_do")) == "có":
        parts.append("hai gò má đỏ")
    if _norm(data.get("phu")) == "có":
        parts.append("mặt phù")
    if _norm(data.get("ban_do")) == "có":
        parts.append("có ban đỏ")
    if _norm(data.get("quang_tham")) == "có":
        parts.append("quầng thâm mắt")
    if _norm(data.get("trang_diem")) == "có":
        parts.append("(có trang điểm)")
    return ("Sắc mặt: " + ", ".join(parts) + ".") if parts else ""
