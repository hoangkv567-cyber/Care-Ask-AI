# src/image_quality.py
"""Gác CHẤT LƯỢNG ẢNH cho vọng chẩn (chỉ dùng Pillow, không cần numpy/OpenCV).

Vì sao cần: màu chất lưỡi / sắc mặt là bằng chứng hàn-nhiệt quan trọng nhất của vọng chẩn khi hệ
KHÔNG có mạch chẩn. Ảnh quá tối, lóa sáng, quá nhỏ hoặc mờ nhòe khiến VLM mô tả sai màu → kéo cả
chẩn đoán lệch. Gác này bắt các ảnh RÕ RÀNG không dùng được và yêu cầu chụp lại, thay vì phân tích
liều. Ngưỡng đặt BẢO THỦ: chỉ chặn ca cực đoan, ảnh thường vẫn qua (tránh chặn oan ảnh lưỡi thật).
"""
import logging

logger = logging.getLogger(__name__)

# Ngưỡng (thang xám 0-255). Bảo thủ — chỉ chặn ca không thể dùng.
_MIN_DIM = 150          # cạnh ngắn tối thiểu (px)
_DARK_MEAN = 25         # tối hơn -> gần như đen
_BRIGHT_MEAN = 235      # sáng hơn -> cháy/lóa mất chi tiết màu
_BLUR_EDGE_MEAN = 2.0   # năng lượng biên trung bình thấp hơn -> gần như không có chi tiết (mờ nặng)


def assess_image_quality(image_path: str):
    """Trả (ok: bool, reason: str). ok=False kèm lý do (tiếng Việt) khi ảnh KHÔNG dùng được.
    Lỗi mở ảnh cũng coi là không dùng được. Không ném ngoại lệ ra ngoài."""
    try:
        from PIL import Image, ImageFilter, ImageStat
    except Exception as e:                       # thiếu Pillow -> không chặn (fail-open, chỉ log)
        logger.warning(f"Không import được Pillow để kiểm chất lượng ảnh: {e}")
        return True, ""

    try:
        img = Image.open(image_path)
        img.load()
    except Exception as e:
        return False, "không mở được ảnh"

    try:
        gray = img.convert("L")
    except Exception:
        return False, "định dạng ảnh không hợp lệ"

    w, h = gray.size
    if min(w, h) < _MIN_DIM:
        return False, "ảnh quá nhỏ / độ phân giải thấp"

    # Thu nhỏ để đo nhanh và ổn định giữa các độ phân giải
    work = gray.copy()
    work.thumbnail((512, 512))

    brightness = ImageStat.Stat(work).mean[0]
    if brightness < _DARK_MEAN:
        return False, "ảnh quá tối"
    if brightness > _BRIGHT_MEAN:
        return False, "ảnh quá sáng / bị lóa"

    # Độ nét xấp xỉ: năng lượng biên trung bình (mờ nặng -> gần 0)
    edge_energy = ImageStat.Stat(work.filter(ImageFilter.FIND_EDGES)).mean[0]
    if edge_energy < _BLUR_EDGE_MEAN:
        return False, "ảnh quá mờ / thiếu chi tiết"

    return True, ""
