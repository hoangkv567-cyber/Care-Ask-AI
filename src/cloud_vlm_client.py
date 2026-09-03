# src/cloud_vlm_client.py — client vision OpenAI-compatible (HuggingFace / Requesty router / ...)
import os
import base64
import logging
from src.prompts import (
    TONGUE_PROMPT_TEMPLATE_VI, FACE_PROMPT_TEMPLATE_VI,
    TONGUE_JSON_PROMPT_VI, FACE_JSON_PROMPT_VI,
)

logger = logging.getLogger("cloud_vlm")

# Cạnh dài tối đa của ảnh gửi lên cloud. Ảnh điện thoại 4000px nén xuống 1280px
# vẫn thừa chi tiết cho vọng chẩn (LLaVA cũ chỉ nhìn 336px) nhưng payload nhỏ hơn ~10 lần.
#
# ⚠ ĐỪNG HẠ XUỐNG ĐỂ "CHỮA LỖI 400". Đã có lần hạ 1280->1024 kèm quality 90->85 với lý do tránh
# DashScope 400 Bad Request, nhưng ĐO LẠI thì payload KHÔNG phải nguyên nhân: gửi thử ảnh nhiễu
# (trường hợp nén XẤU NHẤT, ảnh thật nén tốt hơn nhiều) ở 1024/1280/1536/1792/2048 px đều trả
# HTTP 200, kể cả bản 2048 nặng 2196 KB base64. Lỗi 400 thật đến từ EXIF/chuyển đổi ảnh của máy
# điện thoại và đã được sửa riêng bằng ImageOps.exif_transpose + bọc try/except ở _encode_image.
# Cái giá của việc hạ: 1024 mất 36% số điểm ảnh so với 1280, cộng nén mạnh hơn -> LÀM PHẲNG các
# chuyển sắc da tinh tế. Đo được trên ca thật: CÙNG một ảnh mặt cho 'trắng nhợt' ở cấu hình cũ và
# 'hồng hào bình thường' ở cấu hình mới — hai nhãn đối lập, chỉ vì ảnh gửi đi đã khác.
# Chiều trôi dạt cũng đoán được: prompt bắt 'trắng nhợt' phải có bằng chứng MẠNH (tái bệch + gầy
# hốc hác + môi mất sắc máu), nên khi mất chi tiết là rơi về nhãn còn lại.
_MAX_IMAGE_EDGE = 1280
_JPEG_QUALITY = 90
# Bước phân loại thô (lưỡi/mặt/khác) chỉ cần nhìn tổng thể — ảnh nhỏ giảm mạnh thời gian prefill
_CLASSIFY_IMAGE_EDGE = 512


class CloudVLMClient:
    """Client vision OpenAI-compatible (HuggingFace / Requesty router), giữ nguyên giao diện của OllamaTCMClient
    (diagnose_image / verify_image_modality / set_symptom_list) để pipeline thay thế được.

    Khác biệt với LLaVA local:
    - Nhìn ảnh ở độ phân giải cao (không nén về 336px) -> thấy dấu răng, vân rêu, sắc da thật.
    - Mô tả thẳng bằng TIẾNG VIỆT -> fusion_pipeline bỏ qua được bước dịch Anh->Việt.
    - Khi gọi cloud lỗi (mất mạng, hết quota) tự chuyển về fallback_client (LLaVA local).
    """

    def __init__(self, api_key: str, model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
                 config: dict = None, fallback_client=None, base_url: str = None,
                 classify_model: str = None, timeout: float = 120.0):
        import httpx
        self.model_name = model_name
        self.config = config or {}
        vision_cfg = self.config.get("vision", {})
        self.temperature = float(vision_cfg.get("temperature", 0.0))
        self.top_p = float(vision_cfg.get("top_p", 1.0))
        # Model bước phân loại thô lưỡi/mặt (chỉ trả 1 từ). Ưu tiên THAM SỐ (mỗi router có tên model
        # khác nhau — vd Requesty dùng chính model_name; nếu để classify_model tên-HF sẽ 404 trên
        # Requesty), rồi vision_cfg.classify_model, cuối cùng chính model_name.
        self.classify_model = classify_model or vision_cfg.get("classify_model") or model_name
        self.fallback_client = fallback_client
        # base_url cho phép dùng chung client OpenAI-compatible cho HF / Requesty router.
        self.url = base_url or "https://router.huggingface.co/v1/chat/completions"
        self.http_client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        logger.info(f"Khởi tạo Cloud VLM client với model: {model_name}")

        # Danh sách triệu chứng mặc định (giữ tương thích giao diện với OllamaTCMClient)
        self.symptom_list = [
            "Lưỡi bệu có dấu răng", "Rêu lưỡi trắng mỏng", "Lưỡi đỏ",
            "Rêu vàng dày", "Lưỡi nhợt", "Rêu bong tróc", "Lưỡi tím",
            "Lưỡi có vết nứt", "Lưỡi sưng"
        ]
        self.face_symptom_list = [
            "Mặt đỏ", "Mặt trắng nhợt", "Mặt vàng", "Mặt xanh",
            "Mặt đen", "Mặt phù", "Mặt có ban"
        ]

    def _encode_image(self, image_path: str, max_edge: int = _MAX_IMAGE_EDGE) -> str:
        """Đọc ảnh, xoay chuẩn EXIF, thu nhỏ nếu quá lớn, trả về data URL base64 JPEG cho API."""
        try:
            import io
            from PIL import Image, ImageOps
            img = Image.open(image_path)
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            img = img.convert("RGB")
            w, h = img.size
            scale = max_edge / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            # quality 90 chứ không phải 85: vọng chẩn đọc CHUYỂN SẮC DA tinh tế (tái bệch vs hồng
            # hào, rêu mỏng vs dày), đúng thứ bị nén JPEG làm phẳng trước tiên.
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            data = buf.getvalue()
            mime = "image/jpeg"
        except Exception as e:
            logger.warning(f"PIL process error cho '{image_path}': {e} -> gửi raw bytes")
            with open(image_path, "rb") as f:
                data = f.read()
            ext = os.path.splitext(image_path)[1].lower()
            mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _chat_vision(self, system: str, prompt: str, image_path: str, max_tokens: int = 500,
                     model: str = None, max_edge: int = _MAX_IMAGE_EDGE) -> str:
        """Gọi VLM với 1 ảnh + prompt, trả về text trả lời."""
        # Một số router trả lỗi/kết quả bất ổn với temperature đúng 0.0 -> nâng nhẹ lên 0.01
        temperature = self.temperature if self.temperature > 0 else 0.01
        payload = {
            "model": model or self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._encode_image(image_path, max_edge=max_edge)}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            "temperature": temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self.http_client.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    def diagnose_image(self, image_path: str, modality: str = "tongue") -> list:
        """Gửi ảnh đến Qwen3-VL và nhận mô tả đặc điểm (tiếng Việt)."""
        if modality == "tongue":
            prompt = TONGUE_PROMPT_TEMPLATE_VI
        elif modality == "face":
            prompt = FACE_PROMPT_TEMPLATE_VI
        else:
            raise ValueError(f"Modality '{modality}' không được hỗ trợ")

        try:
            logger.info(f"Gửi ảnh {image_path} đến {self.model_name} (modality={modality})...")
            content = self._chat_vision(
                "Bạn là chuyên gia vọng chẩn Đông y. Chỉ mô tả những gì thấy rõ trong ảnh, bằng tiếng Việt.",
                prompt, image_path,
            )
            logger.info(f"{self.model_name} phân tích {modality}: {content}")
            return [content] if content else []
        except Exception as e:
            logger.error(f"Lỗi gọi Cloud VLM: {e}")
            if self.fallback_client is not None:
                # Nêu ĐÍCH DANH nơi sắp gọi. Câu cũ ghi cứng "chuyển sang LLaVA local" trong khi
                # chuỗi dự phòng thật là DashScope -> Requesty(parasail) -> Ollama, nên log báo
                # "LLaVA local" ngay trước một lượt gọi ĐÁM MÂY -> chẩn lỗi theo log sẽ đi lạc.
                _fb = getattr(self.fallback_client, "model_name", type(self.fallback_client).__name__)
                logger.warning(f"{self.model_name} lỗi -> chuyển sang dự phòng '{_fb}' để phân tích ảnh...")
                return self.fallback_client.diagnose_image(image_path, modality=modality)
            return []

    def diagnose_image_structured(self, image_path: str, modality: str = "tongue") -> dict:
        """Gọi VLM ở chế độ JSON CÓ CẤU TRÚC. Trả {'data': <dict JSON đã parse>, 'raw': <text thô>}
        hoặc None nếu lỗi/không parse được (caller tự fallback về diagnose_image prose)."""
        from src.vision_schema import parse_vlm_json
        if modality == "tongue":
            prompt = TONGUE_JSON_PROMPT_VI
        elif modality == "face":
            prompt = FACE_JSON_PROMPT_VI
        else:
            raise ValueError(f"Modality '{modality}' không được hỗ trợ")
        try:
            logger.info(f"Gửi ảnh {image_path} đến {self.model_name} (JSON, modality={modality})...")
            content = self._chat_vision(
                "Bạn là chuyên gia vọng chẩn Đông y. Chỉ trả về JSON đúng schema, không thêm chữ nào.",
                prompt, image_path,
            )
            data = parse_vlm_json(content)
            if data is None:
                logger.warning(f"VLM không trả JSON hợp lệ (modality={modality}), sẽ fallback prose. Raw: {content[:120]}")
                return self._structured_via_fallback(image_path, modality)
            logger.info(f"{self.model_name} JSON {modality}: {data}")
            return {"data": data, "raw": content}
        except Exception as e:
            logger.error(f"Lỗi gọi VLM JSON: {e}")
            return self._structured_via_fallback(image_path, modality)

    def _structured_via_fallback(self, image_path: str, modality: str):
        """Thử ĐƯỜNG JSON trên client dự phòng trước khi chịu thua về prose.

        VÌ SAO CẦN: trước đây diagnose_image_structured KHÔNG hề đụng tới fallback_client — hỏng là
        trả None, caller chuyển sang prose, và prose lại hỏng tiếp trên CHÍNH client đang chết rồi
        mới rơi xuống dự phòng. Hệ quả đo được trên log thật: DashScope chết -> parasail chỉ từng
        được gọi ở chế độ PROSE, chưa bao giờ được hỏi JSON, dù nó làm được.
        Điều đó quan trọng vì JSON là đường TẤT ĐỊNH (vision_schema map thẳng ra triệu chứng); rơi
        về prose là phải nhờ LLM khớp triệu chứng — nơi đã ghi nhận lỗi đọc câu PHỦ ĐỊNH thành
        khẳng định ('Không thấy rõ ... mặt phù/sưng ...' -> sinh ra triệu chứng 'Mặt phù').
        Nên một lượt sập của nhà cung cấp KHÔNG được phép hạ cấp luôn chất lượng vọng chẩn.
        """
        fb = getattr(self, "fallback_client", None)
        if fb is None or not hasattr(fb, "diagnose_image_structured"):
            return None
        try:
            _n = getattr(fb, "model_name", type(fb).__name__)
            logger.warning(f"{self.model_name} không cho JSON -> thử JSON trên dự phòng '{_n}'...")
            return fb.diagnose_image_structured(image_path, modality=modality)
        except Exception as e:
            logger.error(f"Dự phòng cũng không cho JSON: {e}")
            return None

    def verify_image_modality(self, image_path: str) -> str:
        """Phân loại THÔ ảnh trước khi phân tích: trả 'tongue' | 'face' | 'other' | None (lỗi).
        Cùng vai trò gác cổng như bản Ollama: chặn ảnh tải nhầm ô để prompt không ép model bịa."""
        prompt = (
            "Look at this image and classify what it PRIMARILY shows. "
            "Answer with EXACTLY ONE WORD, nothing else:\n"
            "- TONGUE : a close-up of a human tongue stuck out of the mouth\n"
            "- FACE   : a human face / head portrait (whole face visible)\n"
            "- OTHER  : anything else (hand, object, body part, unclear photo)\n"
            "Your answer (one word only):"
        )
        try:
            txt = self._chat_vision(
                "You are a strict image classifier. Reply with exactly one word: TONGUE, FACE, or OTHER.",
                prompt, image_path, max_tokens=10,
                model=self.classify_model, max_edge=_CLASSIFY_IMAGE_EDGE,
            ).lower()
            logger.info(f"Phân loại ảnh ({image_path}): '{txt[:40]}'")
            first = txt.split()[0].strip(".,:;\"'") if txt.split() else ""
            for key in ("tongue", "face", "other"):
                if first == key:
                    return key
            # fallback: ưu tiên 'tongue' (từ khoá đặc hiệu hơn 'face') rồi 'face'
            if "tongue" in txt:
                return "tongue"
            if "face" in txt or "portrait" in txt:
                return "face"
            return "other"
        except Exception as e:
            logger.error(f"Lỗi phân loại ảnh qua Cloud VLM: {e}")
            if self.fallback_client is not None:
                return self.fallback_client.verify_image_modality(image_path)
            return None   # lỗi -> fail-open, không chặn

    def set_symptom_list(self, symptom_list: list, modality: str = "tongue"):
        """Cập nhật danh sách triệu chứng"""
        if modality == "tongue":
            self.symptom_list = symptom_list
        elif modality == "face":
            self.face_symptom_list = symptom_list
        else:
            raise ValueError(f"Modality '{modality}' không được hỗ trợ")


def create_vision_client(config: dict):
    """Factory khởi tạo client vision theo chuỗi fallback:
    DashScope (qwen3-vl-32b-thinking) -> Requesty (parasail/parasail-qwen25-vl-72b-instruct) -> LLaVA local.
    """
    from src.ollama_client import OllamaTCMClient

    config = config or {}
    ollama_model = config.get("ollama", {}).get("model", "llava:7b")

    # 1. Khởi tạo LLaVA local (cuối cùng của chuỗi fallback)
    try:
        ollama_fb = OllamaTCMClient(model_name=ollama_model, config=config)
    except Exception as e:
        logger.warning(f"Không khởi tạo được LLaVA local làm fallback: {e}")
        ollama_fb = None

    # 2. Khởi tạo Requesty (giữa chuỗi fallback)
    rq_key = (os.environ.get("REQUESTY_VISION_API_KEY") 
              or os.environ.get("REQUESTY_API_KEY")
              or config.get("requesty", {}).get("vision_api_key")
              or config.get("requesty", {}).get("api_key"))
    requesty_client = None
    if rq_key:
        rq_model = "parasail/parasail-qwen25-vl-72b-instruct"
        try:
            requesty_client = CloudVLMClient(
                api_key=rq_key,
                model_name=rq_model,
                config=config,
                fallback_client=ollama_fb,
                base_url="https://router.requesty.ai/v1/chat/completions",
                classify_model=rq_model
            )
            logger.info(f"Đã thiết lập Requesty fallback: {rq_model} -> Ollama")
        except Exception as e:
            logger.warning(f"Không khởi tạo được Requesty fallback client: {e}")
            requesty_client = ollama_fb
    else:
        requesty_client = ollama_fb

    # 3. Khởi tạo DashScope VLM chính (đầu chuỗi)
    ds_key = (os.environ.get("DASHSCOPE_API_KEY") 
              or config.get("dashscope", {}).get("api_key"))
    
    if ds_key:
        ds_model = "qwen3-vl-32b-thinking"
        try:
            primary_client = CloudVLMClient(
                api_key=ds_key,
                model_name=ds_model,
                config=config,
                fallback_client=requesty_client,
                # PHẢI có đuôi /chat/completions. Trong lớp này `base_url` là ENDPOINT ĐẦY ĐỦ để
                # POST thẳng (xem Requesty ở trên: .../v1/chat/completions), KHÔNG phải base_url
                # kiểu SDK OpenAI. Thiếu đuôi -> POST vào .../compatible-mode/v1 -> 404 mọi lượt,
                # DashScope không bao giờ chạy và hệ âm thầm tụt xuống Requesty (đã đo trên log thật).
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
                classify_model=ds_model
            )
            logger.info(f"Đã thiết lập DashScope VLM chính: {ds_model} -> Requesty -> Ollama")
            return primary_client
        except Exception as e:
            logger.warning(f"Không khởi tạo được DashScope client: {e} -> dùng Requesty/Ollama.")
            return requesty_client
            
    # Nếu không có key DashScope thì dùng thẳng Requesty/Ollama
    logger.warning("Thiếu DASHSCOPE_API_KEY -> chuyển qua dùng Requesty hoặc LLaVA local làm mặc định.")
    return requesty_client
