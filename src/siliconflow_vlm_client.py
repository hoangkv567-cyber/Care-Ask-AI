# src/siliconflow_vlm_client.py
import os
import base64
import logging
from src.prompts import (
    TONGUE_PROMPT_TEMPLATE_VI, FACE_PROMPT_TEMPLATE_VI,
    TONGUE_JSON_PROMPT_VI, FACE_JSON_PROMPT_VI,
)

logger = logging.getLogger(__name__)

# Cạnh dài tối đa của ảnh gửi lên cloud. Ảnh điện thoại 4000px nén xuống 1280px
# vẫn thừa chi tiết cho vọng chẩn (LLaVA cũ chỉ nhìn 336px) nhưng payload nhỏ hơn ~10 lần.
_MAX_IMAGE_EDGE = 1280
# Bước phân loại thô (lưỡi/mặt/khác) chỉ cần nhìn tổng thể — ảnh nhỏ giảm mạnh thời gian prefill
_CLASSIFY_IMAGE_EDGE = 512


class SiliconFlowVLMClient:
    """Client vision qua SiliconFlow (Qwen3-VL), giữ nguyên giao diện của OllamaTCMClient
    (diagnose_image / verify_image_modality / set_symptom_list) để pipeline thay thế được.

    Khác biệt với LLaVA local:
    - Nhìn ảnh ở độ phân giải cao (không nén về 336px) -> thấy dấu răng, vân rêu, sắc da thật.
    - Mô tả thẳng bằng TIẾNG VIỆT -> fusion_pipeline bỏ qua được bước dịch Anh->Việt.
    - Khi gọi cloud lỗi (mất mạng, hết quota) tự chuyển về fallback_client (LLaVA local).
    """

    def __init__(self, api_key: str, model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
                 config: dict = None, fallback_client=None):
        import httpx
        self.model_name = model_name
        self.config = config or {}
        vision_cfg = self.config.get("vision", {})
        self.temperature = float(vision_cfg.get("temperature", 0.0))
        self.top_p = float(vision_cfg.get("top_p", 1.0))
        # Model nhỏ cho bước phân loại thô lưỡi/mặt (chỉ trả 1 từ, không cần 32B — tiết kiệm ~40-60s/request)
        self.classify_model = vision_cfg.get("classify_model", "Qwen/Qwen3-VL-8B-Instruct")
        self.fallback_client = fallback_client
        self.url = "https://api.siliconflow.com/v1/chat/completions"
        self.http_client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        logger.info(f"Khởi tạo SiliconFlow VLM client với model: {model_name}")

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
        """Đọc ảnh, thu nhỏ nếu quá lớn, trả về data URL base64 cho API OpenAI-compatible."""
        try:
            import io
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            scale = max_edge / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            data = buf.getvalue()
            mime = "image/jpeg"
        except ImportError:
            # Không có Pillow -> gửi nguyên gốc (Qwen3-VL nhận mọi kích thước)
            with open(image_path, "rb") as f:
                data = f.read()
            ext = os.path.splitext(image_path)[1].lower()
            mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _chat_vision(self, system: str, prompt: str, image_path: str, max_tokens: int = 500,
                     model: str = None, max_edge: int = _MAX_IMAGE_EDGE) -> str:
        """Gọi VLM với 1 ảnh + prompt, trả về text trả lời."""
        # SiliconFlow trả lỗi/kết quả bất ổn với temperature đúng 0.0 (giống SiliconFlowChatClient)
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
            logger.error(f"Lỗi gọi SiliconFlow VLM: {e}")
            if self.fallback_client is not None:
                logger.warning("Cloud VLM lỗi -> chuyển sang LLaVA local (fallback) để phân tích ảnh...")
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
                return None
            logger.info(f"{self.model_name} JSON {modality}: {data}")
            return {"data": data, "raw": content}
        except Exception as e:
            logger.error(f"Lỗi gọi VLM JSON: {e}")
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
            logger.error(f"Lỗi phân loại ảnh qua SiliconFlow VLM: {e}")
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
    """Factory chọn client vision theo config['vision']['provider']:
    - 'siliconflow': Qwen3-VL trên cloud (kèm LLaVA local làm fallback khi lỗi mạng/quota)
    - 'ollama' (hoặc không cấu hình): LLaVA local như cũ
    Thiếu API key thì tự hạ về LLaVA local thay vì chết."""
    from src.ollama_client import OllamaTCMClient

    config = config or {}
    vision_cfg = config.get("vision", {})
    provider = (vision_cfg.get("provider") or "ollama").strip().lower()
    ollama_model = config.get("ollama", {}).get("model", "llava:7b")

    if provider == "siliconflow":
        api_key = os.environ.get("SILICONFLOW_API_KEY") or config.get("siliconflow", {}).get("api_key")
        if api_key:
            model_name = vision_cfg.get("model", "Qwen/Qwen3-VL-32B-Instruct")
            try:
                fallback = OllamaTCMClient(model_name=ollama_model, config=config)
            except Exception as e:
                logger.warning(f"Không khởi tạo được LLaVA local làm fallback: {e}")
                fallback = None
            return SiliconFlowVLMClient(api_key, model_name, config=config, fallback_client=fallback)
        logger.warning("vision.provider='siliconflow' nhưng thiếu SILICONFLOW_API_KEY -> dùng LLaVA local.")

    return OllamaTCMClient(model_name=ollama_model, config=config)
