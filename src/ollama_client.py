import ollama
import json
import logging
from src.prompts import TONGUE_PROMPT_TEMPLATE, FACE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class OllamaTCMClient:
    def __init__(self, model_name="llava:7b", config=None):
        self.model_name = model_name
        self.config = config or {}
        # Lấy tham số từ config, mặc định an toàn
        self.temperature = float(self.config.get("temperature", 0.0))  # Mặc định 0
        self.top_p = float(self.config.get("top_p", 1.0))
        self.seed = int(self.config.get("seed", 42))  # Thêm seed để ổn định hoàn toàn
        
        # Hỗ trợ host remote
        self.host = self.config.get("host") or self.config.get("ollama", {}).get("host")
        if self.host:
            from ollama import Client
            self.client = Client(host=self.host)
            logger.info(f"Khởi tạo Ollama Client kết nối remote host: {self.host}")
        else:
            import ollama
            self.client = ollama
            logger.info(f"Khởi tạo Ollama client với model: {model_name}, temp={self.temperature}, seed={self.seed}")
            
        # Danh sách triệu chứng lưỡi và mặt mặc định (bắt buộc để không bị lỗi Attribute lỗi)
        self.symptom_list = [
            "Lưỡi bệu có dấu răng", "Rêu lưỡi trắng mỏng", "Lưỡi đỏ",
            "Rêu vàng dày", "Lưỡi nhợt", "Rêu bong tróc", "Lưỡi tím",
            "Lưỡi có vết nứt", "Lưỡi sưng"
        ]
        self.face_symptom_list = [
            "Mặt đỏ", "Mặt trắng nhợt", "Mặt vàng", "Mặt xanh",
            "Mặt đen", "Mặt phù", "Mặt có ban"
        ]

    def diagnose_image(self, image_path: str, modality: str = "tongue") -> list:
        """Gửi ảnh đến LLaVA và nhận mô tả đặc điểm"""
        if modality == "tongue":
            prompt = TONGUE_PROMPT_TEMPLATE
        elif modality == "face":
            prompt = FACE_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Modality '{modality}' không được hỗ trợ")
        
        try:
            logger.info(f"Gửi ảnh {image_path} đến LLaVA (modality={modality})...")
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a TCM expert."},
                    {"role": "user", "content": prompt, "images": [image_path]}
                ],
                options={  # Truyền các tuỳ chọn vào options
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "seed": self.seed
                }
            )
            
            content = response['message']['content'].strip()
            logger.info(f"LLaVA phân tích {modality}: {content}")
            return [content]
        except Exception as e:
            logger.error(f"Lỗi khi gọi LLaVA: {e}")
            return []

    def set_symptom_list(self, symptom_list: list, modality: str = "tongue"):
        """Cập nhật danh sách triệu chứng"""
        if modality == "tongue":
            self.symptom_list = symptom_list
        elif modality == "face":
            self.face_symptom_list = symptom_list
        else:
            raise ValueError(f"Modality '{modality}' không được hỗ trợ")
