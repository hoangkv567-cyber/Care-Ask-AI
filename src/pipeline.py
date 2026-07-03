import logging
from src.mapping import SymptomToSyndromeMapper
from src.siliconflow_vlm_client import create_vision_client
from src.neo4j_client import Neo4jTCMClient
from src.utils import logger
from src.config_loader import load_config

class TCMTonguePipeline:
    def __init__(self, config: dict = None, modality: str = "tongue"):
        """
        Khởi tạo pipeline
        config: dict chứa cấu hình (ollama_model, neo4j_uri, neo4j_user, neo4j_password, mapping_file)
        modality: str ("tongue" hoặc "face")
        """
        self.modality = modality
        self.config = config or load_config()
        
        # Tự động quét và tải toàn bộ thư mục mapping để hỗ trợ song song cả Lưỡi và Mặt
        import os
        mapping_dir = "data/mapping"
        if os.path.exists(mapping_dir) and os.path.isdir(mapping_dir):
            self.mapper = SymptomToSyndromeMapper(mapping_dir)
        else:
            # Fallback về một file đơn nếu không có thư mục
            if modality == "tongue":
                mapping_file = self.config.get("mapping", {}).get("symptom_to_syndrome", "data/mapping/symptom_to_syndrome.json")
            elif modality == "face":
                mapping_file = self.config.get("mapping", {}).get("face_to_syndrome", "data/mapping/face_to_syndrome.json")
            else:
                raise ValueError(f"Modality '{modality}' không được hỗ trợ")
            self.mapper = SymptomToSyndromeMapper(mapping_file)
        
        # Lấy các tham số cấu hình dạng lồng nhau (secrets đã được config_loader bơm từ .env)
        neo4j_uri = self.config.get("neo4j", {}).get("uri")
        neo4j_user = self.config.get("neo4j", {}).get("user")
        neo4j_password = self.config.get("neo4j", {}).get("password")

        # Client vision chọn theo config['vision']['provider'] (Qwen3-VL cloud hoặc LLaVA local).
        # Giữ tên thuộc tính 'ollama_client' để không phá vỡ code cũ đang tham chiếu.
        self.ollama_client = create_vision_client(self.config)
        self.neo4j_client = Neo4jTCMClient(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

    def run(self, tongue_image_path: str = None, face_image_path: str = None) -> dict:
        """Chạy pipeline phân tích ảnh Lưỡi và Mặt (Hỗ trợ đa phương thức)"""
        logger.info(f"Bắt đầu pipeline Vision. Lưỡi: {tongue_image_path}, Mặt: {face_image_path}")
        
        all_symptoms = []
        tongue_desc = ""
        face_desc = ""
        vision_warnings = {}   # {'tongue'|'face': thông báo} khi tải nhầm loại ảnh

        _LABEL_VI = {"tongue": "ảnh lưỡi", "face": "ảnh khuôn mặt", "other": "ảnh không xác định (không phải lưỡi/mặt)"}

        def _process_image(modality: str, image_path: str):
            """Xử lý trọn 1 ảnh: GÁC CỔNG phân loại rồi phân tích. Trả (desc, warning)."""
            if modality == "tongue":
                wrong_kinds, slot_vi, hint_vi = ("face", "other"), "LƯỠI", "Vui lòng tải ảnh cận cảnh LƯỠI (thè lưỡi ra)."
            else:
                wrong_kinds, slot_vi, hint_vi = ("tongue", "other"), "KHUÔN MẶT", "Vui lòng tải ảnh KHUÔN MẶT chính diện."
            detected = self.ollama_client.verify_image_modality(image_path)
            if detected in wrong_kinds:          # sai ô rõ ràng -> KHÔNG bịa, chỉ cảnh báo
                logger.warning(f"Ảnh ô {slot_vi} bị từ chối (phân loại: {detected}).")
                return "", (
                    f"⚠️ Ảnh ở ô {slot_vi} có vẻ không phải {_LABEL_VI.get(modality)} "
                    f"(AI nhận thấy: {_LABEL_VI.get(detected, detected)}). "
                    f"Đã bỏ qua để tránh kết quả sai. {hint_vi}"
                )
            # đúng loại hoặc None (lỗi phân loại) -> cho qua
            logger.info(f"Đang gọi mô hình vision phân tích ảnh {slot_vi.capitalize()}...")
            symptoms = self.ollama_client.diagnose_image(image_path, modality=modality)
            return (symptoms[0] if symptoms else ""), None

        # Bước 1: Xử lý ảnh Lưỡi và Mặt SONG SONG (2 ảnh độc lập — chạy tuần tự lãng phí ~45s
        # vì mỗi lượt phân tích cloud mất 40-60s; song song thì tổng = lượt chậm nhất)
        from concurrent.futures import ThreadPoolExecutor
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if tongue_image_path:
                futures["tongue"] = executor.submit(_process_image, "tongue", tongue_image_path)
            if face_image_path:
                futures["face"] = executor.submit(_process_image, "face", face_image_path)

        if "tongue" in futures:
            tongue_desc, warn = futures["tongue"].result()
            if warn:
                vision_warnings["tongue"] = warn
            elif tongue_desc:
                all_symptoms.append(tongue_desc)
        if "face" in futures:
            face_desc, warn = futures["face"].result()
            if warn:
                vision_warnings["face"] = warn
            elif face_desc:
                all_symptoms.append(face_desc)

        # Loại bỏ các triệu chứng bị trùng lặp (nếu cả 2 ảnh đều báo giống nhau)
        all_symptoms = list(set(all_symptoms))

        if not all_symptoms:
            logger.warning("Không phát hiện triệu chứng nào từ (các) ảnh được cung cấp.")
            return {
                "error": "Không thể xác định triệu chứng",
                "detected_symptoms": [],
                "analysis": "",  # Trả về rỗng để Fusion Pipeline không bị lỗi
                "tongue_description": "",
                "face_description": "",
                "vision_warnings": vision_warnings
            }

        # Bước 2: Tạo chuỗi phân tích chuẩn bị cho Tứ chẩn hợp tham
        # Chuyển list ['rêu trắng dày', 'mặt nhợt'] thành chuỗi "rêu trắng dày, mặt nhợt"
        analysis_text = ", ".join(all_symptoms)
        logger.info(f"Mô hình vision đã nhìn thấy: {analysis_text}")

        # Bước 3: Ánh xạ hội chứng & Bài thuốc (Giữ lại logic cũ để hệ thống không bị phá vỡ cấu trúc)
        syndromes = self.mapper.map_symptoms_to_syndromes(all_symptoms)
        treatments = []
        if syndromes:
            for syndrome in syndromes:
                treatment = self.neo4j_client.get_treatment_by_syndrome(syndrome)
                if treatment:
                    treatments.append(treatment)

        # Bước 4: Đóng gói kết quả (Thêm key 'analysis' quan trọng)
        result = {
            "tongue_description": tongue_desc,
            "face_description": face_desc,
            "detected_symptoms": all_symptoms,
            "possible_syndromes": syndromes,
            "treatments": treatments,
            "analysis": analysis_text,  # <--- Key này sẽ được Fusion Pipeline bốc ra ghép vào câu hỏi
            "vision_warnings": vision_warnings
        }
        
        logger.info("Pipeline Vision hoàn tất!")
        return result

    def close(self):
        """Đóng kết nối Neo4j"""
        self.neo4j_client.close()

    def update_mapping(self, symptom: str, syndromes: list):
        """Cập nhật mapping (dùng khi có thêm dữ liệu)"""
        self.mapper.add_mapping(symptom, syndromes)
        
        if self.modality == "tongue":
            mapping_file = self.config.get("mapping", {}).get("symptom_to_syndrome", "data/mapping/symptom_to_syndrome.json")
        else:
            mapping_file = self.config.get("mapping", {}).get("face_to_syndrome", "data/mapping/face_to_syndrome.json")
            
        self.mapper.save_mapping(mapping_file)
