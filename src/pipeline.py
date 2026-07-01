import logging
from src.mapping import SymptomToSyndromeMapper
from src.ollama_client import OllamaTCMClient
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
        ollama_model = self.config.get("ollama", {}).get("model", "llava:7b")
        neo4j_uri = self.config.get("neo4j", {}).get("uri")
        neo4j_user = self.config.get("neo4j", {}).get("user")
        neo4j_password = self.config.get("neo4j", {}).get("password")

        self.ollama_client = OllamaTCMClient(model_name=ollama_model, config=self.config)
        self.neo4j_client = Neo4jTCMClient(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

    def run(self, tongue_image_path: str = None, face_image_path: str = None) -> dict:
        """Chạy pipeline phân tích ảnh Lưỡi và Mặt (Hỗ trợ đa phương thức)"""
        logger.info(f"Bắt đầu pipeline Vision. Lưỡi: {tongue_image_path}, Mặt: {face_image_path}")
        
        all_symptoms = []
        tongue_desc = ""
        face_desc = ""

        # Bước 1A: Xử lý ảnh Lưỡi (Nếu có)
        if tongue_image_path:
            logger.info("Đang gọi LLaVA phân tích ảnh Lưỡi...")
            tongue_symptoms = self.ollama_client.diagnose_image(tongue_image_path, modality="tongue")
            if tongue_symptoms:
                all_symptoms.extend(tongue_symptoms)
                tongue_desc = tongue_symptoms[0]

        # Bước 1B: Xử lý ảnh Mặt (Nếu có)
        if face_image_path:
            logger.info("Đang gọi LLaVA phân tích ảnh Mặt...")
            face_symptoms = self.ollama_client.diagnose_image(face_image_path, modality="face")
            if face_symptoms:
                all_symptoms.extend(face_symptoms)
                face_desc = face_symptoms[0]

        # Loại bỏ các triệu chứng bị trùng lặp (nếu cả 2 ảnh đều báo giống nhau)
        all_symptoms = list(set(all_symptoms))

        if not all_symptoms:
            logger.warning("Không phát hiện triệu chứng nào từ (các) ảnh được cung cấp.")
            return {
                "error": "Không thể xác định triệu chứng",
                "detected_symptoms": [],
                "analysis": "",  # Trả về rỗng để Fusion Pipeline không bị lỗi
                "tongue_description": "",
                "face_description": ""
            }

        # Bước 2: Tạo chuỗi phân tích chuẩn bị cho Tứ chẩn hợp tham
        # Chuyển list ['rêu trắng dày', 'mặt nhợt'] thành chuỗi "rêu trắng dày, mặt nhợt"
        analysis_text = ", ".join(all_symptoms)
        logger.info(f"LLaVA đã nhìn thấy: {analysis_text}")

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
            "analysis": analysis_text  # <--- Key này sẽ được Fusion Pipeline bốc ra ghép vào câu hỏi
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
