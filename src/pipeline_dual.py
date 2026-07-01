import logging
from src.mapping import SymptomToSyndromeMapper
from src.ollama_client import OllamaTCMClient
from src.neo4j_client import Neo4jTCMClient
from src.utils import logger

class TCMTongueFacePipeline:
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Load mapping configurations
        symptom_mapping = self.config.get("mapping", {}).get("symptom_to_syndrome", "data/mapping/symptom_to_syndrome.json")
        face_mapping = self.config.get("mapping", {}).get("face_to_syndrome", "data/mapping/face_to_syndrome.json")
        
        self.mapper_tongue = SymptomToSyndromeMapper(symptom_mapping)
        self.mapper_face = SymptomToSyndromeMapper(face_mapping)
        
        # Load connection configuration parameters
        ollama_model = self.config.get("ollama", {}).get("model", "llava:7b")
        neo4j_uri = self.config.get("neo4j", {}).get("uri")
        neo4j_user = self.config.get("neo4j", {}).get("user")
        neo4j_password = self.config.get("neo4j", {}).get("password")
        
        self.ollama_client = OllamaTCMClient(model_name=ollama_model)
        self.neo4j_client = Neo4jTCMClient(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

    def run(self, image_tongue_path: str, image_face_path: str) -> dict:
        logger.info(f"Bắt đầu chạy pipeline kết hợp: Lưỡi={image_tongue_path}, Mặt={image_face_path}")
        
        # Chẩn đoán lưỡi
        symptoms_tongue = self.ollama_client.diagnose_image(image_tongue_path, modality="tongue")
        syndromes_tongue = self.mapper_tongue.map_symptoms_to_syndromes(symptoms_tongue)
        treatments_tongue = []
        for s in syndromes_tongue:
            t = self.neo4j_client.get_treatment_by_syndrome(s)
            if t: 
                treatments_tongue.append(t)
        
        # Chẩn đoán mặt
        symptoms_face = self.ollama_client.diagnose_image(image_face_path, modality="face")
        syndromes_face = self.mapper_face.map_symptoms_to_syndromes(symptoms_face)
        treatments_face = []
        for s in syndromes_face:
            t = self.neo4j_client.get_treatment_by_syndrome(s)
            if t: 
                treatments_face.append(t)

        # Tổng hợp hội chứng chung (lấy hợp của hai tập hội chứng)
        all_syndromes = list(set(syndromes_tongue) | set(syndromes_face))
        
        # Lấy duy nhất danh sách các bài thuốc để tránh trùng lặp
        combined_treatments = []
        seen_prescriptions = set()
        for t in (treatments_tongue + treatments_face):
            p_name = t.get("bai_thuoc")
            if p_name not in seen_prescriptions:
                seen_prescriptions.add(p_name)
                combined_treatments.append(t)
        
        return {
            "tongue": {
                "symptoms": symptoms_tongue,
                "syndromes": syndromes_tongue,
                "treatments": treatments_tongue
            },
            "face": {
                "symptoms": symptoms_face,
                "syndromes": syndromes_face,
                "treatments": treatments_face
            },
            "combined": {
                "all_syndromes": all_syndromes,
                "treatments": combined_treatments
            }
        }

    def close(self):
        self.neo4j_client.close()
