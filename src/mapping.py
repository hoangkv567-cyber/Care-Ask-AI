import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SymptomToSyndromeMapper:
    def __init__(self, mapping_file=None):
        self.mapping = {}
        if mapping_file:
            # Nếu mapping_file là một list các đường dẫn/chuỗi
            if isinstance(mapping_file, list):
                files = mapping_file
            # Nếu mapping_file là thư mục, quét tất cả các file JSON bên trong
            elif Path(mapping_file).is_dir():
                files = list(Path(mapping_file).glob("*.json"))
            else:
                files = [mapping_file]
                
            for file_path in files:
                if Path(file_path).exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                self.mapping.update(data)
                                logger.info(f"Đã tải thành công mapping từ {file_path} (chứa {len(data)} key)")
                    except Exception as e:
                        logger.error(f"Lỗi khi tải file mapping {file_path}: {e}")
        else:
            self.mapping = {}
    
    def map_symptoms_to_syndromes(self, symptom_list):
        syndromes = set()
        for symptom in symptom_list:
            if symptom in self.mapping:
                syndromes.update(self.mapping[symptom])
            else:
                logger.warning(f"Chưa có mapping cho: {symptom}")
        return list(syndromes)

    def add_mapping(self, symptom: str, syndromes: list):
        """Thêm mapping mới (dùng khi có thêm dữ liệu)"""
        if symptom not in self.mapping:
            self.mapping[symptom] = []
        for s in syndromes:
            if s not in self.mapping[symptom]:
                self.mapping[symptom].append(s)

    def save_mapping(self, mapping_file: str):
        """Lưu mapping hiện tại ra file JSON"""
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu mapping vào {mapping_file}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mapping vào {mapping_file}: {e}")
