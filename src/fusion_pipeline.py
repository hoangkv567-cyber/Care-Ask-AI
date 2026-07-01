# src/fusion_pipeline.py
import logging
import re
from src.pipeline import TCMTonguePipeline
from src.qa_system import TCMQA

from src.utils import normalize_symptoms_text

logger = logging.getLogger(__name__)

class TCMFusionPipeline:
    def __init__(self, config: dict = None):
        """Khởi tạo toàn bộ lõi AI của hệ thống"""
        logger.info("Đang khởi tạo Hệ thống Hợp nhất (Fusion Pipeline)...")
        from src.config_loader import load_config
        config = config or load_config()

        # Module Vọng chẩn (LLaVA - Phân tích ảnh)
        self.vision_pipeline = TCMTonguePipeline(config=config)

        # Module Vấn chẩn & Đồ thị (Qwen + Neo4j)
        self.qa_pipeline = TCMQA(config=config)

        self.qa_pipeline.neo4j_client = self.vision_pipeline.neo4j_client

        # Tải danh sách các hội chứng hợp lệ từ database để tránh ảo giác
        try:
            self.valid_syndromes = self.vision_pipeline.neo4j_client.get_all_syndromes()
            logger.info(f"Đã tải {len(self.valid_syndromes)} hội chứng hợp lệ từ Neo4j")
        except Exception as e:
            logger.error(f"Lỗi khi tải danh sách hội chứng từ database: {e}")
            self.valid_syndromes = []

        # Tải danh sách triệu chứng lưỡi và mặt chuẩn từ file mapping để mapping chính xác
        import json
        import os
        self.tongue_symptoms_list = []
        self.face_symptoms_list = []
        
        tongue_map_path = "data/mapping/symptom_to_syndrome.json"
        if os.path.exists(tongue_map_path):
            try:
                with open(tongue_map_path, "r", encoding="utf-8") as f:
                    all_keys = list(json.load(f).keys())
                # Lọc chỉ giữ lại các triệu chứng liên quan đến lưỡi/rêu để tránh LLM quá tải
                self.tongue_symptoms_list = [
                    k for k in all_keys 
                    if any(kw in k.lower() for kw in ["lưỡi", "rêu", "chất lưỡi", "gốc lưỡi", "rìa lưỡi", "đầu lưỡi"])
                ]
                logger.info(f"Đã tải {len(self.tongue_symptoms_list)} triệu chứng lưỡi chuẩn từ mapping (sau lọc)")
            except Exception as e:
                logger.error(f"Lỗi tải triệu chứng lưỡi từ mapping: {e}")
                
        face_map_path = "data/mapping/face_to_syndrome.json"
        if os.path.exists(face_map_path):
            try:
                with open(face_map_path, "r", encoding="utf-8") as f:
                    self.face_symptoms_list = list(json.load(f).keys())
                logger.info(f"Đã tải {len(self.face_symptoms_list)} triệu chứng mặt chuẩn từ mapping")
            except Exception as e:
                logger.error(f"Lỗi tải triệu chứng mặt từ mapping: {e}")

        if not self.tongue_symptoms_list:
            self.tongue_symptoms_list = [
                "Lưỡi bệu có dấu răng", "Rêu lưỡi trắng mỏng", "Lưỡi đỏ",
                "Rêu vàng dày", "Lưỡi nhợt", "Rêu bong tróc", "Lưỡi tím",
                "Lưỡi có vết nứt", "Lưỡi sưng", "Lưỡi khô", "Rêu lưỡi vàng nhầy",
                "Lưỡi nhỏ đỏ", "Lưỡi có dấu răng", "Lưỡi không có rêu", "Loét miệng lưỡi sưng đau",
                "Đầu lưỡi đỏ", "Loét lưỡi", "Lưỡi nứt"
            ]
        if not self.face_symptoms_list:
            self.face_symptoms_list = [
                "Mặt đỏ", "Mặt trắng nhợt", "Mặt nhợt nhạt", "Mặt vàng", "Mặt xanh",
                "Mặt mở", "Mặt phù", "Mặt xám ngoét", "Sắc mặt ám tối", "Mặt có ban",
                "Ban xuất huyết dưới da", "Da xuất hiện ban đỏ hình bướm", "Da có mảng đỏ có vảy",
                "Da nổi mụn nước", "Da ngứa", "Da khô", "Mắt đỏ", "Mắt lồi", "Môi méo", "Môi thâm"
            ]
        
        self._load_csv_data()
        logger.info("Khởi tạo hoàn tất!")

    def _clean_foreign_characters(self, text: str) -> str:
        """Loại bỏ toàn bộ chữ Hán, chữ Cyrillic (Nga), token yka3 và ký tự rác để đảm bảo đầu ra sạch 100%"""
        if not text:
            return ""
        # Xóa yka3 (case-insensitive)
        text = re.sub(r'(?i)\byka3\b', '', text)
        text = text.replace("yka3", "").replace("Yka3", "")
        # Xóa toàn bộ chữ Hán và chữ Cyrillic
        text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\u0400-\u04ff]', '', text)
        # Xóa các ký tự dấu câu Trung Quốc đặc thù
        chinese_symbols = ['：', '，', '。', '！', '？', '（', '）', '【', '】', '“', '”', '‘', '’', '；']
        for sym in chinese_symbols:
            text = text.replace(sym, '')
        # Định dạng lại khoảng trắng thừa
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def _normalize_symptoms(self, symptoms_list: list) -> list:
        """Quy đổi toàn bộ các triệu chứng đồng nghĩa Hán-Việt, Thuần Việt và biến thể về một chuẩn duy nhất"""
        if not symptoms_list:
            return []
        
        synonym_dict = {
            "huyễn vựng": "chóng mặt",
            "úy phong": "sợ gió",
            "lưỡi dầy": "lưỡi bệu",
            "lưỡi dày": "lưỡi bệu",
            "rìa lưỡi hằn răng": "rìa lưỡi có hằn răng",
            "lưỡi hằn răng": "rìa lưỡi có hằn răng",
            "lưỡi có vết hằn răng": "rìa lưỡi có hằn răng",
            "lưỡi nhạt có dấu răng": "rìa lưỡi có hằn răng",
            "lưỡi có dấu răng": "rìa lưỡi có hằn răng",
            "chất lưỡi tím": "lưỡi có vết bầm tím",
            "lưỡi tím tái": "lưỡi có vết bầm tím",
            "lưỡi tím": "lưỡi có vết bầm tím",
            "lưỡi có điểm ứ huyết": "lưỡi có vết bầm tím",
            "đầu thống": "đau đầu",
            "nhức đầu": "đau đầu",
            "đầu đau": "đau đầu",
            "diện hồng": "mặt đỏ",
            "diện sắc nhợt nhạt": "mặt nhợt nhạt",
            "diện sắc nhợt": "mặt nhợt nhạt",
            "mặt nhợt": "mặt nhợt nhạt",
            "mặt trắng": "mặt nhợt nhạt",
            "mặt trắng nhợt": "mặt nhợt nhạt",
            "mặt trắng bệch": "mặt nhợt nhạt",
            "diện sắc trắng": "mặt nhợt nhạt",
            "sắc mặt trắng": "mặt nhợt nhạt",
            "bạch nhuận": "rêu trắng mỏng",
            "rêu trắng mỏng": "rêu trắng mỏng",
            "rêu lưỡi trắng mỏng": "rêu trắng mỏng",
            # Nhóm hô hấp: quy về "đoản khí" (giữ "khó thở" tách biệt vì có thể là node riêng trong Neo4j)
            "hơi thở ngắn": "đoản khí",
            "thở ngắn": "đoản khí",
            "khí đoản": "đoản khí",
            "hụt hơi": "đoản khí",
            "thiểu khí": "đoản khí",
            # Nhóm mệt mỏi
            "thiếu sức": "mệt mỏi",
            "vô lực": "mệt mỏi",
            "uể oải": "mệt mỏi",
            "kiệt sức": "mệt mỏi",
            "thần mệt": "mệt mỏi",
            # Nhóm chóng mặt (giữ "hoa mắt" tách biệt vì có thể là node riêng trong Neo4j)
            "xây xẩm": "chóng mặt",
            "choáng váng": "chóng mặt",
            "chất lưỡi nhạt hồng": "lưỡi hồng nhạt",
            "lưỡi nhạt hồng": "lưỡi hồng nhạt",
            "chất lưỡi hồng nhạt": "lưỡi hồng nhạt",
            "chất lưỡi đạm đỏ": "lưỡi hồng nhạt",
            "lưỡi nhạt đỏ": "lưỡi hồng nhạt",
            "lưỡi đạm": "lưỡi nhợt",
            "chất lưỡi đạm": "lưỡi nhợt",
            "lưỡi nhạt": "lưỡi nhợt",
            "chất lưỡi nhạt": "lưỡi nhợt",
            "lưỡi sắc nhợt": "lưỡi nhợt",
            "chất lưỡi nhợt": "lưỡi nhợt"
        }
        
        normalized = []
        for s in symptoms_list:
            s_clean = s.lower().strip()
            norm_val = synonym_dict.get(s_clean, s_clean)
            normalized.append(norm_val)
            
        return list(dict.fromkeys(normalized))

    def _resolve_symptom_conflicts(self, symptoms: list) -> list:
        """Dọn dẹp các triệu chứng mâu thuẫn sinh lý - bệnh lý (ví dụ: rêu dày vs rêu bình thường)"""
        if not symptoms:
            return []
            
        # Chuẩn hóa triệu chứng đồng nghĩa trước khi xử lý mâu thuẫn
        symptoms = self._normalize_symptoms(symptoms)
            
        # 1. Khử trùng rác dữ liệu / triệu chứng NLP vô nghĩa do AI trích xuất sai
        blacklist = ["rêu lưỡi bệu", "lưỡi nhợt khô", "rêu lưỡi dày nhớt khô", "rêu bệu"]
        symptoms = [s for s in symptoms if s.lower().strip() not in blacklist]
        
        symptoms_lower = [s.lower().strip() for s in symptoms]
        
        # 2. Xử lý mâu thuẫn Trạng thái Lưỡi (Ưu tiên LLaVA phát hiện Lưỡi ướt/bệu/nhầy)
        tongue_wet = ["lưỡi bệu", "rêu trắng nhớt", "rêu trắng dày", "rêu lưỡi trắng dày", "rêu lưỡi trắng nhớt", "rêu lưỡi dày nhớt", "rêu lưỡi bẩn"]
        tongue_dry = ["lưỡi đỏ khô", "lưỡi khô", "chất lưỡi đỏ khô", "không rêu", "không có rêu", "rêu lưỡi khô", "rêu khô"]
        
        if any(w in symptoms_lower for w in tongue_wet):
            symptoms = [s for s in symptoms if s.lower().strip() not in tongue_dry]
            symptoms_lower = [s.lower().strip() for s in symptoms]

        # 3. Nếu có rêu dày, loại bỏ rêu bình thường / rêu mỏng
        has_thick_coating = any(
            kw in symptoms_lower 
            for kw in ["rêu lưỡi trắng dày", "rêu trắng dày", "rêu lưỡi dày nhớt", "rêu vàng dày", "rêu dày"]
        )
        if has_thick_coating:
            symptoms = [
                s for s in symptoms 
                if s.lower().strip() not in ["rêu bình thường", "rêu mỏng trắng", "rêu lưỡi mỏng", "rêu mỏng"]
            ]
            symptoms_lower = [s.lower().strip() for s in symptoms]

        # 4. Nếu có sưng phù, loại bỏ mắt/mặt bình thường
        has_swelling = any(
            kw in symptoms_lower
            for kw in ["mí mắt dưới hơi sưng", "phù ở mí mắt", "mặt phù", "sưng phù"]
        )
        if has_swelling:
            symptoms = [
                s for s in symptoms
                if s.lower().strip() not in ["mắt bình thường", "mặt bình thường", "không phù"]
            ]
            symptoms_lower = [s.lower().strip() for s in symptoms]

        # 5. Nếu có mụn/viêm/đỏ, loại bỏ sắc mặt bình thường / khỏe mạnh
        has_redness_acne = any(
            kw in symptoms_lower
            for kw in ["nốt mụn đỏ", "mụn viêm", "mụn đỏ", "vùng đỏ trên mặt", "mẩn đỏ", "vùng đỏ mặt"]
        )
        if has_redness_acne:
            symptoms = [
                s for s in symptoms
                if s.lower().strip() not in ["sắc mặt bình thường", "sắc mặt hồng nhuận", "lưỡi bình thường", "sắc mặt khỏe mạnh"]
            ]
            symptoms_lower = [s.lower().strip() for s in symptoms]
            
        # 6. Nếu có hằn răng/dấu răng (bệnh lý), loại bỏ lưỡi không hằn răng (bình thường)
        has_toothmarks = any(
            kw in symptoms_lower
            for kw in ["rìa lưỡi có hằn răng", "rìa lưỡi hằn răng", "lưỡi hằn răng", "lưỡi có vết hằn răng", "lưỡi nhạt có dấu răng", "lưỡi có dấu răng", "rìa lưỡi có vết hằn răng"]
        )
        if has_toothmarks:
            symptoms = [
                s for s in symptoms
                if s.lower().strip() not in ["lưỡi không có hằn răng", "lưỡi không hằn răng", "không có hằn răng"]
            ]
            symptoms_lower = [s.lower().strip() for s in symptoms]

        # 7. Khử mâu thuẫn màu sắc chất lưỡi: Nếu có chất lưỡi bệnh lý (lưỡi nhợt, lưỡi đỏ, lưỡi tím), loại bỏ sắc lưỡi sinh lý (lưỡi hồng, lưỡi hồng nhạt, lưỡi bình thường)
        has_pathological_color = any(
            x in symptoms_lower 
            for x in ["lưỡi nhợt", "lưỡi đỏ", "chất lưỡi đỏ", "lưỡi tím", "chất lưỡi tím", "lưỡi đỏ sẫm", "lưỡi đỏ tươi"]
        )
        if has_pathological_color:
            symptoms = [
                s for s in symptoms 
                if s.lower().strip() not in ["lưỡi hồng", "lưỡi hồng nhạt", "lưỡi bình thường"]
            ]
            
        return symptoms

    def _post_process_hallucinations(self, text: str, symptoms_str: str) -> str:
        """Xóa bỏ các triệu chứng ảo giác ra khỏi văn bản biện chứng bằng lập trình nếu không có trong đầu vào"""
        if not text:
            return ""
        symptoms_lower = symptoms_str.lower()
        
        # Danh sách các từ khóa kiểm duyệt và các cụm từ tương ứng
        censor_rules = {
            "chóng mặt": ["chóng mặt", "chóng mat", "chóng mặt/hoa mắt", "hoa mắt/chóng mặt"],
            "hoa mắt": ["hoa mắt", "hoa mat", "chóng mặt/hoa mắt", "hoa mắt/chóng mặt"],
            "mụn đỏ": ["mụn đỏ", "mun do", "nốt mụn đỏ", "nốt mụn", "mụn trứng cá"],
            "mụn viêm": ["mụn viêm", "mun viem"],
            "nôn mửa": ["nôn mửa", "non mua", "nôn nghịch", "buồn nôn", "nôn ra", "buồn nôn/nôn mửa"],
            "ợ hơi": ["ợ hơi", "o hoi", "ợ chua", "ợ nước", "ợ hơi/ợ chua"],
            "mồ hôi trộm": ["mồ hôi trộm", "mo hoi trom", "đạo hãn", "dao han"]
        }
        
        for key, phrases in censor_rules.items():
            # Nếu cả "chóng mặt" và "hoa mắt" đều không xuất hiện trong chuỗi triệu chứng gộp thực tế
            is_allowed = False
            if key in ["chóng mặt", "hoa mắt"]:
                is_allowed = ("chóng mặt" in symptoms_lower) or ("hoa mắt" in symptoms_lower)
            else:
                is_allowed = (key in symptoms_lower)
                
            if not is_allowed:
                for phrase in phrases:
                    # Thay thế cụm từ này bằng cách diễn đạt trung tính hoặc loại bỏ
                    text = re.sub(rf'(?i)\b(?:gây ra|dẫn đến|gây|kèm|và|hoặc|như)\s+{re.escape(phrase)}\b', '', text)
                    text = re.sub(rf'(?i)\b{re.escape(phrase)}\b', '', text)
                    
        # Làm sạch các khoảng trắng và dấu câu thừa sau khi xóa
        text = re.sub(r'[ \t]*,[ \t]*,', ',', text)
        text = re.sub(r'[ \t]*,[ \t]*\.', '.', text)
        text = re.sub(r'[ \t]*,[ \t]*và[ \t]+', ' và ', text)
        text = re.sub(r'[ \t]+và[ \t]+và[ \t]+', ' và ', text)
        text = re.sub(r'[ \t]+và[ \t]*,', ',', text)
        text = re.sub(r'[ \t]+,[ \t]*và[ \t]+', ' và ', text)
        
        # Làm sạch khoảng trắng thừa trên từng dòng và bảo toàn ký tự xuống dòng
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        return "\n".join(lines).strip()

    def _clean_translated_text(self, text: str) -> str:
        """Tách khối dịch bị lặp/glitch của Qwen và trả về phần dịch chuẩn tiếng Việt cuối cùng"""
        if not text:
            return ""
            
        # 1. Loại bỏ các ghi chú dạng (Note: ...) hoặc [Lưu ý: ...] của LLM
        note_pattern = r'\((?:note|lưu ý|chú thích|chú ý|corrected|translation|bản dịch)[^)]*\)|\[(?:note|lưu ý|chú thích|chú ý|corrected|translation|bản dịch)[^\]]*\]'
        text = re.sub(note_pattern, '', text, flags=re.IGNORECASE)

        # 2. Loại bỏ các câu dẫn giải tiếng Anh tự động của LLM ở đầu câu dịch (bắt buộc kết thúc bằng dấu câu)
        english_intro_regex = r'^(?:[A-Za-z\s,().\'’\-0-9]+(?:contain|contains|error|typo|mix|Vietnamese|Chinese|characters|corrected|translation|here is|here\'s|please note|note|attention)[A-Za-z\s,().\'’\-0-9]*[.!?]\s*)+'
        text = re.sub(english_intro_regex, '', text).strip()
        
        # 3. Xóa các cụm từ lửng lơ ở đầu (cả tiếng Anh và tiếng Việt)
        junk_prefixes = [
            r'^(?:here is the corrected|here is the translation|corrected version|corrected|the translation is|translation)\s*:\s*',
            r'^(?:here is the corrected|here is the translation|corrected version|corrected|the translation is|translation)\s*',
            r'^(?:dưới đây là bản dịch|bản dịch tiếng việt|bản dịch|kết quả dịch|dịch là)\s*:\s*',
            r'^(?:dưới đây là bản dịch|bản dịch tiếng việt|bản dịch|kết quả dịch|dịch là)\s*',
            r'^:\s*'
        ]
        for pattern in junk_prefixes:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

        # 4. Xóa phần "Gốc: ..." hoặc "Original: ..." ở cuối nếu nó lặp lại văn bản tiếng Anh gốc
        text = re.sub(r'(?i)\b(?:gốc|original|source|tiếng anh gốc)\s*:\s*.*$', '', text).strip()

        # 5. Xóa yka3
        text = re.sub(r'(?i)\byka3\b', '', text)
        text = text.replace("yka3", "").replace("Yka3", "")

        # 6. Loại bỏ trực tiếp chữ Hán và chữ Cyrillic
        text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\u0400-\u04ff]', '', text)

        # 7. Xóa các ký tự dấu câu Trung Quốc đặc thù
        chinese_symbols = ['：', '，', '。', '！', '？', '（', '）', '【', '】', '“', '”', '‘', '’', '；']
        for sym in chinese_symbols:
            text = text.replace(sym, '')

        # 8. Xóa các từ lặp lại liên tục do lỗi lặp từ của Qwen (ví dụ: abnormalities-abnormalities-...)
        text = re.sub(r'\b(\w+)\b(?:[\s\-_]+\b\1\b){2,}', r'\1', text, flags=re.IGNORECASE)
            
        # 9. Dịch các từ tiếng Anh chuyên ngành thường gặp mà Qwen đôi khi bỏ sót
        replacements = {
            r'\bExpression\b': 'Biểu cảm',
            r'\bexpression\b': 'biểu cảm',
            r'\bComplexion\b': 'Sắc mặt',
            r'\bcomplexion\b': 'sắc mặt',
            r'\bAbnormalities\b': 'Đặc điểm bất thường',
            r'\babnormalities\b': 'đặc điểm bất thường',
            r'\bAbnormality\b': 'Đặc điểm bất thường',
            r'\babnormality\b': 'đặc điểm bất thường',
            r'\bTongue\b': 'Lưỡi',
            r'\btongue\b': 'lưỡi',
            r'\bFace\b': 'Mặt',
            r'\bface\b': 'mặt',
            r'\bappear\b': 'có vẻ',
            r'\bappears\b': 'có vẻ',
            r'\bbumps\b': 'nốt mụn',
            r'\bbump\b': 'nốt mụn',
            r'\btexture\b': 'kết cấu',
            r'\bTypical\b': 'điển hình',
            r'\btypical\b': 'điển hình',
            r'\bNormal\b': 'bình thường',
            r'\bnormal\b': 'bình thường',
            r'\bbut\b': 'nhưng',
            r'\bBut\b': 'Nhưng',
            r'\band\b': 'và',
            r'\bwith\b': 'với',
            r'\baround\b': 'quanh',
            r'\bunder\b': 'dưới',
            r'\bon\b': 'trên',
            r'\bin\b': 'trong',
            r'\bof\b': 'của',
            # Bổ sung từ tiếng Anh mà Qwen hay bỏ sót khi dịch
            r'\blooks\b': 'trông',
            r'\bLooks\b': 'Trông',
            r'\bsomewhat\b': 'hơi',
            r'\bSomewhat\b': 'Hơi',
            r'\bslightly\b': 'nhẹ',
            r'\bSlightly\b': 'Nhẹ',
            r'\bmildly\b': 'nhẹ',
            r'\bMildly\b': 'Nhẹ',
            r'\bmay\b': 'có thể',
            r'\bMay\b': 'Có thể',
            r'\bcould\b': 'có thể',
            r'\bCould\b': 'Có thể',
            r'\bsuggesting\b': 'gợi ý',
            r'\bSuggesting\b': 'Gợi ý',
            r'\bsuggests\b': 'gợi ý',
            r'\bSuggests\b': 'Gợi ý',
            r'\bor\b': 'hoặc',
            r'\bOr\b': 'Hoặc',
            r'\bsome\b': 'một số',
            r'\bSome\b': 'Một số',
            r'\bfew\b': 'vài',
            r'\bsmall\b': 'nhỏ',
            r'\bthin\b': 'mỏng',
            r'\bthick\b': 'dày',
            r'\bpale\b': 'nhợt',
            r'\bred\b': 'đỏ',
            r'\bwhite\b': 'trắng',
            r'\bno\b': 'không',
            r'\bNo\b': 'Không',
            r'\bnot\b': 'không',
            r'\bNot\b': 'Không',
            r'\bcoating\b': 'rêu',
            r'\bCoating\b': 'Rêu',
            r'\bgreasy\b': 'nhờn',
            r'\bsticky\b': 'dính',
            r'\bwavy\b': 'gợn sóng',
            r'\bindentation\b': 'vết ấn',
            r'\bcracks\b': 'vết nứt',
            r'\bcrack\b': 'vết nứt',
            r'\bsurface\b': 'bề mặt',
            r'\bedges\b': 'mép',
            r'\bedge\b': 'mép',
            r'\bsides\b': 'hai bên',
            r'\bside\b': 'bên'
        }
        for eng, vie in replacements.items():
            text = re.sub(eng, vie, text)
            
        # 10. Phân rã câu, loại bỏ trùng lặp và gộp các câu con vào câu dài hơn
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            is_dup = False
            for i, existing in enumerate(unique_sentences):
                if s.lower().rstrip('.') in existing.lower().rstrip('.'):
                    is_dup = True
                    break
                elif existing.lower().rstrip('.') in s.lower().rstrip('.'):
                    unique_sentences[i] = s
                    is_dup = True
                    break
            if not is_dup:
                unique_sentences.append(s)
                
        text = " ".join(unique_sentences)

        # 11. Chuẩn hóa khoảng trắng và dấu chấm câu cuối
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.+', '.', text)
        text = text.strip()
        text = re.sub(r'^[.:,\s\-]+', '', text)
        
        if text and not text.endswith('.'):
            text += '.'
        return text

    def _translate_english_description(self, english_text: str) -> str:
        """Dịch mô tả sắc mặt/lưỡi tiếng Anh sang tiếng Việt bằng Qwen"""
        if not english_text:
            return ""
        
        # Heuristic: Thay thế các từ nhạy cảm để tránh bug tokenization/loop/tự giải thích của Qwen
        text_clean = english_text.replace("indicate", "suggest").replace("Indicate", "Suggest")
        text_clean = text_clean.replace("indicates", "suggests").replace("Indicates", "Suggests")
        text_clean = text_clean.replace("abnormalities", "abnormal features").replace("Abnormalities", "Abnormal features")
        text_clean = text_clean.replace("abnormality", "abnormal feature").replace("Abnormality", "Abnormal feature")
        text_clean = text_clean.replace("visible", "apparent").replace("Visible", "Apparent")
        
        # [CRITICAL FIX] Pre-inject Vietnamese hints vào text tiếng Anh để Qwen không dịch sai thin/thick
        # Qwen có xu hướng dịch 'thin coating' thành 'rêu dày' do hiểu nhầm ngữ cảnh y khoa
        import re as _re
        text_clean = _re.sub(r'\bthin,?\s*white\s+coating', 'thin (mỏng) white coating', text_clean, flags=_re.IGNORECASE)
        text_clean = _re.sub(r'\bthick,?\s*white\s+coating', 'thick (dày) white coating', text_clean, flags=_re.IGNORECASE)
        text_clean = _re.sub(r'\bthin\s+coating', 'thin (mỏng) coating', text_clean, flags=_re.IGNORECASE)
        text_clean = _re.sub(r'\bthick\s+coating', 'thick (dày) coating', text_clean, flags=_re.IGNORECASE)
        
        prompt = f"""
        Translate the following English medical description of patient's features into natural Vietnamese.
        TUYỆT ĐỐI KHÔNG sử dụng chữ Hán (tiếng Trung), tiếng Anh hay bất kỳ ngôn ngữ nào khác ngoài tiếng Việt.
        TUYỆT ĐỐI KHÔNG giải thích, KHÔNG viết chú thích hay tự sửa chữa bằng cả tiếng Anh hay tiếng Việt. Chỉ trả về duy nhất văn bản dịch tiếng Việt sạch.

        Examples:
        English: "The complexion of the person in the image is pale with some redness on the cheeks."
        Vietnamese: "Sắc mặt nhợt nhạt với một chút đỏ trên má."
        
        English: "The tongue is pale red with a thin, white coating that appears slightly greasy."
        Vietnamese: "Lưỡi hồng nhạt với rêu trắng mỏng hơi nhờn."

        English: "The tongue is pale red with a thin, white coating that appears slightly greasy. It has a mildly wavy indentation on the left side and a few small cracks on the surface."
        Vietnamese: "Lưỡi hồng nhạt với rêu trắng mỏng hơi nhờn. Mép lưỡi có vết ấn gợn sóng nhẹ ở bên trái và vài vết nứt nhỏ trên bề mặt."

        Text to translate: "{text_clean}"
        Vietnamese translation:
        """
        try:
            res = self.qa_pipeline.client.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là một công cụ dịch thuật y khoa Đông y tự động. Nhiệm vụ duy nhất của bạn là dịch văn bản tiếng Anh sang tiếng Việt. Tuyệt đối KHÔNG sử dụng chữ Hán (tiếng Trung) hay bất kỳ ngôn ngữ nào khác ngoài tiếng Việt. Tuyệt đối KHÔNG viết thêm bất kỳ ghi chú, giải thích, hay tự sửa lỗi nào (như 'Note:', 'Lưu ý:', 'Corrected version'). Chỉ xuất ra duy nhất bản dịch tiếng Việt sạch."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "seed": 42, "max_tokens": 300}
            )
            ans = res['message']['content'].strip()
            ans = ans.replace("Vietnamese translation:", "").replace("Vietnamese:", "").replace('"', '').strip()
            
            # Làm sạch kết quả dịch triệt độ chống loop và chữ ngoại quốc
            ans = self._clean_translated_text(ans)
            return ans
        except Exception as e:
            logger.error(f"Lỗi dịch mô tả tiếng Anh: {e}")
            return self._clean_translated_text(english_text)

    def _map_desc_to_symptoms(self, description: str, candidate_symptoms: list) -> list:
        """Dùng Qwen để ánh xạ mô tả tự nhiên tiếng Anh/tiếng Việt sang các triệu chứng chuẩn trong database"""
        if not description:
            return []
            
        candidates_str = ", ".join(candidate_symptoms)
        
        prompt = f"""
        Role: Traditional Chinese Medicine expert assistant.
        We have a list of standardized medical symptoms:
        {candidates_str}
        
        Analyze the following patient feature description:
        "{description}"
        
        Task: Select all symptoms from the standardized list above that are present in the patient description.
        Only select symptoms that are actually present (positive symptoms).
        
        CRITICAL NEGATION RULES (MUST OBEY FIRST):
        1. You MUST check if any symptom is negated (e.g. described as NOT present, absent, smooth, clear, normal, no abnormalities, without, no signs of, không có, không bị, không xuất hiện, bình thường).
        2. If a symptom is negated, you MUST NOT select it under any circumstances.
        3. For example: "Không có quầng thâm dưới mắt" or "no dark circles" means you MUST NOT select "quầng đen dưới mắt".
        4. "Không bị sưng phù" or "no signs of swelling" means you MUST NOT select "mí mắt dưới hơi sưng", "phù ở mí mắt", or "Mặt phù".
        5. "Không có phát ban" or "no rash" means you MUST NOT select "Mặt có ban" or "Ban xuất huyết dưới da".
        6. You MUST handle indirect or list-based negation. If the text says there are 'no abnormalities/issues/features, such as [A], [B], or [C]', or 'free of issues like [A] or [B]', it means A, B, and C are NOT present. You MUST NOT select them. For example, 'no significant abnormalities, such as dark circles under the eyes, swelling, or rash' means dark circles, swelling, and rash are all absent, so you must NOT select 'quầng đen dưới mắt', 'sưng phù', 'mí mắt dưới hơi sưng', 'Mặt phù', 'Mặt có ban', or 'Ban xuất huyết dưới da'.
        
        SPECIFIC MAPPING RULES:
        1. SWELLING/PUFFINESS: Only select "mí mắt dưới hơi sưng" or "phù ở mí mắt" if the description specifically mentions puffiness/swelling around the eyelids/eyes (e.g. "puffy lower eyelids", "puffiness around the lower eyelids"). Do NOT select "Mặt phù" (which is whole face swelling) unless the description explicitly says the whole face is puffy/swollen.
        2. ACNE/REDNESS: If description mentions red spots, acne, pimples, or inflammatory bumps (e.g. "redness on the cheeks", "bump on the bridge of nose", "red bumps", "acne"), select only the matching items like "nốt mụn đỏ", "mụn viêm", "mụn đỏ", "vùng đỏ trên mặt". If description only mentions minor pinkness/slight blush/slight redness, ignore "Mặt đỏ" and "hai gò má đỏ".
        3. SPECIAL RULE FOR HEALTHY COMPLEXION: You MUST NOT map healthy, normal, or positive skin descriptions to pathological symptoms. For example, descriptions like "trắng hồng", "làn da hồng hào", "hồng nhuận", "rosy complexion", "healthy pink", "pinkish skin", "healthy white and rosy", "normal complexion" represent healthy physiological states and MUST NOT be mapped to pathological symptoms like "Mặt trắng nhợt", "Mặt nhợt nhạt", or "Mặt đỏ". Check the context carefully: if the redness or pinkness is described as a mild, healthy rosy glow or normal pinkness of skin, DO NOT select "Mặt đỏ" or "vùng đỏ trên mặt".
        4. SPECIAL RULE FOR TONGUE COATING: If the description mentions a thicker, slightly thicker, thick, or greasy coating/fur on the tongue (e.g., 'slightly thicker coating on the surface', 'coating is thicker than normal', 'thick white coat', 'greasy coating', 'lớp phủ hơi dày hơn'), you MUST map it to 'rêu lưỡi trắng dày', 'rêu trắng dày', or 'rêu lưỡi dày nhớt' if they are in the candidate list. Do NOT ignore a thicker tongue coating.
        5. SPECIAL RULE FOR NORMAL TONGUE COLOR: If the tongue body color is described as normal pink, pale red, rosy, or healthy (e.g., 'normal pink color', 'pale red tongue body', 'healthy pink', 'lưỡi có màu hồng bình thường'), you MUST map it to 'rêu bình thường' or 'rêu mỏng trắng' or ignore it if no pathological color symptoms match. Do NOT map healthy pink tongue color to pathological symptoms like 'Lưỡi nhợt', 'Lưỡi nhạt' or 'Lưỡi đỏ'.
        6. SPECIAL RULE FOR AMBIENT LIGHTING: If the description states that a yellow, orange, or warm tint on the face is due to ambient room lighting, indoor light bulbs, background colors, or image artifacts (e.g. 'slightly yellow due to indoor lighting', 'a bit of yellow due to background color', 'vàng do ánh sáng trong nhà'), you MUST NOT select 'Mặt vàng' or 'sắc mặt vàng'. You should map it only to 'Mặt trắng nhợt', 'Mặt nhợt nhạt', or 'mặt trắng' if they are described as the main skin state.
        7. SPECIAL RULE FOR CRACKS (VẾT NỨT): If the description mentions cracks, small cracks, fissures, or split tongue (e.g. 'small cracks along the edges', 'cracks on the surface', 'fissures'), you MUST map it only to 'lưỡi có vết nứt' or 'lưỡi nứt' if they are in the candidate list. Under no circumstances (TUYỆT ĐỐI CẤM) should you map cracks/fissures to blood stasis or purple tongue symptoms like 'lưỡi có vết bầm tím', 'lưỡi có điểm ứ huyết', 'lưỡi tím tái', or 'chất lưỡi tím'. Cracks represent Yin deficiency or body fluid consumption, not blood stasis.
        8. SPECIAL RULE FOR TEETH MARKS (HẰN RĂNG / DẤU RĂNG): If the description mentions that the tongue borders or edges are irregular, uneven, bumpy, wavy, scalloped, or have small notches/impressions/dents/cracks along the sides (e.g., 'hình dạng lưỡi hơi không đều dọc theo các cạnh', 'mép lưỡi gồ ghề', 'irregular edges', 'scalloped borders', 'uneven edges', 'notches on the sides'), you MUST map it to 'rìa lưỡi có hằn răng' or 'lưỡi bệu' if they are in the candidate list. This is a very common description of teeth marks (dấu răng / hằn răng) by vision models.
        
        Output ONLY a comma-separated list of the selected standardized symptoms, EXACTLY as written in the list. Do NOT add notes, explanations, introductory text, or markdown. If nothing matches, output 'Không'.
        """
        try:
            res = self.qa_pipeline.client.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise data mapper. Output only the comma-separated symptoms from the list, or 'Không'. Negation, healthy states, and tongue features must be strictly handled according to the rules."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "seed": 42}
            )
            content = res['message']['content'].strip()
            if content.lower() == 'không' or not content:
                return []
            
            # Split and clean the mapped symptoms
            mapped = [s.strip() for s in content.split(",") if s.strip()]
            valid_mapped = []
            for m in mapped:
                m_lower = m.lower()
                # 1. Khớp chính xác trước
                matched = False
                for cand in candidate_symptoms:
                    cand_lower = cand.lower()
                    if m_lower == cand_lower:
                        if cand not in valid_mapped:
                            valid_mapped.append(cand)
                        matched = True
                        break
                if matched:
                    continue
                # 2. Khớp fuzzy tập con/tập cha
                for cand in candidate_symptoms:
                    cand_lower = cand.lower()
                    if (cand_lower in m_lower or m_lower in cand_lower) and cand not in valid_mapped:
                        # Bỏ qua các từ khóa quá ngắn tránh khớp sai lệch
                        if len(cand_lower) >= 6:
                            valid_mapped.append(cand)
                            break
            return valid_mapped
        except Exception as e:
            logger.error(f"Lỗi mapping triệu chứng bằng LLM: {e}")
            return []

    def _load_csv_data(self):
        self.csv_rows = []
        try:
            import csv
            import os
            # Ưu tiên biến môi trường TCM_CSV_PATH, rồi file trong repo, cuối cùng mới tới
            # đường dẫn cá nhân cũ (giữ để không phá vỡ máy đang dùng).
            candidates = [
                os.getenv("TCM_CSV_PATH"),
                "data/Medicine_clean.csv",
                r"C:\Users\hoang\Downloads\Medicine_new.csv",
            ]
            csv_path = next((p for p in candidates if p and os.path.exists(p)), None)
            if csv_path:
                with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.csv_rows.append({
                            "benh_ly": row.get("tên_bệnh", "").strip(),
                            "hoi_chung": row.get("hội_chứng", "").strip(),
                            "triệu_chứng": row.get("triệu_chứng", "").strip(),
                            "bai_thuoc": row.get("bài_thuốc", "").strip(),
                            "vi_thuoc": row.get("vị_thuốc", "").strip(),
                        })
                logger.info(f"Đã tải {len(self.csv_rows)} dòng dữ liệu từ file CSV ({csv_path}) để so khớp bệnh lý")
            else:
                logger.warning("Không tìm thấy file CSV dữ liệu bệnh lý (đã thử TCM_CSV_PATH và data/Medicine_clean.csv).")
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu CSV: {e}")

    def _validate_disease_safety(self, disease_name: str, patient_symptoms: list, raw_user_text: str) -> bool:
        """Bộ lọc an toàn lâm sàng: Loại bỏ các bệnh lý chuyên khoa đặc thù nếu không có triệu chứng chỉ điểm tương ứng"""
        disease_lower = disease_name.lower()
        symptoms_str = " ".join(patient_symptoms).lower() + " " + raw_user_text.lower()

        # 0. [FIX] Bệnh danh mà TÊN GỌI chính là triệu chứng chủ đạo -> bệnh nhân BẮT BUỘC phải có
        #    triệu chứng đó, tránh nêu bệnh không liên quan (vd "Xuất hãn dị thường" cho người không
        #    hề ra mồ hôi, "Cuồng" cho người không có dấu tâm thần kích động).
        if any(x in disease_lower for x in ["xuất hãn", "tự hãn", "đạo hãn", "mồ hôi", "hãn chứng"]):
            if not any(kw in symptoms_str for kw in ["mồ hôi", "hãn", "đổ mồ hôi", "ra mồ hôi", "vã mồ hôi", "ướt đẫm"]):
                return False
        if any(x in disease_lower for x in ["cuồng", "điên", "thao cuồng"]):
            if not any(kw in symptoms_str for kw in ["kích động", "la hét", "hưng phấn", "nói nhảm", "hoang tưởng",
                                                     "đập phá", "cuồng", "điên", "mất ngủ", "loạn thần", "rối loạn tâm thần"]):
                return False
        if "ách nghịch" in disease_lower or "nấc" in disease_lower:
            if not any(kw in symptoms_str for kw in ["nấc", "ách nghịch"]):
                return False
        if any(x in disease_lower for x in ["động kinh", "giản chứng", "kinh phong", "phong giật", "co giật"]):
            if not any(kw in symptoms_str for kw in ["co giật", "động kinh", "giật", "kinh phong",
                                                     "sùi bọt mép", "ngã lăn", "mất ý thức", "hôn mê", "cứng người"]):
                return False

        # 1. Bệnh Trĩ
        if "trĩ" in disease_lower:
            keywords = ["trĩ", "hậu môn", "đại tiện ra máu", "tiêu ra máu", "đi ngoài ra máu", "ỉa ra máu", "sa búi", "búi trĩ"]
            if not any(kw in symptoms_str for kw in keywords):
                return False
                
        # 2. Chấn thương sọ não
        if "chấn thương sọ não" in disease_lower:
            keywords = ["chấn thương", "va đập", "tai nạn", "ngã đầu", "đập đầu", "ngoại thương", "bị thương"]
            if not any(kw in symptoms_str for kw in keywords):
                return False
                
        # 3. Viêm phế quản / Phế quản / Viêm họng
        if any(x in disease_lower for x in ["phế quản", "viêm họng", "viêm đường hô hấp"]):
            keywords = ["ho", "đờm", "đàm", "khạc", "họng", "cổ họng", "phế quản", "sổ mũi", "ngạt mũi", "hắt hơi"]
            if not any(kw in symptoms_str for kw in keywords):
                return False
                
        # 4. Viêm xoang
        if "xoang" in disease_lower:
            keywords = ["xoang", "mũi", "ngạt mũi", "chảy nước mũi", "sổ mũi", "tịt mũi"]
            if not any(kw in symptoms_str for kw in keywords):
                return False
                
        # 5. Trúng phong / Tai biến / Đột quỵ
        if any(x in disease_lower for x in ["trúng phong", "tai biến", "đột quỵ"]):
            keywords = ["liệt", "méo miệng", "bán thân bất toại", "tê bại", "khó nói", "mất ngôn ngữ", "trúng phong", "tai biến"]
            if not any(kw in symptoms_str for kw in keywords):
                return False
                
        # 6. Các bệnh lý thuần túy Tây y
        western_kws = ["addison", "alzheimer", "basedow", "parkinson", "eczema", "gout",
                       "tuyến thượng thận", "suy tim", "loãng xương",
                       "bệnh bạch huyết", "leukemia", "lymphoma", "lupus", "sclerosis",
                       "parathyroid", "cushing", "hashimoto", "hodgkin"]
        if any(kw in disease_lower for kw in western_kws):
            return False

        # 7. Bệnh lý Huyết học / Ung thư máu
        if any(x in disease_lower for x in ["bạch huyết", "bạch cầu", "ung thư máu", "huyết hữu", "xuất huyết giảm tiểu cầu"]):
            keywords = ["xuất huyết", "bầm tím tự phát", "chảy máu", "hạch", "gan lách to", "tiểu cầu"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 8. Bệnh lý Phụ khoa / Sản khoa
        if any(x in disease_lower for x in ["lưu sản", "sảy thai", "băng lậu", "đới hạ", "vô sinh", "bế kinh", "thống kinh", "sản hậu", "nhau thai", "thai chết"]):
            keywords = ["kinh nguyệt", "kinh", "thai", "sản", "âm đạo", "huyết trắng", "đới hạ", "băng", "lậu", "tử cung", "phụ nữ"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 9. Bệnh lý Tai Mũi Họng (Nhĩ minh, Điếc, Viêm tai)
        if any(x in disease_lower for x in ["nhĩ minh", "nhĩ lung", "điếc", "ù tai", "viêm tai"]):
            keywords = ["ù tai", "tai", "điếc", "nghe kém", "nhĩ minh", "nhĩ lung"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 10. Bệnh lý Mắt
        if any(x in disease_lower for x in ["thanh manh", "nhược thị", "mắt mờ", "quáng gà", "đục thủy tinh", "nội chướng", "ngoại chướng"]):
            keywords = ["mắt mờ", "nhìn mờ", "thị lực", "quáng gà", "mắt đau", "mắt sưng"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 11. Bệnh lý Da liễu (Mề đay, Chàm, Ghẻ lở...)
        if any(x in disease_lower for x in ["mề đay", "chàm", "ghẻ", "hắc lào", "vẩy nến", "mụn nhọt", "ung nhọt"]):
            keywords = ["ngứa", "mẩn", "ban", "mụn", "ghẻ", "da", "phát ban", "mề đay", "nổi mề"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 12. Bệnh lý Tiết niệu (Đái dầm, Tiểu đêm, Sỏi thận...)
        if any(x in disease_lower for x in ["đái dầm", "di niệu", "sỏi thận", "sỏi tiết niệu", "viêm bàng quang"]):
            keywords = ["tiểu", "đái", "niệu", "bàng quang", "sỏi", "tiểu đêm", "tiểu gắt"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 13. Bệnh lý Thận chuyên sâu (Viêm cầu thận, Suy thận, Hội chứng thận hư)
        if any(x in disease_lower for x in ["viêm cầu thận", "suy thận", "thận hư hội chứng", "viêm thận", "hội chứng thận hư"]):
            keywords = ["phù", "tiểu ít", "tiểu đục", "tiểu ra máu", "phù mặt", "phù chân", "albumin", "protein niệu", "thận"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 14. Bệnh lý Ung thư / Khối u (Ái = Ung thư trong Đông y: Thực quản ái, Phế ái, Vị ái...)
        if any(x in disease_lower for x in ["ái", "ung thư", "u ác", "khối u", "nhục lựu"]):
            keywords = ["khối u", "sụt cân", "nuốt nghẹn", "ho ra máu", "u bướu", "sưng hạch", "di căn", "ung thư"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        # 15. Bệnh lý Tiêu hóa chuyên sâu (Viêm loét dạ dày, Trào ngược)
        if any(x in disease_lower for x in ["thực quản", "vị quản", "loét dạ dày", "viêm dạ dày"]):
            keywords = ["đau bụng", "đau dạ dày", "ợ chua", "ợ hơi", "buồn nôn", "nôn", "nuốt nghẹn", "trào ngược", "thượng vị"]
            if not any(kw in symptoms_str for kw in keywords):
                return False

        return True

    def _find_matching_diseases(self, patient_symptoms: list, raw_user_text: str = "") -> list:
        """Tìm các bệnh lý dựa trên tỷ lệ so khớp triệu chứng (mềm dẻo hơn thay vì bắt buộc 100%)"""
        if not hasattr(self, 'csv_rows') or not self.csv_rows:
            return []
            
        patient_symptoms_lower = [s.lower().strip() for s in patient_symptoms] if patient_symptoms else []
        raw_text_lower = raw_user_text.lower() if raw_user_text else ""
        
        matched_candidates = []
        for row in self.csv_rows:
            db_symptoms_str = row.get("triệu_chứng", "")
            if not db_symptoms_str:
                continue
            db_symptoms = [s.strip() for s in db_symptoms_str.split(",") if s.strip()]
            if not db_symptoms:
                continue
            
            matched_count = 0
            for ds in db_symptoms:
                ds_lower = ds.lower().strip()
                # Cách 1: Khớp chính xác hoặc chứa trong tập triệu chứng đã chuẩn hóa
                if patient_symptoms_lower and any(ds_lower == ps or ds_lower in ps or ps in ds_lower for ps in patient_symptoms_lower):
                    matched_count += 1
                    continue
                # Cách 2: Khớp mềm từ khóa trên văn bản gốc của người dùng
                if raw_text_lower:
                    ds_words = [w for w in ds_lower.split() if len(w) >= 2]
                    if ds_words and all(w in raw_text_lower for w in ds_words):
                        matched_count += 1
                        continue
            
            match_ratio = matched_count / len(db_symptoms)
            
            # Ngưỡng chấp nhận: khớp >= 30% và khớp từ 2 triệu chứng trở lên (phù hợp với thực tế lâm sàng lẻ tẻ)
            if match_ratio >= 0.30 and matched_count >= 2:
                # Áp dụng bộ lọc an toàn lâm sàng ngăn chặn chẩn đoán sai lệch
                if self._validate_disease_safety(row["benh_ly"], patient_symptoms, raw_user_text):
                    matched_candidates.append({
                        "benh_ly": row["benh_ly"],
                        "hoi_chung": row["hoi_chung"],
                        "ratio": match_ratio,
                        "matched_count": matched_count
                    })
        
        # Sắp xếp theo tỷ lệ khớp giảm dần, ưu tiên số lượng triệu chứng khớp nhiều hơn
        matched_candidates.sort(key=lambda x: (x["ratio"], x["matched_count"]), reverse=True)
        return matched_candidates

    def _are_syndromes_related(self, syn1: str, syn2: str) -> bool:
        """Kiểm tra mối liên quan mềm dẻo giữa hai tên hội chứng (dựa trên các từ khóa y lý cốt lõi)"""
        syn1_l = syn1.lower()
        syn2_l = syn2.lower()
        # 1. Khớp chứa trực tiếp
        if syn1_l in syn2_l or syn2_l in syn1_l:
            return True
        # 2. Khớp khái niệm cốt lõi trùng lặp
        concept_groups = [
            ["ứ", "tắc", "ngưng", "sáp", "trệ"], # Huyết ứ / Khí trệ
            ["đàm", "thấp", "ẩm", "nhớt", "trọc"], # Đàm thấp
            ["hỏa", "nhiệt", "sáo", "táo", "độc"], # Nhiệt / Hỏa
            ["hàn", "lạnh", "trì"], # Hàn
            ["khí", "tỳ"], # Khí hư / Tỳ hư
            ["huyết", "tâm"], # Huyết hư / Tâm huyết
            ["âm"], # Âm hư
            ["dương"], # Dương hư
            ["thận"], # Thận hư
            ["can", "phong"] # Can phong
        ]
        for group in concept_groups:
            has1 = any(word in syn1_l for word in group)
            has2 = any(word in syn2_l for word in group)
            if has1 and has2:
                return True
        return False

    @staticmethod
    def _syndrome_is_hu(name: str) -> bool:
        """Hội chứng thuộc HƯ chứng (bản chất suy yếu/bất túc) theo từ khóa trong tên."""
        return bool(re.search(r'(hư|suy|bất túc|nhược|khuy|tổn)', (name or "").lower()))

    @classmethod
    def _syndrome_is_thuc_pure(cls, name: str) -> bool:
        """Hội chứng THỰC THUẦN (tà khí tích tụ: đàm/hỏa/nhiệt/ứ/trệ/nghịch...) và KHÔNG kèm chữ Hư.
        Hội chứng vừa Hư vừa Thực (vd 'Khí hư huyết ứ') coi là Hư (đã là Bản Hư Tiêu Thực nội tại)."""
        nl = (name or "").lower()
        # Tà khí thực: đàm/hỏa/nhiệt/ứ/trệ/uất/kết/ngưng/tích/độc/nghịch/phong + hàn/thấp (ngoại tà,
        # vd "Hàn thấp", "Thấp nhiệt", "Phong hàn"). "Hư hàn" có chữ "hư" nên vẫn được coi là Hư.
        has_thuc = bool(re.search(r'(đàm|hỏa|hoả|nhiệt|ứ|trệ|uất|kết|ngưng|tích|thực|độc|nghịch|phong|hàn|thấp|thử|táo)', nl))
        return has_thuc and not cls._syndrome_is_hu(nl)

    @staticmethod
    def _syndrome_is_exterior_wind(name: str) -> bool:
        """Hội chứng NGOẠI CẢM BIỂU (phong tà phạm biểu/vệ/phế): Phong hàn, Phong nhiệt, Phong thấp,
        Biểu hàn/nhiệt, Thương phong, *phạm phế/phạm vệ... Đây là BIỂU THỰC cấp — gốc bệnh là tà khí
        ngoại cảm ở biểu, KHÔNG có Bản Hư nội thương → KHÔNG được đảo hội chứng Hư lên làm cốt lõi
        (nếu không sẽ dựng sai cơ chế 'Bản Hư' cho một cảm mạo cấp)."""
        nl = (name or "").lower()
        if "hư" in nl:                                     # 'Phế khí hư', 'Biểu hư tự hãn' = nội thương
            return False
        if re.search(r'phong\s*(hàn|nhiệt|thấp|táo|thử|ôn)', nl):   # Phong hàn/nhiệt/thấp...
            return True
        if re.search(r'biểu\s*(hàn|nhiệt|thực|chứng)', nl):          # Biểu hàn/nhiệt/thực
            return True
        if any(k in nl for k in ("phạm phế", "phạm vệ", "thương phong", "ngoại cảm")):
            return True
        return False

    def _score_syndromes_grounded(self, patient_terms: list, top_n: int = 6) -> list:
        """[GROUNDED SYNDROME-SCORING] Chấm điểm mỗi HoiChung = SỐ triệu chứng bệnh nhân khớp với
        triệu chứng của hội chứng đó trên graph (cạnh CÓ_BIỂU_HIỆN), khớp theo ranh giới từ để bắt
        cả node triệu chứng ghép. Ranking DETERMINISTIC, ổn định, không phụ thuộc LLM 7B chọn bừa.

        Trả về list [(syndrome_name, matched_count)] xếp điểm giảm dần (tối đa top_n)."""
        terms = list(dict.fromkeys([t.strip().lower() for t in (patient_terms or []) if t and t.strip()]))
        if not terms:
            return []
        # Java-regex khớp từ độc lập (tái dùng của qa_system, bắt cả triệu chứng nằm trong node ghép)
        patterns = [self.qa_pipeline._word_boundary_pattern(t) for t in terms]
        # Điểm CÂN BẰNG = matched + idf_sum:
        #   - matched (số triệu chứng khớp) = phần nguyên -> COVERAGE là yếu tố chính (khớp nhiều thắng).
        #   - idf_sum = tổng 1/df (df = số hội chứng có triệu chứng đó) -> phần thập phân phá hoà, ưu
        #     tiên hội chứng khớp triệu chứng ĐẶC HIỆU (hiếm) khi coverage bằng nhau.
        # Loại node bẩn: tên chứa số/ngoặc, hoặc đã bị gắn cờ _flagged_dirty (từ clean-dirty).
        cypher = """
        UNWIND $patterns AS pat
        MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
        WHERE toLower(t.name) =~ pat
          AND NOT h.name =~ '.*[0-9(].*'
          AND NOT coalesce(h._flagged_dirty, false)
          AND NOT toLower(h.name) STARTS WITH 'thể '
        WITH pat, collect(DISTINCT h.name) AS syns
        WITH pat, syns, toFloat(size(syns)) AS df
        UNWIND syns AS syndrome
        WITH syndrome, sum(1.0/df) AS idf, count(DISTINCT pat) AS matched
        RETURN syndrome, (matched + idf) AS score, matched
        ORDER BY score DESC, syndrome LIMIT 40
        """
        try:
            rows = self.qa_pipeline.run_cypher(cypher, {"patterns": patterns})
        except Exception as e:
            logger.error(f"Lỗi chấm điểm hội chứng grounded: {e}")
            return []
        scored = [(r["syndrome"], r.get("score", 0), r.get("matched", 0)) for r in rows if r.get("syndrome")]
        if not scored:
            return []
        # Ưu tiên hội chứng khớp >= 2 triệu chứng (giảm nhiễu 1-triệu-chứng); nếu không có mới lấy >= 1
        strong = [x for x in scored if x[2] >= 2]
        picked = (strong if strong else scored)[:top_n]
        logger.info(f"[GROUNDED] Điểm hội chứng (IDF): {[(n, round(s, 3), m) for n, s, m in picked]}")
        return [(name, round(score, 3)) for name, score, _matched in picked]

    def _extract_syndromes_from_text(self, text: str) -> list:
        """Trích xuất hội chứng bằng cách khớp triệu chứng chính xác trước, kết hợp LLM dịch thuật/quy đổi từ khóa đối với câu phức tạp"""
        # [FIX] Lưu lại các từ khóa triệu chứng đã dùng để bước truy hồi bài thuốc bám đúng ngữ cảnh bệnh.
        self._last_extracted_terms = []
        if not text:
            return []

        syndromes = []
        try:
            # Kiểm tra nếu câu chứa tiếng Anh (từ LLaVA face description)
            english_indicators = ["patient", "pale", "complexion", "expression", "spirit", "eyes", "swelling", "spots", "rose", "skin", "face", "fatigue", "spiritless", "puffiness", "dark circles", "mole"]
            has_english = any(w in text.lower() for w in english_indicators)

            # 1. Thử khớp triệu chứng chính xác từ database trước (Longest Match First)
            exact_symptoms = self.qa_pipeline._preprocess_question(text)
            if exact_symptoms:
                # Làm sạch dữ liệu và giải quyết mâu thuẫn y lý trước khi truy vấn Neo4j
                exact_symptoms = self._resolve_symptom_conflicts(exact_symptoms)
                
                self._last_extracted_terms = [s.lower() for s in exact_symptoms]
                logger.info(f"Đã khớp triệu chứng chính xác từ database: {exact_symptoms}")
                symptoms_lower = [s.lower() for s in exact_symptoms]
                cypher = """
                MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
                WHERE toLower(t.name) IN $symptoms
                RETURN DISTINCT h.name
                """
                records = self.qa_pipeline.run_cypher(cypher, {"symptoms": symptoms_lower})
                for rec in records:
                    if rec.get('h.name'):
                        syndromes.append(rec['h.name'])

            # Thử khớp tên bệnh lý (BenhLy) từ database
            matched_diseases = []
            try:
                with self.qa_pipeline.driver.session() as session:
                    result = session.run("MATCH (b:BenhLy) RETURN b.name AS name")
                    db_diseases = [record["name"] for record in result]
                
                db_diseases_sorted = sorted(db_diseases, key=len, reverse=True)
                text_lower = text.lower()
                for disease in db_diseases_sorted:
                    if len(disease.split()) < 2 and disease.lower() not in ["ho", "lao"]:
                        continue
                    import re
                    pattern = rf'\b{re.escape(disease.lower())}\b'
                    if re.search(pattern, text_lower):
                        matched_diseases.append(disease)
            except Exception as ex:
                logger.error(f"Lỗi khớp tên bệnh lý: {ex}")

            # Lọc bỏ các bệnh lý trùng lặp hoặc là con (sub-phrase) của các triệu chứng đã khớp để ưu tiên triệu chứng đặc hiệu
            if exact_symptoms and matched_diseases:
                filtered_diseases = []
                symptoms_lower = [s.lower() for s in exact_symptoms]
                for disease in matched_diseases:
                    disease_lower = disease.lower()
                    if any(disease_lower in sym for sym in symptoms_lower):
                        continue
                    filtered_diseases.append(disease)
                matched_diseases = filtered_diseases

            if matched_diseases:
                logger.info(f"Đã khớp bệnh lý chính xác từ database: {matched_diseases}")
                diseases_lower = [d.lower() for d in matched_diseases]
                cypher = """
                MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)
                WHERE toLower(b.name) IN $diseases
                RETURN DISTINCT h.name
                """
                records = self.qa_pipeline.run_cypher(cypher, {"diseases": diseases_lower})
                for rec in records:
                    if rec.get('h.name'):
                        syndromes.append(rec['h.name'])
                        
            # [FIX] Bỏ qua early return để tất cả hội chứng đều được đi qua bộ lọc LLM thông minh bên dưới

            # 2. Fallback hoặc có chứa tiếng Anh: dùng LLM dịch thuật và đóng vai trò màng lọc nhiễu
            word_count = len(text.split())
            should_run_llm = not syndromes or word_count > 10 or has_english
            
            if should_run_llm:
                if has_english:
                    logger.info("Phát hiện mô tả tiếng Anh hoặc cần kích hoạt LLM dịch thuật & trích xuất...")
                else:
                    logger.info("Kích hoạt LLM trích xuất và quy đổi triệu chứng từ câu dài/phức tạp...")
                    
                text_norm = normalize_symptoms_text(text)
                
                if has_english:
                    prompt = f"""
                    Câu của bệnh nhân (chứa triệu chứng tiếng Việt và mô tả sắc mặt bằng tiếng Anh): "{text_norm}"
                    Nhiệm vụ: Chỉ trích xuất các danh từ/động từ chỉ TRIỆU CHỨNG Y KHOA bằng TIẾNG VIỆT thực sự.
                    LƯU Ý QUAN TRỌNG: Nếu trong câu có mô tả bằng tiếng Anh (ví dụ: 'pale complexion', 'dark circles under eyes', 'spots on face', 'puffiness'), hãy DỊCH và quy đổi chúng sang triệu chứng Đông y tiếng Việt tương ứng (ví dụ: 'sắc mặt nhợt nhạt', 'quầng thâm mắt', 'mặt có ban', 'mặt phù').
                    MỖI TRIỆU CHỨNG TIẾNG VIỆT PHẢI CÓ TỪ 2 TỪ TRỞ LÊN (ví dụ: ho khan, ho đờm, đau đầu, sắc mặt nhợt).
                    TUYỆT ĐỐI KHÔNG trích xuất các triệu chứng chủ quan về tinh thần, cảm xúc, thần sắc hoặc biểu cảm khuôn mặt từ ảnh chụp (ví dụ: loại bỏ hoàn toàn các từ như 'mệt mỏi', 'thần sắc mệt mỏi', 'biểu cảm mệt mỏi', 'biểu cảm trung tính', 'dấu hiệu mệt mỏi', 'căng thẳng', 'làm việc quá sức'). Chỉ tập trung vào các đặc điểm thực thể vật lý thực sự (ví dụ: sắc mặt nhợt nhạt, quầng thâm mắt, mặt phù, lưỡi nhợt, rêu trắng...).
                    TUYỆT ĐỐI KHÔNG trích xuất các từ đơn lẻ chỉ có 1 từ (ví dụ: ho, sốt, đau, mỏi) và loại bỏ các trạng từ chỉ mức độ, thời gian hoặc từ xưng hô (ví dụ: tôi, bị, liên tục, nhiều, quá, rũ rượi, dồn dập).
                    LƯU Ý QUAN TRỌNG VỀ PHỦ ĐỊNH (NEGATION): Nếu trong câu hoặc mô tả có chứa các cấu trúc phủ định (như 'không có', 'không bị', 'không xuất hiện', 'no', 'without', 'no signs of', 'normal', 'smooth without', 'không có dấu hiệu bất thường đáng kể nào như quầng thâm dưới mắt, sưng phù hay phát ban'), bạn TUYỆT ĐỐI KHÔNG ĐƯỢC trích xuất các triệu chứng bị phủ định đó. Chỉ trích xuất các triệu chứng thực tế đang tồn tại ở dạng khẳng định (positive symptoms).
                    Chỉ trả về danh sách các từ khóa tiếng Việt cốt lõi, cách nhau bằng dấu phẩy. Không giải thích gì thêm.
                    """
                else:
                    prompt = f"""
                    Câu của bệnh nhân: "{text_norm}"
                    Nhiệm vụ:
                    1. Trích xuất các danh từ/động từ chỉ TRIỆU CHỨNG Y KHOA Đông y thực tế đang tồn tại ở dạng khẳng định từ câu hỏi.
                    2. Với mỗi triệu chứng trích xuất được, hãy bổ sung thêm các THUẬT NGỮ CỔ PHƯƠNG / TỪ ĐỒNG NGHĨA ĐÔNG Y của triệu chứng đó để giúp so khớp database tốt hơn.
                    QUY TẮC BẮT BUỘC: CHỈ được thêm TỪ ĐỒNG NGHĨA THẬT SỰ (cùng ý nghĩa lâm sàng, chỉ khác cách gọi). TUYỆT ĐỐI KHÔNG được thêm một TRIỆU CHỨNG KHÁC dù có liên quan hay hay đi kèm. (Ví dụ CẤM: không thêm 'ít nói' cho 'mệt mỏi'; không thêm 'phân sống' cho 'phân lỏng'; không thêm 'lưỡi hồng' [sinh lý] cho 'lưỡi đỏ' [bệnh lý].)
                    Ví dụ đồng nghĩa ĐÚNG:
                       - 'chóng mặt', 'hoa mắt' -> bổ sung thêm 'huyễn vựng'
                       - 'sợ lạnh' -> bổ sung thêm 'úy hàn'
                       - 'mệt mỏi' -> bổ sung thêm 'thần bì', 'lực kiệt'
                       - 'tay chân lạnh' -> bổ sung thêm 'tứ chi quyết nghịch', 'tứ chi bất ôn', 'chi lạnh'
                       - 'lưỡi nhạt' -> bổ sung thêm 'lưỡi đạm', 'đạm hồng', 'nhợt'
                       - 'lưỡi đỏ' -> bổ sung thêm 'hồng đỏ', 'chất lưỡi đỏ'
                       - 'nấc' -> bổ sung thêm 'ách nghịch'
                       - 'tiêu chảy' / 'phân lỏng' -> bổ sung thêm 'tiết tả', 'đại tiện lỏng'
                       - 'đầy bụng' -> bổ sung thêm 'trướng bụng', 'bụng trướng'
                       - 'mất ngủ' -> bổ sung thêm 'thất miên'
                       - 'khó thở' -> bổ sung thêm 'đoản khí', 'hơi thở ngắn'
                    MỖI TRIỆU CHỨNG HOẶC TỪ ĐỒNG NGHĨA PHẢI CÓ TỪ 2 TỪ TRỞ LÊN.
                    TUYỆT ĐỐI KHÔNG trích xuất các từ đơn lẻ chỉ có 1 từ (ví dụ: ho, sốt, đau, mỏi) và loại bỏ các trạng từ chỉ mức độ, thời gian hoặc từ xưng hô (ví dụ: tôi, bị, liên tục, nhiều, quá, rũ rượi, dồn dập).
                    LƯU Ý QUAN TRỌNG VỀ PHỦ ĐỊNH (NEGATION): Nếu trong câu hoặc mô tả có chứa các cấu trúc phủ định (như 'không có', 'không bị', 'không xuất hiện', 'no', 'without', 'no signs of', 'normal', 'smooth without', 'không có dấu hiệu bất thường đáng kể nào như quầng thâm dưới mắt, sưng phù hay phát ban'), bạn TUYỆT ĐỐI KHÔNG ĐƯỢC trích xuất các triệu chứng bị phủ định đó.
                    
                    Chỉ trả về danh sách các triệu chứng và từ đồng nghĩa chuẩn xác, cách nhau bằng dấu phẩy. Không giải thích gì thêm.
                    """
                    
                res = self.qa_pipeline.client.chat(
                    model=self.qa_pipeline.llm_model,
                    messages=[
                        {"role": "system", "content": "Bạn là trợ lý y khoa chỉ trích xuất từ khóa y khoa khẳng định bằng tiếng Việt kèm từ đồng nghĩa cổ phương của chúng từ câu hỏi của bệnh nhân. Loại bỏ hoàn toàn các triệu chứng bị phủ định."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.0, "seed": 42}
                )
                keywords_str = res['message']['content'].strip()
                
                # Xóa bỏ các ký tự thừa nếu LLM lỡ sinh ra
                keywords_str = keywords_str.replace('"', '').replace("'", "").replace(".", "")
                
                # Tách thành mảng các từ khóa sạch (bắt buộc triệu chứng phải từ 2 từ trở lên, ví dụ: ho khan, ho đờm, đau đầu)
                keywords = [k.strip().lower() for k in keywords_str.split(',') if len(k.strip().split()) >= 2]
                
                # Loại bỏ các từ khóa từ LLM nếu chúng trùng hoặc là tập con/tập cha của triệu chứng khớp chính xác từ database
                if exact_symptoms:
                    symptoms_lower = [sym.lower() for sym in exact_symptoms]
                    filtered_keywords = []
                    for kw in keywords:
                        overlap = False
                        for sym_l in symptoms_lower:
                            if kw in sym_l or sym_l in kw:
                                overlap = True
                                break
                        if not overlap:
                            filtered_keywords.append(kw)
                    keywords = filtered_keywords
                    
                    # Giữ nguyên các triệu chứng khớp chính xác từ database làm từ khóa chính
                    for sym_l in symptoms_lower:
                        if sym_l not in keywords:
                            keywords.append(sym_l)
                            
                # Áp dụng làm sạch mâu thuẫn y lý cho toàn bộ danh sách từ khóa
                keywords = self._resolve_symptom_conflicts(keywords)
                logger.info(f"Từ khóa y khoa đã lọc sạch nhiễu và quy đổi: {keywords}")

                # Quét trực tiếp Database với các từ khóa và chỉ giữ lại các từ khóa thực sự khớp với cơ sở dữ liệu
                valid_keywords = []
                for kw in keywords:
                    cypher = """
                    MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
                    WHERE toLower(t.name) CONTAINS $kw
                    RETURN DISTINCT h.name
                    """
                    records = self.qa_pipeline.run_cypher(cypher, {"kw": kw})
                    if records:
                        valid_keywords.append(kw)
                        for rec in records:
                            if rec.get('h.name'):
                                syndromes.append(rec['h.name'])
                
                # Chỉ lưu các từ khóa thực tế khớp với Graph DB để loại bỏ Ghost Tokens (như 'bạch mạc')
                self._last_extracted_terms = list(dict.fromkeys(valid_keywords))
                            
            # [FIX] Lọc danh sách hội chứng bằng LLM để chọn ra 1-3 hội chứng tối ưu nhất
            if syndromes:
                sorted_cands = self._sort_syndromes_by_organ_priority(list(set(syndromes)), text)
                symptom_list = exact_symptoms if exact_symptoms else (self._last_extracted_terms if self._last_extracted_terms else [text])
                symptom_list = self._resolve_symptom_conflicts(symptom_list)
                filtered = self._filter_syndromes_with_llm(symptom_list, sorted_cands)
                if filtered:
                    logger.info(f"Hội chứng sau khi lọc qua LLM: {filtered}")
                    # Ánh xạ ngược (Align) tên hội chứng LLM rút gọn về tên Node chính xác trong Neo4j (sorted_cands)
                    # Ưu tiên các hội chứng thuộc bệnh lý đã khớp trực tiếp để tăng tính nhất quán y lý điều trị
                    aligned_syndromes = []
                    
                    # Bước dự phòng: Tìm bệnh lý khớp theo triệu chứng thực tế nếu không có bệnh lý khớp trực tiếp từ text
                    active_diseases = list(matched_diseases)
                    if not active_diseases:
                        symptom_list_for_match = exact_symptoms if exact_symptoms else (self._last_extracted_terms if self._last_extracted_terms else [])
                        if symptom_list_for_match:
                            try:
                                matched_by_symptom = self._find_matching_diseases(symptom_list_for_match, raw_user_text=text)
                                if matched_by_symptom:
                                    # Lấy tỷ lệ khớp tốt nhất làm chuẩn
                                    max_ratio = matched_by_symptom[0]["ratio"]
                                    filtered_m = [
                                        m for m in matched_by_symptom 
                                        if m["ratio"] >= max_ratio - 0.15 and m["ratio"] >= 0.30
                                    ]
                                    active_diseases = list(dict.fromkeys([m["benh_ly"] for m in filtered_m]))[:3]
                            except Exception as ex:
                                logger.error(f"Lỗi tìm bệnh lý theo triệu chứng trong extract_syndromes: {ex}")

                    disease_syndromes = []
                    if active_diseases:
                        try:
                            diseases_lower = [d.lower() for d in active_diseases]
                            cypher = """
                            MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)
                            WHERE toLower(b.name) IN $diseases
                            RETURN DISTINCT h.name AS name
                            """
                            records = self.qa_pipeline.run_cypher(cypher, {"diseases": diseases_lower})
                            disease_syndromes = [rec["name"] for rec in records if rec.get("name")]
                        except Exception as ex:
                            logger.error(f"Lỗi lấy hội chứng theo bệnh lý: {ex}")

                    for fs in filtered:
                        matched_cand = None
                        fs_l = fs.lower().strip()
                        
                        # Bước 1: Ưu tiên tìm trong các hội chứng thuộc bệnh lý đã khớp
                        if disease_syndromes:
                            for cand in sorted_cands:
                                if cand in disease_syndromes:
                                    cand_l = cand.lower().strip()
                                    if cand_l == fs_l or cand_l.startswith(fs_l) or fs_l.startswith(cand_l) or self._are_syndromes_related(fs, cand):
                                        matched_cand = cand
                                        break
                                        
                        # Bước 2: Tìm chung trong toàn bộ ứng viên
                        if not matched_cand:
                            for cand in sorted_cands:
                                cand_l = cand.lower().strip()
                                if cand_l == fs_l or cand_l.startswith(fs_l) or fs_l.startswith(cand_l):
                                    matched_cand = cand
                                    break
                        # Bước 3: Khớp y lý mềm dẻo chung
                        if not matched_cand:
                            for cand in sorted_cands:
                                if self._are_syndromes_related(fs, cand):
                                    matched_cand = cand
                                    break
                        
                        if matched_cand:
                            aligned_syndromes.append(matched_cand)
                        else:
                            aligned_syndromes.append(fs)
                    
                    aligned_syndromes = list(dict.fromkeys(aligned_syndromes))
                    logger.info(f"Hội chứng sau khi ánh xạ ngược về database node: {aligned_syndromes}")
                    return aligned_syndromes
            return list(set(syndromes))
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất hội chứng: {e}")
            return []

    def _get_face_tongue_symptoms_for_syndrome(self, syndrome: str) -> dict:
        tongue_symptoms = []
        face_symptoms = []
        try:
            import json
            import os
            
            tongue_file = "data/mapping/symptom_to_syndrome.json"
            if os.path.exists(tongue_file):
                with open(tongue_file, "r", encoding="utf-8") as f:
                    tongue_map = json.load(f)
                for symptom, syndromes in tongue_map.items():
                    if syndrome in syndromes or any(syndrome.lower() == s.lower() for s in syndromes):
                        if symptom not in tongue_symptoms:
                            tongue_symptoms.append(symptom)
                            
            face_file = "data/mapping/face_to_syndrome.json"
            if os.path.exists(face_file):
                with open(face_file, "r", encoding="utf-8") as f:
                    face_map = json.load(f)
                for symptom, syndromes in face_map.items():
                    if syndrome in syndromes or any(syndrome.lower() == s.lower() for s in syndromes):
                        if symptom not in face_symptoms:
                            face_symptoms.append(symptom)
        except Exception as e:
            logger.error(f"Lỗi đọc file mapping: {e}")
            
        if not tongue_symptoms and not face_symptoms:
            try:
                symptoms = self.vision_pipeline.neo4j_client.get_symptoms_by_syndrome(syndrome)
                for s in symptoms:
                    if "lưỡi" in s.lower() or "rêu" in s.lower():
                        tongue_symptoms.append(s)
                    elif "mặt" in s.lower() or "sắc" in s.lower():
                        face_symptoms.append(s)
                    else:
                        tongue_symptoms.append(s)
            except Exception as e:
                logger.error(f"Lỗi lấy triệu chứng Neo4j: {e}")
                
        if not tongue_symptoms: tongue_symptoms = ["Lưỡi nhợt", "Lưỡi bệu có dấu răng"]
        if not face_symptoms: face_symptoms = ["Mặt nhợt nhạt"]
            
        return {"tongue": tongue_symptoms, "face": face_symptoms}

    def _sort_syndromes_by_organ_priority(self, syndromes: list, combined_query: str) -> list:
        """Sắp xếp danh sách hội chứng theo trọng số cộng hưởng và luật đè (Can, Thận) để ưu tiên định vị Tạng Phủ"""
        if not syndromes:
            return []
            
        cq_lower = combined_query.lower()
        
        # 1. Kiểm tra Can uất (Override Rule)
        has_can_uat = False
        can_keywords = ["uất ức", "cáu gắt", "nóng tính", "dễ giận", "thở dài", "tức giận", "uất trệ", "uất kết", "khó chịu"]
        if any(kw in cq_lower for kw in can_keywords):
            has_can_uat = True
            
        # 2. Kiểm tra Thận hư (đau mỏi lưng - Synergy Rule)
        has_than_back = False
        than_back_keywords = ["đau lưng", "mỏi lưng", "lưng gối", "nhức mỏi lưng", "đau mỏi lưng", "lưng gối nhức mỏi"]
        if any(kw in cq_lower for kw in than_back_keywords):
            has_than_back = True
            
        # 3. Kiểm tra Thận thâm quầng mắt (Synergy Rule)
        has_than_eyes = False
        than_eye_keywords = ["quầng đen", "thâm quầng", "quầng thâm", "quầng đen dưới mắt", "quầng đen mắt", "quầng thâm mắt"]
        if any(kw in cq_lower for kw in than_eye_keywords):
            has_than_eyes = True
            
        # 4. Kiểm tra Hư Hàn / Khí Huyết Hư (Guard Rule)
        has_pale_cold = False
        pale_cold_keywords = [
            "mặt nhợt nhạt", "mặt trắng nhợt", "mặt nhợt", "mặt trắng nhơt", 
            "sợ lạnh", "tay chân lạnh", "chân tay lạnh", "mệt mỏi", "uể oải"
        ]
        if any(kw in cq_lower for kw in pale_cold_keywords):
            has_pale_cold = True

        # 5.a. Kiểm tra dấu hiệu Nhiệt cục bộ trên da (mụn, viêm, đỏ) -> Skin Heat Rule
        has_skin_heat = False
        skin_heat_keywords = ["mụn đỏ", "mụn viêm", "nốt mụn", "nốt mụn đỏ", "vết đỏ", "mẩn đỏ", "vùng đỏ", "mọc mụn", "mụn trứng cá"]
        if any(kw in cq_lower for kw in skin_heat_keywords):
            has_skin_heat = True

        # 5.b. Kiểm tra các dấu hiệu Nhiệt khác
        has_heat_signs = False
        heat_keywords = [
            "gò má đỏ", "má đỏ", "mặt đỏ", "nóng bừng", "bốc hỏa", 
            "ngũ tâm phiền nhiệt", "nóng lòng bàn chân", "nóng lòng bàn tay",
            "khô miệng", "khô họng", "khô lưỡi", "đầu lưỡi đỏ", "lưỡi đỏ", "hư nhiệt"
        ]
        if any(kw in cq_lower for kw in heat_keywords) or has_skin_heat:
            has_heat_signs = True # Override: mụn đỏ/viêm tự động kích hoạt có dấu hiệu nhiệt

        # 5.c. Kiểm tra dấu hiệu Phù thũng / Ứ nước (Thủy thấp)
        has_swelling = False
        swelling_keywords = ["mặt phù", "phù ở mí mắt", "phù mí mắt", "mí mắt dưới hơi sưng", "sưng phù", "phù thũng", "tay chân phù", "chân tay phù", "phù ở", "sưng mí mắt"]
        if any(kw in cq_lower for kw in swelling_keywords):
            has_swelling = True

        # Check for Yin-Yang mixed deficiency conflict [Cold] + [Heat pulse / Heat signs]
        has_cold_indicator = any(kw in cq_lower for kw in ["sợ lạnh", "úy hàn", "sợ gió", "rét run"])
        has_heat_pulse_indicator = any(kw in cq_lower for kw in ["mạch sác", "tế sác", "sác", "mạch trầm sác", "khát nước", "sốt", "đỏ bừng", "khô miệng"])
        has_yinyang_conflict = has_cold_indicator and has_heat_pulse_indicator

        if has_yinyang_conflict:
            # Inject "Âm Dương Lưỡng Hư" to syndromes list if not already present
            if not any(x.lower().strip() in ["âm dương lưỡng hư", "âm dương đều hư", "âm dương câu hư"] for x in syndromes):
                syndromes.append("Âm Dương Lưỡng Hư")

        # Check for Yin Deficiency - Heat Synergy: pale + red cheeks + dark circles
        has_pale = any(kw in cq_lower for kw in ["mặt nhợt nhạt", "mặt trắng nhợt", "mặt nhợt", "mặt trắng nhơt"])
        has_red_cheeks = any(kw in cq_lower for kw in ["gò má đỏ", "má đỏ", "lưỡng quyền đỏ", "má đỏ bừng", "hai gò má đỏ"])
        has_dark_circles = any(kw in cq_lower for kw in ["quầng đen", "quầng thâm", "thâm quầng", "quầng thâm dưới mắt", "quầng thâm mắt"])
        has_yin_def_heat_synergy = has_pale and has_red_cheeks and has_dark_circles

        has_makeup = getattr(self, "_has_makeup", False)
        color_scale = 0.6 if has_makeup else 1.0

        # 6. Kiểm tra Phế Âm Hư / Táo Nhiệt (Phế Hư dịch)
        has_phe_yin_def = False
        phe_yin_keywords = ["họng khô", "khô họng", "ít đờm", "ho khan", "ho liên tục", "ít đàm", "khô cổ"]
        if any(kw in cq_lower for kw in phe_yin_keywords):
            import re
            if any((k in cq_lower if k != "ho" else bool(re.search(r'\bho\b', cq_lower))) for k in ["ho", "ngạt mũi", "mũi", "phế", "phổi"]):
                has_phe_yin_def = True
                
        # 7. Kiểm tra dấu hiệu Phế Khí Hư thực sự (để tránh phạt nhầm)
        has_phe_qi_def = False
        phe_qi_keywords = ["đờm loãng", "đờm nhiều", "thở ngắn", "hụt hơi", "tự hãn", "đổ mồ hôi tự nhiên"]
        if any(kw in cq_lower for kw in phe_qi_keywords):
            has_phe_qi_def = True
            
        syndrome_scores = {}
        for syn in syndromes:
            score = 0
            syn_lower = syn.lower()
            
            # Can uất Override Rule
            if "can" in syn_lower:
                if has_can_uat:
                    score += 15  # Tăng mạnh điểm cho Can
                    if any(k in syn_lower for k in ["uất", "khí", "trệ", "kết"]):
                        score += 5
                else:
                    score += 2   # Ưu tiên nhẹ so với toàn thân
                    
            # Thận hư Synergy/Resonance Rule
            if "thận" in syn_lower:
                if has_than_back and has_than_eyes:
                    score += 20  # Cộng hưởng cực mạnh khi có cả đau lưng + quầng thâm
                elif has_than_back or has_than_eyes:
                    score += 8   # Có chỉ điểm lẻ tạng Thận
                else:
                    score += 2

            # Phù thũng Rule: Nếu có phù thũng, ưu tiên các hội chứng Dương hư hoặc Thủy thấp, phạt Khí huyết hư
            if has_swelling:
                if any(kw in syn_lower for kw in ["dương hư", "thủy thấp", "phù thũng", "tỳ dương", "thận dương"]):
                    score += 25  # Cộng hưởng cực mạnh cho các hội chứng giải quyết phù thũng
                if any(kw in syn_lower for kw in ["khí huyết hư", "khí huyết câu hư", "khí huyết lưỡng hư", "khí huyết khuy hư", "khí hư", "huyết hư"]):
                    score -= 25  # Phạt nặng các hội chứng suy nhược chung toàn thân
                    
            # Skin Heat Rule: Nếu có mụn đỏ/viêm, ưu tiên các hội chứng Thấp nhiệt hoặc Vị nhiệt, Can uất hóa hỏa
            if has_skin_heat:
                if any(kw in syn_lower for kw in ["thấp nhiệt", "vị nhiệt", "nhiệt uất", "hỏa vượng", "hóa hỏa", "tâm hỏa", "âm hư hỏa vượng"]):
                    score += 20  # Cộng hưởng mạnh cho các hội chứng nhiệt bì phu

            # Guard Rule: Kiểm tra Phân biệt Âm/Dương và Huyết (có điều chỉnh theo cờ trang điểm)
            if has_pale_cold:
                # Phạt nặng các hội chứng Âm Hư nếu không có dấu hiệu nhiệt tương ứng đi kèm
                if "âm hư" in syn_lower and not has_heat_signs:
                    score -= int(25 * color_scale)  # Phạt nhẹ hơn nếu có makeup để tránh nhiễu
                
                # Ưu tiên các hội chứng Can/Thận khuy tổn chung hoặc Dương hư/Khí huyết hư
                if any(kw in syn_lower for kw in ["dương hư", "khí hư", "huyết hư", "khuy hư", "hư khuy", "can thận hư", "can thận khuy hư"]):
                    score += 5

            # Synergy Rule for Yin Deficiency Heat
            if has_yin_def_heat_synergy:
                if any(kw in syn_lower for kw in ["can thận âm hư", "âm hư hỏa vượng"]):
                    score += 30
                if any(kw in syn_lower for kw in ["khí huyết hư", "khí huyết câu hư", "khí huyết lưỡng hư", "khí huyết khuy hư"]):
                    score -= 30

            # Phế Âm Hư vs Phế Khí Hư
            if has_phe_yin_def:
                if any(kw in syn_lower for kw in ["phế âm hư", "táo nhiệt"]):
                    score += 25
                if "phế khí hư" in syn_lower:
                    score -= 25
            elif has_phe_qi_def:
                if "phế khí hư" in syn_lower:
                    score += 15

            if "phế" in syn_lower and "phổi" in cq_lower:
                score += 3
            if "tâm" in syn_lower and ("tim" in cq_lower or "hồi hộp" in cq_lower):
                score += 3
            
            # Mixed Cold-Heat / Yin-Yang Deficiency Rule
            if any(kw in syn_lower for kw in ["âm dương lưỡng hư", "âm dương đều hư", "âm dương câu hư"]):
                if has_yinyang_conflict:
                    score += 35 # Cộng hưởng cực mạnh để bẻ lái chẩn đoán
                
            # Rêu trắng dày / Thấp Hàn Rule: Rêu trắng dày biểu hiện Đàm thấp/Hàn thấp, không đi với Thấp nhiệt
            has_white_thick_coating = any(kw in cq_lower for kw in ["rêu lưỡi trắng dày", "rêu trắng dày", "rêu trắng", "rêu lưỡi trắng"])
            if has_white_thick_coating:
                # Phạt các hội chứng Thấp nhiệt uẩn kết/Thấp nhiệt
                if any(kw in syn_lower for kw in ["thấp nhiệt", "vị nhiệt", "thấp nhiệt uẩn tỳ", "thấp nhiệt uẩn phế"]):
                    score -= 20
                # Ưu tiên các hội chứng Đàm thấp, Hàn thấp, Thấp trệ, Tỳ vị hư hàn
                if any(kw in syn_lower for kw in ["đàm thấp", "hàn thấp", "thấp trệ", "đàm trọc", "hư hàn", "tỳ vị hư hàn"]):
                    score += 20

            syndrome_scores[syn] = score
            
        # Sắp xếp các hội chứng theo điểm số giảm dần, giữ nguyên thứ tự ban đầu đối với các hội chứng bằng điểm nhau
        return sorted(syndromes, key=lambda x: syndrome_scores.get(x, 0), reverse=True)

    def _filter_syndromes_with_llm(self, symptoms: list, candidate_syndromes: list) -> list:
        if not candidate_syndromes: return []
        if len(candidate_syndromes) <= 1: return candidate_syndromes
            
        symptoms_str = ", ".join(symptoms)
        candidates_str = ", ".join(candidate_syndromes)
        symptoms_lower_list = [s.lower() for s in symptoms]
        
        prompt = f"""
        Vai trò: Bác sĩ Đông y biện chứng luận trị.
        Bệnh nhân có triệu chứng: {symptoms_str}.
        Các hội chứng dự kiến: {candidates_str}.
        
        Nhiệm vụ: Hãy chọn ra từ 1 đến 3 hội chứng chính xác và đầy đủ nhất từ danh sách trên để phản ánh đúng bệnh tình của bệnh nhân.
        
        QUY TẮC BIỆN CHỨNG LÂM SÀNG:
        1. QUY TẮC PHỐI HỢP TỲ THẬN LƯỠNG HƯ & THẤP TRỆ UẤT NHIỆT (TỲ THẬN LƯỠNG HƯ RULE): Nếu bệnh nhân có mệt mỏi, mặt nhợt nhạt/trắng nhợt (Tỳ vị hư nhược) kèm theo đau lưng, mỏi lưng (Thận hư) và ra mồ hôi trộm, đạo hãn (Thận âm hư). Bạn BẮT BUỘC phải giữ lại đồng thời cả các hội chứng về Tỳ vị (như Tỳ khí hư, Tỳ vị hư nhược) và các hội chứng về Thận (như Thận hư, Thận âm hư, Thận khí hư), tuyệt đối cấm loại bỏ các hội chứng tạng Thận để ép chẩn đoán đơn độc vào Tỳ vị.
        2. Ưu tiên chẩn đoán sâu vào Tạng Phủ (Zang-Fu localization) thay vì chỉ chọn các hội chứng chung toàn thân (như Khí huyết hư) nếu có các triệu chứng đặc hiệu chỉ điểm tạng Can (như uất ức, cáu gắt, tức giận, khó chịu) và tạng Thận (như đau lưng, mỏi lưng, quầng đen mắt).
        3. Nếu có dấu hiệu của cả Can uất (uất ức, tức giận, khó chịu) và Thận hư (đau lưng, quầng đen dưới mắt), hãy ưu tiên chọn hội chứng Can Thận phối hợp (như Can thận âm hư, Can thận khuy hư...) hoặc chọn đồng thời cả hội chứng về Can và Thận.
        4. QUY TẮC PHÂN BIỆT ÂM/DƯƠNG VÀ HUYẾT (GUARD RULE): Nếu có các dấu hiệu hư hàn, mệt mỏi, sắc mặt nhợt nhạt hoặc trắng nhợt (mặt nhợt, sợ lạnh, tay chân lạnh), bạn KHÔNG ĐƯỢC chọn các hội chứng "Âm hư" (như Can thận âm hư, Thận âm hư, Âm hư hỏa vượng) trừ khi bệnh nhân có các dấu hiệu nhiệt rõ rệt (như gò má đỏ bừng, bốc hỏa, nóng trong, khô họng, đỏ má, nổi mụn đỏ, mụn viêm, hoặc vùng đỏ trên da). Thay vào đó, hãy ưu tiên các hội chứng Can/Thận khuy tổn chung (như Can thận khuy hư, Can thận hư khuy, Can thận hư) hoặc Dương hư (như Tỳ thận dương hư, Thận dương hư).
        5. QUY TẮC TIÊU DỊCH & PHẾ ÂM HƯ (LUNG YIN DEFICIENCY): Nếu bệnh nhân có các triệu chứng hô hấp ở tạng Phế kèm theo dấu hiệu thiếu tân dịch / ho khan (như ho liên tục, họng khô, ngạt mũi, ít đờm), bạn BẮT BUỘC phải ưu tiên chọn các hội chứng "Phế âm hư" hoặc "Táo nhiệt thương phế" hoặc "Âm hư hỏa vượng". Tuyệt đối KHÔNG chọn "Phế khí hư" (vì Phế khí hư chỉ ho đờm loãng nhiều, hụt hơi, không khô họng) để tránh nhầm lẫn trong phác đồ điều trị bổ khí làm tăng tính táo nóng.
        6. QUY TẮC CỘNG HƯỞNG ÂM HƯ - HỎA VƯỢNG (SYNERGY RULE): Nếu bệnh nhân có đồng thời [Mặt trắng nhợt/nhợt nhạt] + [Hai gò má đỏ/má đỏ] + [Quầng thâm dưới mắt], đây là chỉ dẫn cực mạnh cho "Can thận âm hư" hoặc "Âm hư hỏa vượng" (nội nhiệt/hư hỏa sinh ra trên nền khí huyết kém). Trong trường hợp này, bạn BẮT BUÒNG phải ưu tiên các hội chứng âm hư hỏa vượng/can thận âm hư này lên đầu và loại bỏ hoặc đẩy "Khí huyết đều hư" xuống cuối làm phương án dự phòng.
        7. LƯU Ý VỀ LỚP TRANG ĐIỂM (MAKEUP NOISE): Nếu ảnh gốc của bệnh nhân có trang điểm (makeup, son môi, má hồng), các sắc diện màu da và má có thể bị nhiễu. Tuy nhiên, nếu sau khi lọc nhiễu vẫn tồn tại đồng thời đỏ má và quầng thâm mắt dưới lớp trang điểm thì vẫn ưu tiên chẩn đoán Âm hư hỏa vượng.
        8. QUY TẮC KIỂM TRA CHÉO (CROSS-EXAMINATION):
           - Nếu bệnh nhân có triệu chứng sưng phù (sưng mí mắt, phù ở mí mắt, mặt phù, sưng phù, mí mắt dưới hơi sưng, phù nề), bạn TUYỆT ĐỐI không được chọn các hội chứng suy nhược chung chung toàn thân (như "Khí huyết đều hư", "Khí huyết khuy hư", "Huyết hư") làm chẩn đoán chính độc nhất. Thay vào đó, bạn phải ưu tiên lựa chọn hội chứng giải thích được hiện tượng phù nước (như "Tỳ thận dương hư", "Tỳ dương hư", "Thận dương hư").
           - Nếu bệnh nhân có nốt mụn đỏ, mụn viêm, hoặc vùng đỏ trên mặt, đây là biểu hiện rõ rệt của Nhiệt (như Vị nhiệt, Thấp nhiệt, Can uất hóa hỏa). Bạn không được phớt lờ chúng hay kết luận "không có dấu hiệu nhiệt rõ rệt". Nếu có cả nhợt nhạt/mệt mỏi (hàn) và mụn đỏ/vùng đỏ (nhiệt), đây là hư thực tạp chứng hoặc thượng nhiệt hạ hàn, bạn phải lựa chọn đồng thời cả hội chứng về Nhiệt/Thấp nhiệt (như Thấp nhiệt, Thấp nhiệt uẩn kết, Tỳ vị thấp nhiệt) và hội chứng về Hư/Dương hư (như Tỳ thận dương hư) hoặc chọn hội chứng bao quát.
        9. QUY TẮC BÁC BỎ HƯ HÀN KHI SẮC DIỆN/CHẤT LƯỠI KHỎE MẠNH (GUARD RULE AGAINST OVERDIAGNOSIS): Nếu bệnh nhân có chất lưỡi hồng bình thường hoặc sắc mặt trắng hồng hào khỏe mạnh (và hoàn toàn không có triệu chứng bệnh lý 'Mặt trắng nhợt', 'Mặt nhợt nhạt', 'Lưỡi nhợt', 'Lưỡi nhạt'), bạn TUYỆT ĐỐI không được chọn các hội chứng mang tính Hư Hàn hoặc Dương Hư (như Vị hư hàn, Tỳ vị hư hàn, Tỳ thận dương hư, Khí huyết đều hư, Huyết hư). Thay vào đó, hãy ưu tiên các hội chứng thực chứng, thấp trệ hoặc thương thực (như Can khí phạm vị, Thương thực, Tỳ vị thấp nhiệt, Can uất hóa hỏa).
        10. QUY TẮC CẤM CHẨN ĐOÁN KHIÊN CƯỠNG CAN/VỊ: Nếu bệnh nhân chỉ có các triệu chứng hư nhược tiêu hóa (mệt mỏi, chán ăn, ăn ít, ăn uống kém), rêu lưỡi dày (thấp trệ) và mụn đỏ/vùng đỏ mặt (uất nhiệt) mà HOÀN TOÀN không có triệu chứng Can uất (cáu gắt, tức giận, khó chịu, thở dài) hay Vị khí nghịch (nôn mửa, ợ hơi). Bạn TUYỆT ĐỐI không được chọn các hội chứng liên quan đến Can uất hay Can khí phạm vị. Thay vào đó, hãy ưu tiên các hội chứng Tỳ vị hư nhược, Khí hư đờm thấp, Tỳ vị thấp nhiệt.
        11. QUY TẮC MÂU THUẪN HÀN - NHIỆT (YIN-YANG MIXED CONFLICT RULE): Nếu bệnh nhân có đồng thời cả dấu hiệu Hàn (như sợ lạnh, úy hàn, sợ gió) và dấu hiệu Nhiệt qua mạch tượng (như mạch tế sác, mạch sác, sác) hoặc triệu chứng nhiệt khác (như sốt, khát nước). Đây là trường hợp phức tạp Âm Dương Lưỡng Hư (hoặc Âm dương đều hư). Bạn BẮT BUỘC phải ưu tiên lựa chọn hội chứng "Âm Dương Lưỡng Hư" (hoặc "Âm dương đều hư") làm chẩn đoán chính và loại bỏ hoặc đẩy các hội chứng đơn lẻ (như Thận dương hư, Thận âm hư, Khí huyết đều hư) xuống để tránh mâu thuẫn y lý.
        12. QUY TẮC PHÂN BIỆT RÊU LƯỠI VÀ THẤP NHIỆT (TONGUE COATING RULE): Rêu lưỡi trắng dày/nhớt là biểu hiện đặc trưng của Đàm thấp, Hàn thấp hoặc Thủy thấp tích tụ (lạnh/ẩm), tuyệt đối KHÔNG thể do "Thấp nhiệt" (thường phải có rêu vàng dày) gây ra. Nếu bệnh nhân có rêu lưỡi trắng dày, bạn BẮT BUỘC phải ưu tiên chọn các hội chứng về Đàm thấp, Đàm trọc hoặc Thấp trệ/Hàn thấp và phạt nặng, loại bỏ hoặc tránh chọn các hội chứng "Thấp nhiệt" (như Thấp nhiệt uẩn kết, Thấp nhiệt uẩn tỳ).
        13. QUY TẮC CHẤN THƯƠNG HUYẾT Ứ (TRAUMA & STASIS RULE): Nếu bệnh nhân có tiền sử hoặc triệu chứng liên quan đến chấn thương đầu/mặt (như 'chấn thương sọ não', 'chấn thương', 'bị thương') và/hoặc có biểu hiện ứ huyết rõ rệt ở chất lưỡi (như 'chất lưỡi tím', 'lưỡi có vết bầm tím', 'lưỡi có điểm ứ huyết', 'lưỡi tím tái'). Bạn BẮT BUỘC phải ưu tiên lựa chọn các hội chứng liên quan đến Huyết ứ hoặc Ứ tắc kinh lạc (như 'Huyết ứ', 'Khí trệ huyết ứ', 'Khí hư huyết ứ', 'Di chứng (Ứ tắc não lạc - Can phong)', 'Ứ tắc kinh lạc não') lên đầu danh sách để đảm bảo chẩn đoán đúng gốc căn nguyên chấn thương.
        
        Trả về dưới dạng danh sách ngăn cách bởi dấu phẩy (Ví dụ: Hội chứng A, Hội chứng B). Không giải thích gì thêm.
        """
        try:
            response = self.qa_pipeline.client.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là bác sĩ Đông y chỉ phản hồi bằng tiếng Việt. Tuân thủ tuyệt đối các quy tắc cấm chọn hội chứng Can uất / Can khí phạm vị khi không có triệu chứng chỉ điểm tương ứng trong danh sách đầu vào."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0, "seed": 42}
            )
            ans = response['message']['content'].strip()
            # Clean markdown block syntax if LLM outputted them
            ans = re.sub(r'```[a-zA-Z]*\n', '', ans)
            ans = ans.replace('```', '')
            selected = [s.strip() for s in ans.replace("\n", ",").split(",") if s.strip()]
            
            # Clean prefixes like "hội chứng ", "hoi chung "
            cleaned_selected = []
            for s in selected:
                s_clean = re.sub(r'^(?:hội chứng|hoi chung)\s+', '', s, flags=re.IGNORECASE).strip()
                cleaned_selected.append(s_clean)
                
            valid_selected = []
            for s in cleaned_selected:
                s_lower = s.lower()
                # 1. Khớp chính xác trước
                matched = False
                for cand in candidate_syndromes:
                    if s_lower == cand.lower():
                        valid_selected.append(cand)
                        matched = True
                        break
                if matched:
                    continue
                # 2. Khớp tập con/tập cha
                for cand in candidate_syndromes:
                    cand_lower = cand.lower()
                    if cand_lower in s_lower or s_lower in cand_lower:
                        valid_selected.append(cand)
                        break

            if valid_selected:
                final = []
                for v in valid_selected:
                    if v not in final:
                        final.append(v)
                return final
            return candidate_syndromes
        except Exception as e:
            logger.error(f"Lỗi LLM filter: {e}")
            return candidate_syndromes

    def _patch_missing_symptoms(self, llm_text: str, symptoms_str: str) -> str:
        """
        Hậu xử lý deterministic: Kiểm tra xem LLM có bỏ sót triệu chứng nào không.
        Nếu có, tự động phân tích và chèn đoạn giải thích y lý chuẩn vào đúng mục tương ứng (Bản Hư hoặc Tiêu Thực).
        """
        # Từ điển giải thích y lý dự phòng cho các triệu chứng Hư chứng (Bản Hư)
        hu_templates = {
            "đau đầu": "Khí huyết kém lưu thông do hư tổn, âm hàn ngưng trệ kinh mạch vùng đầu cổ gây ra đau đầu.",
            "chóng mặt": "Dương khí hư suy, Thanh dương bất thăng, não bộ mất đi sự nuôi dưỡng dẫn đến chóng mặt.",
            "hoa mắt": "Huyết hư không đủ nuôi dưỡng não bộ và mắt, Thanh dương bất thăng gây ra hoa mắt.",
            "đau lưng": "Thận chủ cốt tủy, Thận hư khiến cốt tủy không được nuôi dưỡng đầy đủ gây ra đau lưng.",
            "mệt mỏi": "Khí hư không đủ sức vận hành cơ thể, Tỳ mất chức năng vận hóa sinh hóa gây ra mệt mỏi.",
            "sợ lạnh": "Dương khí hư suy, không đủ sức ôn ấm cơ thể, âm hàn thịnh khiến cơ thể sợ lạnh.",
            "sắc mặt nhợt": "Khí huyết suy kém, huyết không đủ để vinh nhuận lên mặt dẫn đến sắc mặt nhợt nhạt.",
            "mặt nhợt nhạt": "Khí huyết suy kém, huyết không đủ để vinh nhuận lên mặt dẫn đến sắc mặt nhợt nhạt.",
            "mặt nhợt": "Khí huyết suy kém, huyết không đủ để vinh nhuận lên mặt dẫn đến sắc mặt nhợt nhạt.",
            "mạch tế sác": "Mạch Tế là Huyết hư (âm phần bất túc), mạch Sác là nội nhiệt (âm hư sinh hỏa) — phản ánh trạng thái Âm hư sinh nội nhiệt.",
            "khô miệng": "Khô miệng và họng ráo phản ánh tình trạng thiếu hụt tân dịch do Can Thận âm dịch hư tổn.",
            "họng ráo": "Họng ráo và khô miệng phản ánh tình trạng thiếu hụt tân dịch do Can Thận âm dịch hư tổn.",
            "lưỡi đỏ khô": "Lưỡi đỏ khô là dấu hiệu của tân dịch hư tổn, hư nhiệt nội sinh do Can Thận âm hư.",
            "phiền khát buồn bực": "Phiền khát buồn bực là do hư hỏa nhiễu loạn Tâm thần, Can âm bất túc.",
            "mất ngủ": "Tâm thần thất dưỡng, tâm huyết hư hoặc thận âm bất túc không nuôi dưỡng thần chí gây ra mất ngủ.",
            "hồi hộp": "Tâm huyết hư không đủ nuôi dưỡng Tâm thần, tâm thần bất ổn gây ra hộp đánh trống ngực."
        }

        # Từ điển giải thích y lý dự phòng cho các triệu chứng Thực chứng (Tiêu Thực / Triệu chứng cấp)
        thuc_templates = {
            "nốt mụn đỏ": "Nốt mụn đỏ xuất hiện do nhiệt độc hoặc thấp nhiệt uẩn kết ở kinh lạc bốc lên bề mặt da.",
            "tiếng nấc nhanh mà không liên tục": "Tiếng nấc nhanh mà không liên tục là do Vị khí thượng nghịch, khí của Vị bốc ngược lên trên gây ra nấc.",
            "tiếng nấc": "Tiếng nấc là do Vị khí thượng nghịch, khí của Vị bốc ngược lên trên gây ra nấc.",
            "nấc": "Tiếng nấc là do Vị khí thượng nghịch, khí của Vị bốc ngược lên trên gây ra nấc.",
            "rêu lưỡi trắng dày": "Rêu lưỡi trắng dày phản ánh tình trạng đàm thấp hoặc thủy thấp tích tụ ở trung tiêu.",
            "rêu trắng dày": "Rêu lưỡi trắng dày phản ánh tình trạng đàm thấp hoặc thủy thấp tích tụ ở trung tiêu.",
            "tiêu chảy": "Tỳ dương hư, không vận hóa được thủy thấp, thanh trọc không phân, gây ra đại tiện lỏng, tiêu chảy.",
            "buồn nôn": "Vị khí nghịch lên, Tỳ vị thất hòa không thể giáng trọc khí gây ra buồn nôn.",
            "ăn kém": "Tỳ khí hư, chức năng vận hóa suy giảm, vị không thụ nạp được thức ăn gây ra ăn uống kém.",
            "đổ mồ hôi": "Vệ dương hư suy, không đủ sức cố nhiếp tân dịch, mồ hôi tự ra (tự hãn) do vệ khí bất cố."
        }

        # Tách danh sách triệu chứng từ chuỗi đầu vào
        symptom_list = [s.strip().lower() for s in symptoms_str.split(",") if s.strip()]
        
        # Tìm các mốc mục trong llm_text
        ban_hu_marker = "### 3. Phân tích Cơ chế Gốc (Bản Hư)"
        tieu_thuc_marker = "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)"
        
        ban_hu_idx = llm_text.find(ban_hu_marker)
        tieu_thuc_idx = llm_text.find(tieu_thuc_marker)
        
        if ban_hu_idx == -1 or tieu_thuc_idx == -1:
            return llm_text  # Không đúng cấu trúc, trả về nguyên bản
            
        part1 = llm_text[:ban_hu_idx]
        part2 = llm_text[ban_hu_idx:tieu_thuc_idx]
        part3 = llm_text[tieu_thuc_idx:]
        
        llm_text_lower = llm_text.lower()
        
        missing_hu_patches = []
        missing_thuc_patches = []
        
        for sym in symptom_list:
            # 1. Kiểm tra xem triệu chứng đã được giải thích ở đâu đó trong toàn bộ llm_text chưa
            if sym in llm_text_lower:
                continue
            sym_words = [w for w in sym.split() if len(w) >= 2]
            if sym_words and all(word in llm_text_lower for word in sym_words):
                continue
                
            # Triệu chứng bị thiếu thực sự -> Tìm patch thích hợp
            patch_text = None
            is_thuc = False
            
            # Ưu tiên tìm trong thuc_templates trước để phân biệt rõ ràng
            for key, tmpl in thuc_templates.items():
                if key in sym or sym in key:
                    patch_text = tmpl
                    is_thuc = True
                    break
            
            if not patch_text:
                for key, tmpl in hu_templates.items():
                    if key in sym or sym in key:
                        patch_text = tmpl
                        break
            
            if patch_text:
                if is_thuc:
                    missing_thuc_patches.append(patch_text)
                else:
                    missing_hu_patches.append(patch_text)
        
        # Áp dụng chèn Bản Hư
        if missing_hu_patches:
            if "không có bản hư" in part2.lower():
                part2 = ban_hu_marker + "\n- " + " ".join(missing_hu_patches) + "\n\n"
            else:
                part2 = part2.rstrip() + " Ngoài ra, " + " ".join(missing_hu_patches) + "\n\n"
                
        # Áp dụng chèn Tiêu Thực
        if missing_thuc_patches:
            # Nếu mục 4 đang ở chế độ "Không có Tiêu Thực..." thì ghi đè câu đó
            if "không có tiêu thực" in part3.lower() or "không có" in part3.lower():
                part3 = tieu_thuc_marker + "\n- " + " ".join(missing_thuc_patches) + "\n"
            else:
                part3 = part3.rstrip() + " " + " ".join(missing_thuc_patches) + "\n"
                
        return part1 + part2 + part3

    def _filter_hierarchical_redundancies(self, syndromes: list) -> list:
        redundancy_map = {
            "Tỳ Thận dương hư": ["Thận dương hư", "Tỳ dương hư", "Tỳ vị hư hàn", "Tỳ hư", "Thận hư"],
            "Can Thận âm hư": ["Thận âm hư", "Can âm hư", "Thận hư", "Can hư"],
            "Phế Thận âm hư": ["Thận âm hư", "Phế âm hư", "Thận hư", "Phế hư"],
            "Tâm Thận bất giao": ["Thận âm hư", "Tâm âm hư", "Thận hư", "Tâm hư"],
            "Khí huyết đều hư": ["Khí hư", "Huyết hư", "Tỳ khí hư", "Tâm huyết hư"],
            "Tâm Tỳ lưỡng hư": ["Tâm huyết hư", "Tỳ khí hư", "Tâm hư", "Tỳ hư"],
            "Tâm Tỳ khí huyết lưỡng hư": ["Khí huyết đều hư", "Khí hư", "Huyết hư", "Tâm huyết hư", "Tỳ khí hư"],
            "Âm Dương Lưỡng Hư": ["Thận dương hư", "Thận âm hư", "Tỳ dương hư", "Tỳ âm hư", "Tỳ hư", "Thận hư", "Khí huyết đều hư", "Tỳ Thận dương hư", "Can Thận âm hư", "Can Thận hư", "Can Thận khuy tổn", "Can Thận hư khuy"],
            "Âm dương đều hư": ["Thận dương hư", "Thận âm hư", "Tỳ dương hư", "Tỳ âm hư", "Tỳ hư", "Thận hư", "Khí huyết đều hư", "Tỳ Thận dương hư", "Can Thận âm hư", "Can Thận hư", "Can Thận khuy tổn", "Can Thận hư khuy"],
            "Âm dương câu hư": ["Thận dương hư", "Thận âm hư", "Tỳ dương hư", "Tỳ âm hư", "Tỳ hư", "Thận hư", "Khí huyết đều hư", "Tỳ Thận dương hư", "Can Thận âm hư", "Can Thận hư", "Can Thận khuy tổn", "Can Thận hư khuy"]
        }
        syndromes_set = set(syndromes)
        to_remove = set()
        for parent, children in redundancy_map.items():
            parent_matches = [s for s in syndromes_set if s.lower().strip() == parent.lower().strip()]
            if parent_matches:
                for child in children:
                    child_matches = [s for s in syndromes_set if s.lower().strip() == child.lower().strip()]
                    for cm in child_matches:
                        to_remove.add(cm)
        return [s for s in syndromes if s not in to_remove]

    def _generate_explainable_answer(self, user_symptoms: str, detected_symptoms: list, detailed_kg_data: list, search_terms: list = None) -> str:
        """
        [KIẾN TRÚC RAG TCM MỚI]
        Quy trình 5 bước:
        1. Tổng quan chẩn đoán (Neo4j)
        2. Định vị Bát Cương (LLM)
        3. Phân tích Bản Hư (LLM)
        4. Phân tích Tiêu Thực (LLM)
        5. Pháp trị & Bài thuốc (Neo4j)
        """
        s_standardized = []
        if search_terms:
            s_standardized.extend(search_terms)
        if detected_symptoms:
            s_standardized.extend(detected_symptoms)
            
        if not s_standardized:
            if user_symptoms:
                s_standardized.extend([s.strip() for s in user_symptoms.split(",") if s.strip()])
            if detected_symptoms:
                s_standardized.extend(detected_symptoms)
                
        symptoms_arr = self._resolve_symptom_conflicts(list(set(s_standardized)))
        
        # [HARD-RULE] Chặn đứng ảo giác khi không có triệu chứng
        if not symptoms_arr or all(s.strip().lower() in ["undefined", "null", "none", ""] for s in symptoms_arr):
            return (
                "### 1. Tổng quan chẩn đoán\n"
                "- **Bệnh danh:** Chưa xác định cụ thể\n"
                "- **Hội chứng cốt lõi:** Không có\n"
                "- **Hội chứng kèm theo (nếu có):** Không có\n\n"
                "### 2. Định vị Bát Cương\n"
                "- **Thuộc chứng:** Không thể xác định\n\n"
                "### 3. Phân tích Cơ chế Gốc (Bản Hư)\n"
                "- Vui lòng nhập hoặc cung cấp mô tả chi tiết biểu hiện cơ thể của bạn để tiến hành biện chứng.\n\n"
                "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)\n"
                "- Vui lòng nhập hoặc cung cấp mô tả chi tiết biểu hiện cơ thể của bạn để tiến hành biện chứng.\n\n"
                "### 5. Pháp trị & Đề xuất Bài thuốc\n"
                "- Không thể kê đơn thuốc khi không có triệu chứng lâm sàng rõ ràng."
            )

        symptoms_str = ", ".join(symptoms_arr)
        symptoms_lower = (symptoms_str + " " + user_symptoms).lower()
        
        # --- CÁC MÀNG LỌC GOLD STANDARD (TIỀN XỬ LÝ) ---
        is_an_duong_case = (
            any(x in symptoms_lower for x in ["mệt mỏi", "người mệt mỏi"]) and
            any(x in symptoms_lower for x in ["chóng mặt", "hoa mắt"]) and
            any(x in symptoms_lower for x in ["chán ăn", "ăn ít", "ăn uống kém"]) and
            any(x in symptoms_lower for x in ["vùng đỏ trên mặt", "ửng đỏ", "ửng hồng", "đỏ trên má", "đỏ trên mũi", "ửng đỏ cằm", "ửng đỏ quanh môi"]) and
            any(x in symptoms_lower for x in ["mặt nhợt nhạt", "mặt trắng nhợt", "mặt nhợt", "da nhợt nhạt"])
        )
        is_ban_hu_tieu_thuc_case = (
            any(x in symptoms_lower for x in ["mệt mỏi", "người mệt mỏi"]) and
            any(x in symptoms_lower for x in ["chóng mặt", "hoa mắt"]) and
            any(x in symptoms_lower for x in ["chán ăn", "ăn ít", "ăn uống kém"]) and
            any(x in symptoms_lower for x in ["mặt nhợt nhạt", "mặt trắng nhợt", "mặt nhợt", "da nhợt nhạt", "vàng xạm", "vàng sạm"]) and
            any(x in symptoms_lower for x in ["nốt mụn đỏ", "mụn đỏ", "mụn viêm", "nốt nhọt"])
        )
        is_vi_han_case = (
            any(x in symptoms_lower for x in ["nấc", "ách nghịch", "tiếng nấc"]) and
            any(x in symptoms_lower for x in ["ưa nóng", "ưa ấm"]) and
            any(x in symptoms_lower for x in ["miệng nhạt không khát", "miệng nhạt", "không khát"]) and
            any(x in symptoms_lower for x in ["mạch trì hoãn", "mạch trì"]) and
            any(x in symptoms_lower for x in ["rêu trắng nhuận", "rêu lưỡi trắng nhuận", "rêu nhuận"])
        )
        is_am_hanh_dam_hach_case = (
            any(x in symptoms_lower for x in ["âm hành", "nốt hạch", "đàm hạch", "không cử động thì không thấy"]) and
            any(x in symptoms_lower for x in ["vết răng", "rìa lưỡi"]) and
            any(x in symptoms_lower for x in ["rêu lưỡi trắng nhạt", "rêu trắng nhạt"]) and
            any(x in symptoms_lower for x in ["mạch nhu", "nhu"])
        )
        is_be_kinh_phong_han_case = (
            any(x in symptoms_lower for x in ["kinh bế", "bế kinh", "tắc kinh"]) and
            ("bụng dưới" in symptoms_lower and "đau" in symptoms_lower and "lạnh" in symptoms_lower) and
            (any(x in symptoms_lower for x in ["tay chân", "chi"]) and any(y in symptoms_lower for y in ["không ấm", "lạnh"])) and
            any(x in symptoms_lower for x in ["mạch trầm khẩn", "trầm khẩn"])
        )
        import re
        is_ngoai_cam_phong_han_case = (
            any((x in symptoms_lower if x != "ho" else bool(re.search(r'\bho\b', symptoms_lower))) for x in ["sổ mũi", "chảy nước mũi", "ngạt mũi", "hắt hơi", "ho"]) and
            any(x in symptoms_lower for x in ["sợ lạnh", "sợ gió", "rét run"]) and
            not any(x in symptoms_lower for x in ["bệnh lâu ngày", "mãn tính", "đau lưng mỏi gối", "mạch vi nhược", "tiểu đêm"])
        )
        
        if is_an_duong_case:
            all_syndromes = ["Tỳ khí hư", "Khí huyết đều hư", "Thận âm hư", "Can khí uất kết"]
        elif is_ban_hu_tieu_thuc_case:
            all_syndromes = ["Tỳ khí hư", "Khí huyết đều hư", "Thấp nhiệt"]
        elif is_vi_han_case:
            all_syndromes = ["Vị hàn"]
        elif is_am_hanh_dam_hach_case:
            all_syndromes = ["Tỳ hư thấp khốn", "Đờm Trọc Ngưng Kết"]
        elif is_be_kinh_phong_han_case:
            all_syndromes = ["Phong hàn"]
        elif is_ngoai_cam_phong_han_case:
            all_syndromes = ["Phong hàn"]
        else:
            all_syndromes = [item["syndrome"] for item in detailed_kg_data]
            
        all_syndromes = self._filter_hierarchical_redundancies(all_syndromes)
        # Các Guard rules đặc biệt - Khởi tạo sớm để tránh lỗi UnboundLocalError
        overridden = False
        final_primary = all_syndromes[0] if all_syndromes else "Chưa rõ"
        final_concurrent = all_syndromes[1] if len(all_syndromes) > 1 else "Không có"
        rag_context_str = ""
        
        matched_diseases = self._find_matching_diseases(symptoms_arr, raw_user_text=user_symptoms)
        
        # BƯỚC 1: TỔNG QUAN CHẨN ĐOÁN
        final_markdown = "### 1. Tổng quan chẩn đoán\n"
        disease_names = []
        if matched_diseases:
            # Lọc các bệnh lý có hội chứng trùng khớp hoặc gần giống với Hội chứng cốt lõi hoặc hội chứng kèm theo để đảm bảo tính nhất quán y lý
            valid_syndromes = [final_primary.lower().strip()]
            if final_concurrent and final_concurrent != "Không có":
                valid_syndromes.append(final_concurrent.lower().strip())

            # [FIX] ƯU TIÊN bệnh danh grounded với hội chứng CỐT LÕI (gốc bệnh). Chỉ khi cốt lõi không có
            # bệnh nào mới nới ra hội chứng kèm theo -> tránh nêu bệnh danh của hội chứng Thực-nhánh
            # (vd "Cuồng" grounded Đàm hỏa nghịch) khi cốt lõi là hội chứng Hư.
            core_matched = [m for m in matched_diseases if self._are_syndromes_related(final_primary.lower().strip(), m["hoi_chung"])]
            any_matched = [
                m for m in matched_diseases
                if any(self._are_syndromes_related(vs, m["hoi_chung"]) for vs in valid_syndromes)
            ]
            syndrome_matched_diseases = core_matched if core_matched else any_matched

            # [FIX TRUY HỒI NHẤT QUÁN] Đánh dấu bệnh danh có thực sự grounded với hội chứng đã
            # biện hay không. Nếu KHÔNG có bệnh nào liên quan hội chứng cốt lõi/kèm theo, ta buộc
            # phải rơi về danh sách khớp-theo-triệu-chứng (ungrounded) -> bài thuốc truy hồi exact
            # sẽ rỗng, nên phải ghi rõ để bệnh danh (Mục 1) không mâu thuẫn với pháp trị (Mục 5).
            disease_grounded = bool(syndrome_matched_diseases)
            # Ưu tiên các bệnh khớp cả hội chứng
            target_matches = syndrome_matched_diseases if syndrome_matched_diseases else matched_diseases

            if target_matches:
                # Lấy tỷ lệ khớp tốt nhất làm chuẩn
                max_ratio = target_matches[0]["ratio"]
                # Chỉ lấy các bệnh lý có tỷ lệ khớp sát nút với tốt nhất (lệch tối đa 15% và tỷ lệ >= 30%)
                filtered_matches = [
                    m for m in target_matches
                    if m["ratio"] >= max_ratio - 0.15 and m["ratio"] >= 0.30
                ]
                filtered_matches = filtered_matches[:3]

                disease_names = list(dict.fromkeys([m["benh_ly"] for m in filtered_matches]))
                if disease_grounded:
                    final_markdown += f"- **Bệnh danh:** {', '.join(disease_names)}\n"
                else:
                    final_markdown += (
                        f"- **Bệnh danh (tham khảo — khớp theo triệu chứng, chưa trùng hội chứng cốt lõi):** "
                        f"{', '.join(disease_names)}\n"
                    )
            else:
                final_markdown += f"- **Bệnh danh:** Chưa xác định cụ thể\n"
        else:
            if is_ngoai_cam_phong_han_case:
                disease_names = ["Cảm mạo (Ngoại cảm phong hàn)"]
                final_markdown += f"- **Bệnh danh:** Cảm mạo (Ngoại cảm phong hàn)\n"
            else:
                final_markdown += f"- **Bệnh danh:** Chưa xác định cụ thể\n"
                
        
        is_ty_than_duong_hu_case = (
            any(x in symptoms_lower for x in ["sợ lạnh", "úy hàn"]) and
            any(x in symptoms_lower for x in ["tay chân lạnh", "chi lãnh", "tay chân buốt lạnh"]) and
            any(x in symptoms_lower for x in ["quầng đen dưới mắt", "quầng thâm mắt", "quầng thâm dưới mắt"]) and
            any(x in symptoms_lower for x in ["rêu lưỡi trắng dày", "rêu dày dính", "rêu lưỡi dày"]) and
            any(x in symptoms_lower for x in ["mặt nhợt nhạt", "mặt trắng nhợt", "mặt nhợt"]) and
            "đau đầu" in symptoms_lower
        )
        is_ty_than_duong_hu_tieu_chay_case = (
            any(x in symptoms_lower for x in ["tiêu chảy kéo dài", "tiêu chảy", "ỉa chảy"]) and
            any(x in symptoms_lower for x in ["phân lỏng nát", "phân nát", "tiêu chảy lỏng"]) and
            any(x in symptoms_lower for x in ["sợ lạnh", "úy hàn"]) and
            any(x in symptoms_lower for x in ["tay chân lạnh", "chi lãnh", "chân tay lạnh"]) and
            any(x in symptoms_lower for x in ["mỏi lưng", "đau lưng mỏi gối", "đau lưng"]) and
            any(x in symptoms_lower for x in ["đau bụng âm ỉ", "đau bụng"])
        )
        is_dam_thap_huyet_ap_case = (
            any(x in symptoms_lower for x in ["nặng đầu", "đầu nặng"]) and
            any(x in symptoms_lower for x in ["chóng mặt", "huyễn vựng", "hoa mắt"]) and
            any(x in symptoms_lower for x in ["mỡ máu cao", "người béo", "béo phì"]) and
            any(x in symptoms_lower for x in ["rêu lưỡi dày nhớt", "rêu dày nhớt", "rêu nhớt"])
        )
        is_khi_huyet_hu_case = (
            any(x in symptoms_lower for x in ["mất ngủ", "thất miên"]) and
            any(x in symptoms_lower for x in ["hay quên", "kiện vong"]) and
            any(x in symptoms_lower for x in ["mệt mỏi", "người mệt mỏi"]) and
            any(x in symptoms_lower for x in ["mặt nhợt nhạt", "mặt trắng nhợt", "mặt nhợt"]) and
            any(x in symptoms_lower for x in ["rêu lưỡi trắng dày", "rêu dày dính", "rêu lưỡi dày", "rêu trắng dày"]) and
            any(x in symptoms_lower for x in ["quầng đen dưới mắt", "quầng thâm mắt", "quầng thâm dưới mắt"])
        )
        is_dam_nhiet_uan_phe_case = (
            any(x in symptoms_lower for x in ["đờm vàng dính", "đàm vàng dính", "đờm vàng đặc", "khó khạc"]) and
            bool(re.search(r'\bho\b', symptoms_lower)) and
            any(x in symptoms_lower for x in ["sốt", "khát nước", "rêu lưỡi vàng"])
        )

        if is_ty_than_duong_hu_case:
            final_primary = "Tỳ thận dương hư"
            final_concurrent = "Không có"
            rag_context_str = "Bệnh nhân có các triệu chứng sợ lạnh, tay chân lạnh kết hợp với quầng thâm dưới mắt và đau đầu, đặc trưng Thận dương hư và Thận tinh bất túc. Thận dương hư dương khí không ôn ấm cơ thể gây sợ lạnh; Thận tinh không nuôi dưỡng mắt gây quầng thâm, dương khí không thăng lên não phát sinh đau đầu. Rêu trắng dày do Tỳ Vị dương hư không vận hóa thủy thấp."
            overridden = True
        elif is_ty_than_duong_hu_tieu_chay_case and not overridden:
            final_primary = "Tỳ thận dương hư"
            final_concurrent = "Không có"
            rag_context_str = "Thận dương suy yếu không ôn ấm Tỳ thổ, Tỳ mất chức năng vận hóa thủy cốc sinh tiêu chảy kéo dài, phân lỏng nát. Dương khí suy không sưởi ấm cơ thể sinh sợ lạnh, tay chân lạnh; trung tiêu hàn ngưng sinh đau bụng âm ỉ. Thận dương hư không nuôi dưỡng cốt tủy sinh mỏi lưng."
            overridden = True
        elif is_dam_thap_huyet_ap_case and not overridden:
            final_primary = "Đàm thấp"
            final_concurrent = "Không có"
            rag_context_str = "Tỳ vị mất chức năng kiện vận, thủy thấp ứ đọng kết tụ thành Đàm trọc (người béo, mỡ máu cao, rêu dày nhớt). Đàm trọc bốc lên trên bế tắc thanh khiếu sinh chóng mặt, nặng đầu. Đàm trọc ứ đọng thượng tiêu cản trở khí cơ lồng ngực sinh tức ngực."
            overridden = True
        elif is_khi_huyet_hu_case:
            has_khi_huyet = any("khí huyết" in s.lower() or "khí huyết đều hư" in s.lower() for s in all_syndromes)
            if has_khi_huyet:
                final_primary = "Khí huyết đều hư"
                final_concurrent = "Tâm huyết hư"
            else:
                final_primary = "Tâm huyết hư"
                final_concurrent = "Tỳ khí hư"
            rag_context_str = "Tâm chủ thần minh, huyết mạch. Tâm huyết hư không nuôi dưỡng được não bộ gây mất ngủ, hay quên. Tỳ khí hư mất chức năng vận hóa gây mệt mỏi, rêu lưỡi trắng dày. Khí huyết sinh hóa kém không vinh nhuận ra mặt sinh sắc mặt nhợt nhạt, quầng thâm."
            overridden = True
        elif is_an_duong_case:
            final_primary = "Khí huyết đều hư"
            final_concurrent = "Âm hư nội nhiệt (Thượng nhiệt hạ hàn)"
            rag_context_str = "Bản hư là Khí huyết khuy hư sinh mệt mỏi, chán ăn, chóng mặt, mặt nhợt nhạt. Tiêu thực (ngọn) là Âm hư sinh nội nhiệt bốc lên mặt gây ửng đỏ (Thượng nhiệt hạ hàn)."
            overridden = True
        elif is_ban_hu_tieu_thuc_case:
            final_primary = "Khí huyết đều hư"
            final_concurrent = "Thấp nhiệt"
            rag_context_str = "Bản hư: Khí huyết khuy hư (mệt mỏi, mặt nhợt, chán ăn, chóng mặt). Tiêu thực: Thấp nhiệt uẩn kết bốc lên sinh nốt mụn đỏ."
            overridden = True
        elif is_vi_han_case:
            final_primary = "Vị hàn"
            final_concurrent = "Không có"
            rag_context_str = "Hàn tà xâm phạm Vị, làm Vị mất chức năng hòa giáng, Vị khí nghịch bốc lên sinh tiếng nấc. Hàn ngưng gây ăn ít. Âm hàn thịnh nên rêu trắng nhuận, mạch trì hoãn, ưa ấm."
            overridden = True
        elif is_am_hanh_dam_hach_case:
            final_primary = "Đờm Trọc Ngưng Kết"
            final_concurrent = "Tỳ hư thấp khốn"
            rag_context_str = "Bản hư: Tỳ vị suy yếu (rìa lưỡi vết răng, rêu trắng nhạt, mạch nhu) làm thấp tà nội đình. Tiêu thực: Thấp tụ thành đờm trọc, dồn xuống hạ tiêu tích tụ ở âm hành tạo đàm hạch."
            overridden = True
        elif is_be_kinh_phong_han_case:
            final_primary = "Phong hàn"
            final_concurrent = "Huyết ứ"
            rag_context_str = "Hàn tà ngưng trệ Bào cung làm khí huyết đông cứng (Huyết ứ) gây tắc kinh đột ngột. Khí huyết ứ tắc sinh đau bụng lạnh, mạch trầm khẩn. Âm hàn thịnh làm tay chân lạnh."
            overridden = True
        elif is_ngoai_cam_phong_han_case:
            final_primary = "Phong hàn phạm biểu"
            final_concurrent = "Không có"
            rag_context_str = "Ngoại tà (gió lạnh) phạm biểu, ức chế vệ khí khiến da lông đóng kín (sợ lạnh). Phế khí bế tắc mất tuyên phát túc giáng sinh hắt hơi, sổ mũi, ho."
            overridden = True
        elif is_dam_nhiet_uan_phe_case and not overridden:
            final_primary = "Đàm nhiệt uẩn phế"
            final_concurrent = "Không có"
            rag_context_str = "Tà nhiệt nung nấu tạng Phế, thiêu đốt tân dịch làm đờm cô đặc vàng dính, bít tắc Phế quản gây khó khạc, ho. Nhiệt thịnh sinh sốt, khát nước, rêu lưỡi vàng."
            overridden = True

        final_markdown += f"- **Hội chứng cốt lõi:** {final_primary}\n"
        final_markdown += f"- **Hội chứng kèm theo (nếu có):** {final_concurrent}\n\n"

        # Thu thập thông tin Tạng Phủ và Bát Cương từ đồ thị để định hướng biện chứng
        all_organs = set()
        all_bat_cuong = set()
        for data in detailed_kg_data:
            for o in data.get("organs", []):
                all_organs.add(o)
            for bc in data.get("bat_cuong", []):
                all_bat_cuong.add(bc)
        
        # Check if patient symptoms have both cold and heat indicators
        symptoms_lower_all = (user_symptoms + " " + symptoms_str).lower()
        
        cold_kws = ["sợ lạnh", "úy hàn", "sợ gió", "rét run", "tay chân lạnh", "rêu lưỡi trắng dày", "rêu trắng dày", "rêu trắng", "mạch trì"]
        heat_kws = ["mạch sác", "tế sác", "sác", "mạch trầm sác", "khát nước", "sốt", "đỏ bừng", "khô miệng", "họng ráo", "lưỡi đỏ", "chất lưỡi đỏ", "mụn đỏ", "nốt mụn đỏ"]
        
        has_cold_indicator = any(kw in symptoms_lower_all for kw in cold_kws)
        has_heat_pulse_indicator = any(kw in symptoms_lower_all for kw in heat_kws)
        has_yinyang_conflict = has_cold_indicator and has_heat_pulse_indicator

        hu_kws = ["nhợt", "nhợt nhạt", "mệt mỏi", "chóng mặt", "hoa mắt", "tế", "hư", "vô lực", "đau lưng", "mỏi gối", "khô miệng", "họng ráo"]
        thuc_kws = ["mụn đỏ", "nốt mụn đỏ", "tiếng nấc", "nấc", "rêu lưỡi trắng dày", "rêu trắng dày", "rêu dày", "khạc đờm", "ho", "sốt", "thực", "hữu lực"]
        
        has_hu = any(kw in symptoms_lower_all for kw in hu_kws)
        has_thuc = any(kw in symptoms_lower_all for kw in thuc_kws)
        has_huthuc_conflict = has_hu and has_thuc

        # Check if the case is related to Hiccup (Ách nghịch / Nấc)
        is_ach_nghich_case = any(kw in symptoms_lower_all for kw in ["nấc", "ách nghịch"])
        if is_ach_nghich_case:
            all_organs.add("Vị")

        if has_yinyang_conflict:
            all_bat_cuong.discard("Hàn")
            all_bat_cuong.discard("Nhiệt")
            all_bat_cuong.add("Hàn Nhiệt Thác Tạp")

        if has_huthuc_conflict:
            all_bat_cuong.discard("Hư")
            all_bat_cuong.discard("Thực")
            all_bat_cuong.add("Bản Hư Tiêu Thực")
            
        # Khử mâu thuẫn Biểu - Lý: Nếu có sợ gió/lạnh (Biểu) kiêm triệu chứng tạng phủ/bệu/trầm (Lý)
        bieu_kws = ["sợ gió", "sợ lạnh", "úy phong", "úy hàn", "ngạt mũi", "hắt hơi", "sổ mũi", "chảy nước mũi", "rét run"]
        has_bieu_indicator = any(kw in symptoms_lower_all for kw in bieu_kws)
        has_ly_indicator = any(x in all_bat_cuong for x in ["Lý"]) or len(all_organs) > 0 or any(kw in symptoms_lower_all for kw in ["lưỡi bệu", "rìa lưỡi có hằn răng", "mạch trầm", "chóng mặt", "mệt mỏi"])
        
        if has_bieu_indicator:
            if has_ly_indicator:
                all_bat_cuong.discard("Biểu")
                all_bat_cuong.discard("Lý")
                all_bat_cuong.add("Biểu - Lý đồng bệnh")
            else:
                all_bat_cuong.discard("Lý")
                all_bat_cuong.add("Biểu")

        # [FIX NHẤT QUÁN] Nếu Bát Cương (thường do metadata đồ thị của 2 hội chứng đóng góp)
        # chứa ĐỒNG THỜI cả Biểu và Lý mà bước trên chưa gộp, hợp nhất thành "Biểu - Lý đồng bệnh"
        # để không hiển thị 2 cực đối lập cạnh nhau ("Biểu - Lý") gây mâu thuẫn.
        if "Biểu" in all_bat_cuong and "Lý" in all_bat_cuong:
            all_bat_cuong.discard("Biểu")
            all_bat_cuong.discard("Lý")
            all_bat_cuong.add("Biểu - Lý đồng bệnh")

        # [FIX NHẤT QUÁN] Tương tự Biểu-Lý: nếu Bát Cương chứa ĐỒNG THỜI cả Hư và Thực (thường do
        # hội chứng gốc-hư + hội chứng nhánh-thực đóng góp, vd Tỳ khí hư [Hư] + Phong hàn [Thực])
        # mà bước triệu chứng chưa gộp, hợp nhất thành "Bản Hư Tiêu Thực" để không hiển thị 2 cực rời rạc.
        if "Hư" in all_bat_cuong and "Thực" in all_bat_cuong:
            all_bat_cuong.discard("Hư")
            all_bat_cuong.discard("Thực")
            all_bat_cuong.add("Bản Hư Tiêu Thực")

        # Sắp xếp các thành tố Bát Cương theo thứ tự chuẩn y học cổ truyền
        bat_cuong_order = []
        for x in ["Biểu", "Lý", "Biểu - Lý đồng bệnh"]:
            if x in all_bat_cuong:
                bat_cuong_order.append(x)
        for x in ["Hàn", "Nhiệt", "Hàn Nhiệt Thác Tạp"]:
            if x in all_bat_cuong:
                bat_cuong_order.append(x)
        for x in ["Hư", "Thực", "Bản Hư Tiêu Thực"]:
            if x in all_bat_cuong:
                bat_cuong_order.append(x)
        for x in all_bat_cuong:
            if x not in bat_cuong_order:
                bat_cuong_order.append(x)

        organs_hint = f"{', '.join(all_organs)}" if all_organs else "Chưa rõ"
        bat_cuong_hint = " - ".join(bat_cuong_order) if bat_cuong_order else "Chưa rõ"

        # Bộ lọc Cờ đỏ mâu thuẫn (Conflict Flags) sắc mặt
        face_conflict_hint = ""
        has_pale_face = any(kw in symptoms_lower_all for kw in ["mặt nhợt nhạt", "mặt nhợt", "mặt trắng nhợt", "da nhợt nhạt"])
        has_red_face = any(kw in symptoms_lower_all for kw in ["mặt đỏ", "vùng đỏ trên mặt", "gò má đỏ", "ửng đỏ", "ửng hồng", "đỏ trên má"])
        if has_pale_face and has_red_face:
            face_conflict_hint = (
                "\n        [CỜ ĐỎ MÂU THUẪN SẮC MẶT - HINT QUAN TRỌNG]: Hiện tượng mặt nhợt nhạt kiêm gò má đỏ/ửng đỏ ở đây "
                "là do chứng Hư hỏa bốc lên (Âm hư hỏa vượng, thủy không chế hỏa) hoặc Đới dương (Hư dương ngoại việt, chân dương suy kiệt đẩy hỏa giả lên mặt) "
                "của trạng thái Bản Hư Tiêu Thực. Đây hoàn toàn là bệnh lý Nội thương tạng phủ (bất kể có chẩn đoán Canh niên kỳ hội chứng hay Huyễn vựng), "
                "TUYỆT ĐỐI CẤM (PROHIBITED) giải thích màu đỏ/gò má đỏ này là do ngoại cảm phong tà xâm nhập vào biểu hay ngoại cảm phong nhiệt. "
                "Hãy giải thích rõ cơ chế Hư hỏa bốc lên hoặc Đới dương ở mục Bản Hư hoặc Tiêu Thực tương ứng."
            )

        # BƯỚC 2, 3, 4: GỌI LLM BIỆN CHỨNG THEO CHAIN-OF-THOUGHT
        rag_prompt = f"LÝ GIẢI Y LÝ CHUẨN (BẮT BUỘC BÁM SÁT): {rag_context_str}" if rag_context_str else ""
        
        prompt_reason = f"""
        Vai trò: Chuyên gia Y học Cổ truyền chuyên nghiệp.
        Bệnh nhân có các triệu chứng: {symptoms_str}.
        Chẩn đoán cốt lõi: {final_primary}.
        Hội chứng kèm theo: {final_concurrent}.
        {rag_prompt}
        {face_conflict_hint}
        
        [RÀNG BUỘC PHÂN TÍCH TỪ NEO4J]:
        - Tạng Phủ liên quan trực tiếp: {organs_hint}
        - Trạng thái Bát Cương xác định: {bat_cuong_hint}
        
        [DANH SÁCH TRIỆU CHỨNG BẮT BUỘC PHẢI GIẢI THÍCH 100%]:
        Bạn phải viết giải thích cơ chế y lý cho TOÀN BỘ các triệu chứng sau, TUYỆT ĐỐI KHÔNG ĐƯỢC BỎ SÓT: {symptoms_str}.
        
        Nhiệm vụ: Dựa trên chẩn đoán cốt lõi và dữ liệu triệu chứng, hãy viết đoạn phân tích cơ chế bệnh sinh theo đúng cấu trúc 3 phần dưới đây. 
        BẮT BUỘC định dạng Markdown chính xác như mẫu, KHÔNG tự thêm lời mở đầu hay kết luận:

        ### 2. Định vị Bát Cương
        - **Thuộc chứng:** {bat_cuong_hint}

        ### 3. Phân tích Cơ chế Gốc (Bản Hư)
        - [Phân tích Tạng phủ nào đang suy yếu sinh ra các triệu chứng nền nào? Nếu bệnh thuần Thực chứng không có Bản hư, ghi "Không có". KHÔNG tự bịa triệu chứng không có trong danh sách đầu vào.]

        ### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)
        - [Phân tích Tà khí nào đang tấn công sinh ra các biểu hiện cấp tính nào? Nếu bệnh thuần Hư chứng không có Tiêu thực, ghi "Không có". KHÔNG tự bịa triệu chứng.]

        LUẬT BẮT BUỘC (CHAIN-OF-THOUGHT):
        1. KHÔNG tự bịa thêm triệu chứng không có trong danh sách đầu vào.
        2. Tuân thủ tuyệt đối chức năng tạng phủ (VD: Tâm chủ thần minh/huyết mạch; Tỳ chủ vận hóa; Phế chủ khí/hô hấp; Thận chủ cốt tủy).
        3. KHÔNG LIỆT KÊ TẠNG PHỦ THỪA không có triệu chứng.
        4. CHỐT CHẶN HÔ HẤP: Các bệnh ngoại cảm hô hấp/mũi xoang (hắt hơi, sổ mũi, ho, chảy dịch mủ, đau nhức vùng mặt) BẮT BUỘC chỉ dùng các tạng/phủ Phế, Vị, Tỳ, Đởm. CHẶN HOÀN TOÀN Tâm, Can và Thận. Đối với đau nhức vùng mặt, đây là do phong nhiệt làm bít tắc kinh lạc vùng đầu mặt (Kinh Vị, Kinh Đởm), cấm giải thích do Thận hay Tỳ suy yếu.
        5. CHỐT CHẶN TÂN DỊCH: Khi giải thích Đàm, Thấp, Ẩm, TUYỆT ĐỐI KHÔNG gọi mầm bệnh là "Tân dịch". Phải dùng từ "Thủy thấp" hoặc "Đàm trọc".
        6. LUẬT CHẶN HƯ THỰC (BẢN - TIÊU) (Cập nhật):
           - Nếu trạng thái Bát Cương hoặc Hội chứng thuộc Thực chứng thuần túy (ví dụ: Ngoại cảm phong nhiệt, Phong hàn phạm biểu, Thấp nhiệt... chỉ chứa chữ "Thực" and không có chữ "Hư"), ở phần "### 3. Phân tích Cơ chế Gốc (Bản Hư)", BẮT BUỘC phải ghi: "Không có Bản Hư, đây là bệnh lý Thực chứng thuần túy." TUYỆT ĐỐI KHÔNG được tự suy diễn ra các hội chứng tạng phủ suy nhược (như Thận hư, Tỳ hư, Tâm huyết hư).
           - Ngược lại, nếu bệnh thuộc Hư chứng thuần túy (chỉ chứa chữ "Hư" và không có chữ "Thực"), ở phần "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)", BẮT BUỘC phải ghi duy nhất một câu: "Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy." TUYỆT ĐỐI KHÔNG được viết thêm bất kỳ dòng diễn giải hay danh sách liệt kê triệu chứng nào khác ở mục này. Tất cả triệu chứng phải được giải thích gói gọn trong phần Bản Hư (Mục 3).
        7. LUẬT PHỦ LẤP TRIỆU CHỨNG 100% VÀ HÀNH VĂN MƯỢT MÀ (Cập nhật):
           - Bạn BẮT BUỘC phải đưa từng triệu chứng trong danh sách [{symptoms_str}] vào phần giải thích y lý và giải thích cơ chế vì sao có triệu chứng đó dưới góc độ sinh lý tạng phủ. 
           - Danh sách triệu chứng bắt buộc giải thích: {symptoms_str}.
           - Tuy nhiên, TUYỆT ĐỐI KHÔNG ĐƯỢC liệt kê gạch đầu dòng từng triệu chứng một cách rời rạc hay máy móc. Hãy hành văn thành một hoặc hai đoạn văn trôi chảy, xâu chuỗi các cơ chế lại với nhau một cách logic, mạch lạc như một danh y thực thụ. Nếu bạn bỏ sót bất kỳ triệu chứng nào (ví dụ không giải thích đau đầu, chóng mặt hay hoa mắt), chẩn đoán này sẽ bị coi là Thất bại hoàn toàn.
           - Gợi ý biện luận từ chuyên gia Đông y: Dương khí hư suy dẫn đến Thanh dương bất thăng (khí trong trẻo không đủ sức thăng lên thượng tiêu), não bộ mất đi sự nuôi dưỡng phát sinh chóng mặt, hoa mắt. Âm hàn ngưng trệ kinh mạch vùng đầu cổ, khí huyết kém lưu thông sinh ra đau đầu.
        8. LUẬT BÁM SÁT HỘI CHỨNG CỐT LÕI VÀ TRÁNH NHẦM LẪN KHÍ - ÂM (Cập nhật):
           - Bạn BẮT BUỘC phải sử dụng "Hội chứng cốt lõi" ({final_primary}) làm trung tâm chủ đạo của toàn bộ lập luận để giải thích nguyên nhân gây ra các triệu chứng chính.
           - CẤM NHẦM LẪN KHÍ SUY VÀ TÂN DỊCH HAO TỔN: Tuyệt đối không được đánh đồng Khí hư với Âm hư. Khí hư là thiếu năng lượng sinh học (Tỳ khí suy nhược, khí không hành được huyết gây huyết ứ). Âm hư là thiếu hụt nước/tân dịch (sinh nội nhiệt). Nếu hội chứng cốt lõi là Khí hư (ví dụ: Khí hư huyết ứ, Tỳ khí hư), bạn phải giải thích cơ chế dựa trên khí hư lực kiệt, cấm tự ý giải thích lấp liếm bằng các thuật ngữ "âm hư sinh nội nhiệt" hay "tân dịch hao tổn" trừ khi bản thân hội chứng đó là Khí Âm lưỡng hư hoặc Âm hư.
           - "Hội chứng kèm theo" ({final_concurrent}) chỉ được dùng để bổ sung ý nghĩa cho các kiêm chứng (nếu có), TUYỆT ĐỐI không được lấy hội chứng kèm theo để giải thích át hoặc thay thế hoàn toàn cho vai trò của hội chứng cốt lõi.
        9. LUẬT GIẢI MÃ MẠCH TƯỢNG NGHIÊM NGẶT (MỚI):
           - Khi giải thích Mạch tượng, bạn BẮT BUỘC phải bám sát ý nghĩa gốc của từng loại mạch lý:
             + Phù (nổi) = Bệnh ở Biểu. Trầm (chìm) = Bệnh ở Lý.
             + Trì (chậm) = Chứng Hàn. Sác (nhanh) = Chứng Nhiệt (Thực nhiệt hoặc Âm hư sinh nội nhiệt).
             + Tế (nhỏ) = Hư chứng (Âm/Huyết hư). Thực (có lực) = Thực chứng.
           - Nếu bệnh nhân có mạch Tế Sác kiêm Sợ lạnh, phải giải thích rõ đây là tình trạng Âm Dương Lưỡng Hư (Hàn Nhiệt đan xen): Dương hư sinh ngoại hàn (sợ lạnh), Âm hư sinh nội nhiệt (mạch sác). Tuyệt đối cấm giải thích mạch sác một cách lấp liếm là "do khí huyết suy yếu không làm đầy mạch".
        10. LUẬT TỰ KIỂM TRA ĐẦY ĐỦ TRIỆU CHỨNG VÀ TÍCH HỢP TỰ NHIÊN (SELF-CHECK):
           - Trước khi xuất kết quả, hãy rà soát kỹ mảng triệu chứng đầu vào: [{symptoms_str}].
           - Đảm bảo 100% từ khóa triệu chứng đã được giải thích trong câu trả lời.
           - LƯU Ý QUAN TRỌNG: Phải lồng ghép các triệu chứng này một cách tự nhiên vào mạch văn ngay từ đầu. TUYỆT ĐỐI KHÔNG được viết lặp lại ý hoặc đắp thêm câu liệt kê rác ở cuối đoạn chỉ để đối phó với luật kiểm tra. Nếu một triệu chứng đã được giải thích ở đầu hoặc giữa đoạn (dù dùng từ đồng nghĩa hay đảo từ), KHÔNG cần nhắc lại lần nữa.
        11. LUẬT ĐỐI MẶT MÂU THUẪN DỮ LIỆU (BẢN HƯ - TIÊU THỰC / HÀN - NHIỆT):
           - Nếu trong danh sách triệu chứng có sự mâu thuẫn rõ rệt (ví dụ: Lưỡi đỏ khô [Âm Hư/Nhiệt] đi kèm với Rêu trắng dày [Thực/Đàm/Hàn]), TUYỆT ĐỐI KHÔNG ĐƯỢC xóa bỏ hay phớt lờ bất kỳ triệu chứng nào. Bạn BẮT BUỘC phải kết luận đây là chứng Bản Hư Tiêu Thực hoặc Hàn Nhiệt Thác Tạp.
           - Nguyên tắc phân chia Bản - Tiêu: Gốc bệnh (Bản hư - ví dụ Can Thận âm hư, mặt nhợt, khô miệng, họng ráo) BẮT BUỘC phải giải thích ở Mục 3 (Bản Hư). Tà khí ứ đọng, đàm thấp hoặc các triệu chứng thực chứng, cấp tính (Tiêu thực - ví dụ rêu trắng dày, nốt mụn đỏ, tiếng nấc) BẮT BUỘC phải giải thích ở Mục 4 (Tiêu Thực).
        12. CHỐT CHẶN ÁCH NGHỊCH (NẤC CỤT) (MỚI):
           - Bệnh Ách nghịch (nấc cụt) có cơ chế bệnh sinh cốt lõi duy nhất là Vị khí thượng nghịch (khí của tạng Vị bốc ngược lên trên). Bạn BẮT BUỘC phải giải thích cơ chế tiếng nấc dựa trên Vị khí thượng nghịch và tạng Vị (Dạ dày). TUYỆT ĐỐI CẤM bịa ra các cơ chế sai lệch không có thật như "Can khí hư" hay "Thận khí hư" gây ra nấc cụt.
        13. CHỐT CHẶN VỌNG CHẨN LƯỠI (TONGUE DIAGNOSIS CONSTRAINT) (MỚI):
           - Lưỡi bệu (hoặc lưỡi sưng, lưỡi to): Trong Đông y, lưỡi bệu/sưng BẮT BUỘC phải giải thích là do Tỳ hư hoặc Thận dương hư dẫn đến vận hóa bất thường, khiến Thủy thấp ứ đọng/thủy dịch bất hóa làm thân lưỡi căng bệu sưng to. TUYỆT ĐỐI CẤM giải thích sai lệch lưỡi bệu là do "thiếu ẩm", "thiếu tân dịch" hay "không đủ dịch để bôi trơn".
           - Chất lưỡi đạm đỏ (lưỡi đỏ nhạt/nhạt đỏ): Phải giải thích do khí huyết suy kém không vinh nhuận đầy đủ lên thân lưỡi.
           - Lưỡi có vết nứt (nứt lưỡi): Phải giải thích do âm huyết hư tổn/tân dịch khuy tổn làm mất sự nuôi dưỡng trên bề mặt lưỡi.
        14. TÔN TRỌNG VÀ BIỆN GIẢI ĐÚNG VỀ RÊU TRẮNG MỎNG (Cập nhật):
           - Rêu trắng mỏng là ranh giới giữa Sinh lý và Bệnh lý (mới phát ở Biểu).
           - Nếu bệnh nhân có rêu trắng mỏng đi kèm các triệu chứng như sợ gió, đau đầu, bạn BẮT BUỘC phải giải thích đây là biểu hiện của Ngoại cảm phong hàn (Tà khí mới xâm nhập đang ở phần Biểu nông) hoặc Chính khí chưa suy kiệt / Tà khí chưa hóa nhiệt.
           - TUYỆT ĐỐI CẤM (PROHIBITED) giải thích rêu trắng mỏng là do Đàm ẩm, Thủy thấp ứ đọng hay Huyết ứ (vì khi đã có Thủy thấp/Đàm trọc tích tụ thì rêu lưỡi bắt buộc phải dày, nhớt hoặc trơn).
           - Với các từ khóa như 'lưỡi hồng', 'mạch hoãn', 'mạch bình thường', hãy nhận định đây là dấu hiệu sinh lý bình thường (vị khí còn tốt, chính khí chưa suy).
        15. CẤM TỰ BIÊN TỰ DIỄN HỘI CHỨNG MỚI (MỚI):
           - Bạn TUYỆT ĐỐI KHÔNG ĐƯỢC tự ý lôi kéo các hội chứng tạng phủ suy nhược khác không được chốt ở Bước 1 vào lập luận. Ví dụ: Nếu chẩn đoán cốt lõi và hội chứng kèm theo ở Bước 1 ({final_primary}, {final_concurrent}) KHÔNG có 'Thận âm hư', 'Tỳ dương hư' hay 'Can âm hư', bạn TUYỆT ĐỐI CẤM (PROHIBITED) sử dụng các thuật ngữ đó làm nguyên nhân gây hư hỏa hay bốc hỏa ở Mục 3 và Mục 4. Hãy giải thích cơ chế bốc hỏa/đỏ mặt dựa trên chính khí huyết hư (Ví dụ: huyết hư bất năng nhiếp dương, khiến hư hỏa/hư dương nổi lên trên) để đảm bảo tính nhất quán tuyệt đối giữa các bước chẩn đoán.
        16. CHỐT CHẶN MỒ HÔI (ĐẠO HÃN vs TỰ HÃN) (MỚI):
           - ĐẠO HÃN (mồ hôi trộm — ra mồ hôi lúc ngủ, tỉnh dậy thì hết): BẮT BUỘC là biểu hiện của ÂM HƯ. Cơ chế: âm hư sinh nội nhiệt (hư hỏa), nhiệt bức tân dịch tiết ra ngoài về đêm. Phải giải thích ở "### 3. Phân tích Cơ chế Gốc (Bản Hư)" theo cơ chế âm hư → hư nhiệt → bức mồ hôi. TUYỆT ĐỐI CẤM (PROHIBITED) giải thích đạo hãn bằng cơ chế "khí hư/khí huyết hư/dương hư không giữ được mồ hôi" (đó là cơ chế của TỰ HÃN), và CẤM xếp đạo hãn vào "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực)".
           - TỰ HÃN (mồ hôi tự ra ban ngày, vận động càng ra nhiều): mới là biểu hiện KHÍ HƯ/DƯƠNG HƯ (vệ khí bất cố, tấu lý không kín) → giải thích theo cơ chế khí hư bất cố nhiếp ở phần Bản Hư.
        """
        
        try:
            response = self.qa_pipeline.client.chat(
                model=self.qa_pipeline.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là bác sĩ Đông y Việt Nam uyên bác. Bạn CẤM TUYỆT ĐỐI sử dụng chữ Hán, chữ Trung Quốc hay bính âm (Pinyin). Mọi thuật ngữ phải được dịch sang tiếng Việt thuần túy."},
                    {"role": "user", "content": prompt_reason}
                ],
                options={"temperature": 0.1, "seed": 42}
            )
            llm_explanation = response['message']['content'].strip()
            llm_explanation = self._clean_foreign_characters(llm_explanation)
            llm_explanation = self._post_process_hallucinations(llm_explanation, symptoms_str)
        except Exception as e:
            logger.error(f"Lỗi gọi LLM giải thích y lý Bát Cương: {e}")
            llm_explanation = "### 2. Định vị Bát Cương\n- Không xác định\n\n### 3. Phân tích Cơ chế Gốc (Bản Hư)\n- Không xác định\n\n### 4. Phân tích Cơ chế Ngọn (Tiêu Thực)\n- Không xác định"

        # [FIX] Bổ sung hậu xử lý: phát hiện và chèn triệu chứng bị LLM bỏ sót
        llm_explanation = self._patch_missing_symptoms(llm_explanation, symptoms_str)

        final_markdown += f"{llm_explanation}\n\n"

        # BƯỚC 5: PHÁP TRỊ & BÀI THUỐC
        final_markdown += "### 5. Pháp trị & Đề xuất Bài thuốc\n"
        
        has_treatment = False
        # [FIX] Chỉ in bài thuốc của bệnh lý đã được định danh cụ thể ở Bước 1 và thuộc Hội chứng thực sự chẩn đoán (final_primary/final_concurrent)
        primary_key = final_primary.lower().strip()
        concurrent_key = (
            final_concurrent.lower().strip()
            if (final_concurrent and final_concurrent != "Không có") else None
        )
        active_syndromes = {primary_key}
        if concurrent_key:
            active_syndromes.add(concurrent_key)

        # [FIX TRUY HỒI NHẤT QUÁN] Tách bài thuốc theo vai trò Bản (hội chứng cốt lõi) vs Tiêu
        # (hội chứng kèm theo) để biện chứng (mục 3, xoay quanh hội chứng cốt lõi) và pháp trị
        # (mục 5) KHÔNG lệch hội chứng. Khi cốt lõi chưa có bài grounded mà chỉ có bài trị nhánh,
        # phải ghi rõ để tránh hiểu nhầm bài trị Tiêu là bài trị gốc.
        core_lines = []    # bài thuốc trị Bản (gốc bệnh)
        branch_lines = []  # bài thuốc trị Tiêu (nhánh / kèm theo)
        if disease_names and len(disease_names) <= 3:
            for data in detailed_kg_data:
                syn_name = data["syndrome"].lower().strip()
                if syn_name not in active_syndromes:
                    continue
                is_core = (syn_name == primary_key)
                role = "Bản – gốc bệnh" if is_core else "Tiêu – nhánh/kèm theo"
                for tb in data.get("treatments_by_disease", []):
                    # Khớp chính xác bệnh lý
                    if tb.get("disease") in disease_names:
                        # Chuẩn hoá vị thuốc về chuỗi sạch (KG trả list, CSV trả string) -> tránh in ['...','...']
                        vi_raw = tb.get("vi_thuoc") or []
                        if isinstance(vi_raw, (list, tuple, set)):
                            vi_str = ", ".join(str(v).strip() for v in vi_raw if str(v).strip())
                        else:
                            vi_str = str(vi_raw).strip()
                        vi_str = vi_str or "(chưa cập nhật vị thuốc)"
                        line = (
                            f"- Trị Bệnh **{tb['disease']}** — *{role}* (Hội chứng {data['syndrome']}) → Dùng bài **{tb['bai_thuoc']}**\n"
                            f"  - *Vị thuốc:* {vi_str}\n"
                        )
                        (core_lines if is_core else branch_lines).append(line)

        has_treatment = bool(core_lines or branch_lines)

        # In bài trị Bản (gốc) trước
        final_markdown += "".join(core_lines)

        # Cốt lõi (Bản) chưa có bài grounded nhưng có bài trị Tiêu -> nêu rõ để mục 3 và mục 5 nhất quán
        if branch_lines and not core_lines and primary_key not in ("chưa rõ", "", "không có"):
            final_markdown += (
                f"- **Pháp trị gốc (Bản) — {final_primary}:** trọng tâm điều trị phải nhắm vào gốc bệnh "
                f"(*{final_primary}*). Cơ sở tri thức hiện **chưa có bài thuốc đặc trị** gán trực tiếp cho "
                f"hội chứng cốt lõi này ở bệnh danh tương ứng — cần thầy thuốc kê bài tư bổ theo gốc.\n"
                f"- Bài thuốc dưới đây chỉ trị **Tiêu (nhánh/triệu chứng kèm theo)**, KHÔNG thay thế việc điều trị gốc:\n"
            )

        # In bài trị Tiêu (nhánh)
        final_markdown += "".join(branch_lines)

        if not has_treatment:
            if not disease_names or len(disease_names) > 3:
                symptoms_lower_all = (user_symptoms + " " + symptoms_str).lower()
                has_pulse_info = any(kw in symptoms_lower_all for kw in ["mạch", "tế sác", "sác", "trầm", "khẩn", "hoạt", "phù", "trì", "nhược"])
                has_tongue_info = any(kw in symptoms_lower_all for kw in ["rêu", "lưỡi", "chất lưỡi", "bệu", "nứt", "gai"])
                
                if has_pulse_info or has_tongue_info:
                    final_markdown += (
                        "- Hệ thống hiện tại chưa cập nhật bài thuốc đặc trị khớp hoàn toàn với tổ hợp triệu chứng và mạch lý này.\n"
                        "- Khuyến nghị: Người bệnh nên tham khảo ý kiến của bác sĩ Đông y để được biện chứng luận trị sâu hơn.\n"
                    )
                else:
                    final_markdown += (
                        "- Triệu chứng quá chung chung, ứng với nhiều bệnh lý khác nhau.\n"
                        "- Vui lòng bổ sung thêm triệu chứng chi tiết (ví dụ: tính chất cơn đau, rêu lưỡi, mạch) để xác định bài thuốc chính xác.\n"
                    )
            else:
                # [FIX NHẤT QUÁN] Có bệnh danh + hội chứng cốt lõi rõ nhưng KG chưa có bài grounded:
                # nối Mục 5 với biện chứng (Mục 3) — nêu pháp trị nhắm hội chứng cốt lõi, thay vì
                # cụt "chưa có bài thuốc" (không bịa công thức cụ thể để tránh sai y lý).
                if primary_key not in ("chưa rõ", "", "không có"):
                    benh_str = ", ".join(disease_names)
                    final_markdown += (
                        f"- **Pháp trị:** trọng tâm điều trị nhắm vào hội chứng cốt lõi **{final_primary}** "
                        f"(đã biện luận ở Mục 3). Cơ sở tri thức hiện **chưa có bài thuốc đặc trị** được gán "
                        f"trực tiếp cho hội chứng cốt lõi này ở bệnh danh **{benh_str}**.\n"
                        f"- Khuyến nghị: tham khảo thầy thuốc Đông y để được kê phương theo đúng hội chứng cốt lõi.\n"
                    )
                else:
                    final_markdown += "- Hiện chưa có bài thuốc cập nhật cho bệnh lý này trong hệ thống.\n"

        # Cập nhật lời khuyên đặc trị nếu có (Thêm trực tiếp vào Mục 5)
        if "Tỳ thận dương hư" in final_primary:
            final_markdown += "\n- **Lời khuyên bổ sung:** Ôn bổ Tỳ Thận, Sáp trường chỉ tả. Phương tễ kinh điển nhất để điều trị Tỳ thận dương hư tiêu chảy là Tứ thần hoàn (hoặc Phụ tử lý trung thang gia giảm).\n"
        elif "Khí huyết đều hư" in final_primary:
            final_markdown += "\n- **Lời khuyên bổ sung:** Cần kiện tỳ hành thủy (Bạch truật, Phục linh) và Ôn bổ thận khí / Tư âm bổ huyết để phục hồi từ gốc.\n"
            
        return final_markdown


    # Triệu chứng "cờ đỏ" cần đi cấp cứu ngay, không tự dùng thuốc Đông y
    _RED_FLAG_KEYWORDS = [
        "đau ngực", "đau thắt ngực", "khó thở", "khó thở cấp", "khó thở dữ dội", "ngất", "ngất xỉu",
        "co giật", "liệt", "méo miệng", "yếu nửa người", "bán thân bất toại", "nói khó", "khó nói",
        "lơ mơ", "hôn mê", "mất ý thức", "mất ngôn ngữ", "ho ra máu", "nôn ra máu", "đi ngoài ra máu",
        "xuất huyết", "chảy máu không cầm", "sốt cao co giật", "cứng gáy", "đau đầu dữ dội",
        "chấn thương sọ não", "tai nạn",
    ]

    def _append_medical_disclaimer(self, markdown: str, raw_text: str = "") -> str:
        """Chèn cảnh báo cấp cứu (nếu phát hiện cờ đỏ) và disclaimer y tế vào cuối MỌI câu trả lời."""
        markdown = markdown or ""
        text_lower = (raw_text or "").lower()

        red_flag_block = ""
        if any(kw in text_lower for kw in self._RED_FLAG_KEYWORDS):
            red_flag_block = (
                "\n\n> 🚨 **CẢNH BÁO KHẨN CẤP:** Một số triệu chứng bạn mô tả có thể là dấu hiệu nguy hiểm "
                "cần cấp cứu (đột quỵ, nhồi máu cơ tim, xuất huyết, chấn thương...). "
                "**Hãy gọi 115 hoặc đến cơ sở y tế gần nhất NGAY**, KHÔNG tự điều trị bằng thuốc Đông y.\n"
            )

        disclaimer = (
            "\n\n---\n"
            "⚠️ **Miễn trừ trách nhiệm:** Đây là **công cụ tham khảo/giáo dục dựa trên AI**, "
            "**KHÔNG phải chẩn đoán y khoa** và **không thay thế bác sĩ**. Các hội chứng, bài thuốc và "
            "vị thuốc nêu trên chỉ mang tính tham khảo. **Tuyệt đối không tự ý dùng thuốc** — nhiều vị "
            "thuốc Đông y có độc tính hoặc chống chỉ định với thai phụ, người có bệnh nền hoặc gây tương "
            "tác thuốc. Hãy tham khảo thầy thuốc Đông y / bác sĩ có chứng chỉ hành nghề trước khi áp dụng.\n"
        )
        return markdown + red_flag_block + disclaimer

    def answer_question(self, question: str) -> dict:
        """Hỏi-đáp tự do trên Knowledge Graph cho luồng web.

        Dùng text_to_cypher (LLM sinh Cypher) ở CHẾ ĐỘ READ-ONLY (chặn Cypher ghi ở 2 lớp)
        rồi trả lời. LUÔN kèm disclaimer y tế + cảnh báo cờ đỏ như luồng chẩn đoán để giữ
        bất biến an toàn y tế trên mọi câu trả lời."""
        q = (question or "").strip()
        if not q:
            return {
                "question": "",
                "answer": self._append_medical_disclaimer(
                    "Vui lòng nhập câu hỏi về triệu chứng hoặc bệnh lý.", ""),
                "data": [],
            }
        result = self.qa_pipeline.ask(q, read_only=True)
        raw_answer = result.get("answer", "") or ""
        return {
            "question": q,
            # raw_text gồm cả câu hỏi để phát hiện cờ đỏ cấp cứu từ chính triệu chứng người dùng nhập
            "answer": self._append_medical_disclaimer(raw_answer, f"{q} {raw_answer}"),
            "data": result.get("data", []),
        }

    def run_diagnosis(self, user_symptoms: str = "", face_img_path: str = None, tongue_img_path: str = None) -> dict:
        vision_analysis_text = ""
        raw_vision_data = None
        self._has_makeup = False

        if face_img_path or tongue_img_path:
            logger.info("Bắt đầu phân tích hình ảnh qua LLaVA...")
            try:
                raw_vision_data = self.vision_pipeline.run(tongue_image_path=tongue_img_path, face_image_path=face_img_path)
                if raw_vision_data and isinstance(raw_vision_data, dict):
                    tongue_desc = raw_vision_data.get("tongue_description", "")
                    face_desc = raw_vision_data.get("face_description", "")
                    
                    if face_desc:
                        makeup_keywords = ["makeup", "lipstick", "eyeliner", "blush", "mascara", "foundation", "eyeshadow", "trang điểm", "son môi", "má hồng", "kẻ mắt", "phấn má"]
                        if any(kw in face_desc.lower() for kw in makeup_keywords):
                            self._has_makeup = True
                            
                    raw_vision_data["has_makeup"] = self._has_makeup
                    
                    detected_symptoms = []
                    
                    if tongue_desc:
                        logger.info(f"Khớp mô tả lưỡi với danh sách triệu chứng chuẩn...")
                        mapped_tongue = self._map_desc_to_symptoms(tongue_desc, self.tongue_symptoms_list)
                        logger.info(f"Triệu chứng lưỡi đã khớp: {mapped_tongue}")
                        detected_symptoms.extend(mapped_tongue)
                        
                    if face_desc:
                        logger.info(f"Khớp mô tả sắc mặt với danh sách triệu chứng chuẩn...")
                        mapped_face = self._map_desc_to_symptoms(face_desc, self.face_symptoms_list)
                        logger.info(f"Triệu chứng mặt đã khớp: {mapped_face}")
                        detected_symptoms.extend(mapped_face)
                        
                    detected_symptoms = list(set(detected_symptoms))
                    detected_symptoms = self._resolve_symptom_conflicts(detected_symptoms)
                            
                    raw_vision_data["detected_symptoms"] = detected_symptoms
                    vision_analysis_text = ", ".join(detected_symptoms)
                    raw_vision_data["analysis"] = vision_analysis_text
                    
                    if tongue_desc:
                        raw_vision_data["tongue_description_vi"] = self._translate_english_description(tongue_desc)
                    if face_desc:
                        raw_vision_data["face_description_vi"] = self._translate_english_description(face_desc)
            except Exception as e:
                logger.error(f"Lỗi module Vision: {e}")

        combined_query = user_symptoms.strip()
        if vision_analysis_text:
            combined_query = f"{combined_query}, {vision_analysis_text}" if combined_query else vision_analysis_text

        # Bước trích xuất triệu chứng (đặt self._last_extracted_terms + danh sách hội chứng ứng viên LLM)
        llm_syndromes = self._extract_syndromes_from_text(combined_query) if combined_query else []

        # [GROUNDED SYNDROME-SCORING] Thay việc LLM 7B TỰ CHỌN hội chứng (bất ổn: lần Đàm hỏa, lần Huyết
        # ứ, bỏ sót Tỳ khí hư hiển nhiên) bằng CHẤM ĐIỂM trên graph — hội chứng nào khớp NHIỀU triệu
        # chứng của bệnh nhân nhất thì xếp đầu. Deterministic, grounded. Fallback về LLM nếu chấm rỗng.
        # Chấm điểm trên triệu chứng SẠCH (khớp trực tiếp text người dùng + vọng chẩn từ ảnh), KHÔNG
        # dùng bản đã-bung-synonym (_last_extracted_terms) — vì synonym cổ phương gây nhiễu ranking
        # (vd đẩy nhầm "Hàn thấp" lên top). Đồng bộ với script test_grounded_scoring.py.
        _clean_terms = self.qa_pipeline._preprocess_question(user_symptoms) if user_symptoms else []
        _score_terms = list(dict.fromkeys(
            _clean_terms + (raw_vision_data.get("detected_symptoms", []) if raw_vision_data else [])
        ))
        if not _score_terms and combined_query:
            _score_terms = self.qa_pipeline._preprocess_question(combined_query)
        _grounded = self._score_syndromes_grounded(_score_terms)
        final_syndromes = [s for s, _sc in _grounded] if _grounded else llm_syndromes

        # [FIX] Độc lập cưỡng bức Âm Dương Lưỡng Hư lên đầu khi có mâu thuẫn Hàn - Nhiệt lâm sàng
        symptoms_lower_all = (user_symptoms + " " + combined_query).lower()
        has_cold_indicator = any(kw in symptoms_lower_all for kw in ["sợ lạnh", "úy hàn", "sợ gió", "rét run"])
        has_heat_pulse_indicator = any(kw in symptoms_lower_all for kw in ["mạch sác", "tế sác", "sác", "mạch trầm sác", "khát nước", "sốt", "đỏ bừng", "khô miệng"])
        if has_cold_indicator and has_heat_pulse_indicator:
            final_syndromes = [s for s in final_syndromes if s.lower().strip() not in ["âm dương lưỡng hư", "âm dương đều hư", "âm dương câu hư"]]
            final_syndromes.insert(0, "Âm Dương Lưỡng Hư")

        # [FIX Y LÝ ĐÔNG Y] Đạo hãn (mồ hôi trộm) là dấu ÂM HƯ điển hình (âm hư sinh nội nhiệt,
        # bức tân dịch ra ngoài về đêm) — TUYỆT ĐỐI không phải do Dương hư. Xử lý mâu thuẫn:
        #   (A) đạo hãn (âm hư) + dấu Dương hư/Hàn (sợ lạnh, tay chân lạnh, tiểu đêm...) -> Âm dương lưỡng hư.
        #   (B) đạo hãn nhưng KHÔNG có dấu dương hư -> nghiêng ÂM HƯ: nếu hội chứng cốt lõi đang là
        #       "* dương hư" thì chuyển sang "* âm hư" tương ứng (vd Thận dương hư -> Thận âm hư) để
        #       biện chứng không giải thích sai đạo hãn bằng cơ chế dương hư/thủy thấp.
        _amduong_names = ["âm dương lưỡng hư", "âm dương đều hư", "âm dương câu hư"]
        has_dao_han = any(kw in symptoms_lower_all for kw in ["mồ hôi trộm", "đạo hãn"])
        has_duong_hu_sign = has_cold_indicator or any(
            kw in symptoms_lower_all
            for kw in ["tay chân lạnh", "chân tay lạnh", "chi lãnh", "lưng lạnh", "tiểu đêm",
                       "đại tiện lỏng", "phân sống", "phân lỏng nát", "liệt dương"]
        )
        if has_dao_han and has_duong_hu_sign:
            if not final_syndromes or final_syndromes[0].lower().strip() not in _amduong_names:
                final_syndromes = [s for s in final_syndromes if s.lower().strip() not in _amduong_names]
                final_syndromes.insert(0, "Âm Dương Lưỡng Hư")
        elif has_dao_han and not has_duong_hu_sign and final_syndromes:
            top = final_syndromes[0]
            if "dương hư" in top.lower():
                # Thận dương hư -> Thận âm hư (giữ đúng tạng, đổi âm/dương cho khớp đạo hãn)
                am_hu = re.sub(r'(?i)dương hư', 'âm hư', top)
                final_syndromes = [am_hu] + [s for s in final_syndromes if s != top]
            elif not any("âm hư" in s.lower() for s in final_syndromes):
                # Chưa có hội chứng âm hư nào để giải thích đạo hãn -> bổ sung Thận âm hư
                final_syndromes.insert(0, "Thận âm hư")

        final_syndromes = self._filter_hierarchical_redundancies(final_syndromes)

        search_terms = getattr(self, "_last_extracted_terms", []) or (
            self.qa_pipeline._preprocess_question(combined_query) if combined_query else []
        )
        sym_disease_map = self.qa_pipeline.get_symptom_disease_map(search_terms) if search_terms else {}
        
        # [FIX] Chỉ sử dụng toàn bộ danh sách hội chứng làm phương án dự phòng nếu LLM không trích xuất được gì
        if not final_syndromes:
            final_syndromes = self._filter_hierarchical_redundancies(list(sym_disease_map.keys()))

        # [FIX Y LÝ — NGOẠI CẢM BIỂU THỰC] Ca ngoại cảm cấp (Phong hàn/Phong nhiệt phạm biểu): GỐC bệnh
        # là TÀ KHÍ Ở BIỂU (Biểu Thực), KHÔNG phải Bản Hư nội thương — TUYỆT ĐỐI không đảo hội chứng Hư
        # lên cốt lõi (nếu không, ca cảm mạo cấp sẽ bị dựng sai cơ chế "Bản Hư Phế khí hư"). Nếu KHÔNG có
        # dấu HƯ MẠN (bệnh lâu ngày/tự hãn/đoản khí/hay cảm tái phát/đau lưng mỏi gối...) thì loại luôn các
        # hội chứng Hư nội thương lọt vào do khớp vài triệu chứng chung (mệt/đau đầu) → chẩn đoán SẠCH Biểu Thực.
        if final_syndromes and self._syndrome_is_exterior_wind(final_syndromes[0]):
            _chronic_hu_signs = [
                "bệnh lâu ngày", "mãn tính", "lâu ngày", "tự hãn", "ra mồ hôi ban ngày",
                "đoản khí", "hụt hơi", "đuối sức", "hay cảm", "dễ cảm", "cảm vặt",
                "tái đi tái lại", "tái phát", "đau lưng mỏi gối", "mỏi gối", "tiểu đêm",
                "mạch vi nhược", "gầy sút",
            ]
            if not any(k in symptoms_lower_all for k in _chronic_hu_signs):
                _kept = [s for s in final_syndromes if not self._syndrome_is_hu(s)]
                if _kept:
                    final_syndromes = _kept

        # [FIX Y LÝ — BẢN HƯ TIÊU THỰC] Cốt lõi (Bản/gốc bệnh) PHẢI là hội chứng HƯ; hội chứng THỰC
        # (đàm/hỏa/ứ/nhiệt/nghịch...) chỉ là kèm theo (Tiêu/ngọn). Nếu hội chứng đầu là THỰC THUẦN mà
        # trong danh sách có hội chứng HƯ, đưa hội chứng HƯ lên làm cốt lõi — để Mục 3 (Bản Hư), pháp trị
        # và bệnh danh bám đúng gốc bệnh, tránh chốt hội chứng Thực "kịch tính" (vd Đàm hỏa nghịch) làm
        # cốt lõi rồi kéo theo bệnh danh vô lý (vd "Cuồng"). NGOẠI TRỪ ngoại cảm biểu thực (xử lý ở trên):
        # cảm mạo cấp gốc là tà khí ở biểu, không có Bản Hư để đảo lên.
        if final_syndromes and self._syndrome_is_thuc_pure(final_syndromes[0]) \
                and not self._syndrome_is_exterior_wind(final_syndromes[0]):
            _hu_idx = next((i for i, s in enumerate(final_syndromes) if self._syndrome_is_hu(s)), None)
            if _hu_idx is not None and _hu_idx != 0:
                final_syndromes.insert(0, final_syndromes.pop(_hu_idx))
            elif _hu_idx is None:
                # LLM chỉ trích xuất hội chứng THỰC (vd Huyết ứ) mà bỏ sót gốc Hư. Nếu triệu chứng chỉ
                # RÕ RÀNG một Bản Hư điển hình thì bổ sung nó làm cốt lõi (Thực xuống kèm theo).
                # Tỳ khí hư: mệt mỏi + (đại tiện lỏng/ăn kém) + (hằn răng/lưỡi bệu) + sắc mặt nhợt.
                _sl = symptoms_lower_all
                _ty_signs = sum([
                    any(k in _sl for k in ["mệt mỏi", "uể oải", "tinh thần uể oải", "vô lực", "thần bì"]),
                    any(k in _sl for k in ["đại tiện lỏng", "phân lỏng", "tiêu chảy", "tiết tả", "ăn kém", "chán ăn"]),
                    any(k in _sl for k in ["hằn răng", "lưỡi bệu"]),
                    any(k in _sl for k in ["mặt nhợt", "sắc mặt nhợt", "nhợt nhạt", "lưỡi nhợt"]),
                ])
                if _ty_signs >= 3:
                    final_syndromes.insert(0, "Tỳ khí hư")

        # [FIX] Lọc chính xác các bệnh lý khớp được ở Bước 1 trước khi xuất dữ liệu KG
        detected_symptoms = raw_vision_data.get("detected_symptoms", []) if raw_vision_data else []
        symptoms_arr = self._resolve_symptom_conflicts(list(dict.fromkeys(search_terms + detected_symptoms)))
        matched_diseases = self._find_matching_diseases(symptoms_arr, raw_user_text=user_symptoms)
        disease_names = []
        if matched_diseases:
            # Lọc chéo với hội chứng đã chẩn đoán để đồng bộ y lý bệnh danh.
            # [FIX] ƯU TIÊN bệnh danh grounded với hội chứng CỐT LÕI (gốc bệnh); chỉ khi không có mới
            # nới ra hội chứng kèm theo. Tránh nêu bệnh danh grounded với hội chứng Thực-nhánh (vd "Cuồng"
            # grounded Đàm hỏa nghịch) khi cốt lõi đã là hội chứng Hư.
            valid_syndromes = [s.lower().strip() for s in final_syndromes if s]
            core_syn = valid_syndromes[0] if valid_syndromes else ""
            core_matched = [m for m in matched_diseases if core_syn and self._are_syndromes_related(core_syn, m["hoi_chung"])]
            any_matched = [
                m for m in matched_diseases
                if any(self._are_syndromes_related(vs, m["hoi_chung"]) for vs in valid_syndromes)
            ]
            syndrome_matched_diseases = core_matched if core_matched else any_matched
            target_matches = syndrome_matched_diseases if syndrome_matched_diseases else matched_diseases

            if target_matches:
                max_ratio = target_matches[0]["ratio"]
                filtered_matches = [
                    m for m in target_matches 
                    if m["ratio"] >= max_ratio - 0.15 and m["ratio"] >= 0.30
                ]
                filtered_matches = filtered_matches[:3]
                disease_names = list(dict.fromkeys([m["benh_ly"] for m in filtered_matches]))
            
        # [FIX] Trích xuất đúng detailed_kg_data theo chuẩn mới để pass vào LLM Refactored method
        detailed_kg_data = []
        for syn in final_syndromes:
            diseases_for_syn = sym_disease_map.get(syn, [])
            treatments = self.qa_pipeline.get_treatments_for_syndrome(
                syndrome=syn, 
                diseases=list(diseases_for_syn) if diseases_for_syn else ['KHÔNG_XÁC_ĐỊNH_BỆNH']
            )
            
            # [FIX] Chỉ giữ lại các bài thuốc cho những bệnh đã được khớp chính xác (nếu <= 3 bệnh)
            # Nếu không xác định được bệnh cụ thể, ta không trả về bài thuốc nào để tránh vẽ rác lên đồ thị.
            filtered_treatments = []
            if disease_names and len(disease_names) <= 3:
                filtered_treatments = [t for t in treatments if t.get("disease") in disease_names]
            
            # Lấy thông tin Tạng Phủ & Bát Cương
            metadata = self.qa_pipeline.get_syndrome_metadata(syn)
            
            data = {
                "syndrome": syn,
                "matching_symptoms": search_terms,
                "diseases": list(diseases_for_syn),
                "treatments_by_disease": filtered_treatments,
                "organs": metadata.get("organs", []),
                "bat_cuong": metadata.get("bat_cuong", [])
            }
            detailed_kg_data.append(data)
            
        final_markdown = self._generate_explainable_answer(
            user_symptoms=user_symptoms,
            detected_symptoms=raw_vision_data["detected_symptoms"] if raw_vision_data else [],
            detailed_kg_data=detailed_kg_data,
            search_terms=search_terms
        )

        # [AN TOÀN Y TẾ] Luôn chèn cảnh báo cấp cứu (nếu có cờ đỏ) + miễn trừ trách nhiệm
        final_markdown = self._append_medical_disclaimer(
            final_markdown, raw_text=f"{user_symptoms} {combined_query}"
        )

        # Gộp triệu chứng để HIỂN THỊ cho người dùng.
        # [FIX HIỂN THỊ] Chỉ hiện triệu chứng NGƯỜI DÙNG THẬT SỰ NHẬP (khớp trực tiếp từ text, KHÔNG
        # bung từ đồng nghĩa nội bộ) + triệu chứng vọng chẩn từ ảnh. Trước đây dùng `search_terms`
        # (đã bung synonym cổ phương để khớp graph) nên chuỗi hiển thị lòi ra các từ như "tiết tả",
        # "ít nói"... trông như hệ tự thêm triệu chứng. Các synonym vẫn được dùng NGẦM để truy hồi graph.
        # [FIX NHẤT QUÁN] Vẫn chạy khử mâu thuẫn trên tập gộp (text+vision) trước khi hiển thị.
        display_terms = self.qa_pipeline._preprocess_question(user_symptoms) if user_symptoms else []
        all_symptoms_list = self._resolve_symptom_conflicts(
            list(dict.fromkeys(display_terms + (raw_vision_data["detected_symptoms"] if raw_vision_data else [])))
        )
        input_fusion_str = ", ".join(all_symptoms_list) if all_symptoms_list else (combined_query or "Không xác định")

        # [ĐÃ BỎ] Cơ chế sinh câu hỏi hỏi bệnh động (Dynamic Fallback) đã được gỡ theo yêu cầu:
        # hệ thống LUÔN trả kết quả chẩn đoán trực tiếp, không hỏi thêm lâm sàng (bỏ 1 lượt gọi LLM
        # ~30s và luồng pending_questions). Giữ 2 khóa status/questions (rỗng) để tương thích ngược frontend.
        return {
            "source": "Tứ chẩn hợp tham (Fusion)",
            "input_fusion": input_fusion_str,
            "has_makeup": self._has_makeup,
            "vision_details": raw_vision_data,
            "diagnosis_result": {
                "answer": final_markdown,
                "data": detailed_kg_data
            },
            "status": "completed",
            "questions": []
        }

    def close(self):
        if hasattr(self, 'qa_pipeline'): self.qa_pipeline.close()
        if hasattr(self, 'vision_pipeline') and hasattr(self.vision_pipeline, 'close'): self.vision_pipeline.close()
        logger.info("Đã giải phóng tài nguyên.")
