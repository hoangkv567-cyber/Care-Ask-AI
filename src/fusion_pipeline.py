# src/fusion_pipeline.py
import logging
import re
import unicodedata
from src.pipeline import TCMTonguePipeline
from src.qa_system import TCMQA

from src.utils import normalize_symptoms_text

logger = logging.getLogger(__name__)


def _fold_vn(s: str) -> str:
    """Bỏ dấu thanh + hạ hoa-thường để so khớp tên hội chứng bất kể biến thể chính tả cũ/mới
    ('Can Hoả Phạm Phế' == 'Can hỏa phạm phế' — dấu thanh đặt trên nguyên âm khác nhau). Đổi đ->d,
    gộp khoảng trắng. CHỈ dùng cho so khớp trùng tên (không dùng để hiển thị)."""
    s = (s or "").lower().strip().replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip()

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
            # 'vết răng' là tên trong file mapping vọng chẩn, nhưng KG (10 node) và CSV (11 dòng)
            # đều dùng 'hằn răng' — không quy đổi thì mất bằng chứng Tỳ khí hư mạnh nhất khi so khớp.
            "rìa lưỡi có vết răng": "rìa lưỡi có hằn răng",
            "lưỡi có vết răng": "rìa lưỡi có hằn răng",
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

        # 3b. Mâu thuẫn rêu MỎNG vs rêu ÍT: một lưỡi không thể vừa 'rêu trắng mỏng' (có rêu, gần
        # sinh lý) vừa 'rêu ít' (thiểu rêu — dấu âm hư). LLM matcher hay map cả hai từ một mô tả
        # 'rêu mỏng' -> giữ quan sát cụ thể (trắng mỏng), loại 'rêu ít' để không bơm bằng chứng
        # âm hư giả vào chấm điểm hội chứng.
        has_thin_coating = any(
            kw in symptoms_lower
            for kw in ["rêu trắng mỏng", "rêu lưỡi trắng mỏng", "rêu mỏng trắng", "rêu mỏng", "rêu lưỡi mỏng"]
        )
        if has_thin_coating:
            symptoms = [
                s for s in symptoms
                if s.lower().strip() not in ["rêu ít", "ít rêu", "rêu lưỡi ít", "lưỡi ít rêu"]
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

        # 5. Mâu thuẫn dấu răng: VLM đôi khi tả nước đôi "vết lõm gợn sóng nhẹ" (map -> có vết răng)
        # rồi chốt "không thấy rõ dấu răng" (map -> lưỡi không có hằn răng). Có phát hiện dương tính
        # thì loại triệu chứng phủ định tương ứng (ưu tiên phát hiện, đồng bộ với rule 2).
        _tooth_kw = ("vết răng", "dấu răng", "hằn răng")
        has_tooth_positive = any(
            any(kw in s for kw in _tooth_kw) and "không" not in s
            for s in symptoms_lower
        )
        if has_tooth_positive:
            symptoms = [
                s for s in symptoms
                if not (any(kw in s.lower() for kw in _tooth_kw) and "không" in s.lower())
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

    def _get_symptom_vocab(self) -> set:
        """Từ vựng triệu chứng (các trường triệu_chứng tách phẩy từ CSV, 4-30 ký tự) — dùng cho
        bộ censor chống bịa triệu chứng. Cache 1 lần trên instance."""
        vocab = getattr(self, "_symptom_vocab", None)
        if vocab is not None:
            return vocab
        vocab = set()
        for r in getattr(self, "csv_rows", None) or []:
            for f in r.get("triệu_chứng", "").split(","):
                f = f.strip().lower()
                # Ngưỡng 2 ký tự để KHÔNG bỏ sót triệu chứng ngắn thật ('ho','sốt','nấc','phù','nôn') —
                # bộ censor gỡ theo ranh giới từ (\b + lookahead) nên term ngắn không khớp chuỗi con.
                if 2 <= len(f) <= 30:
                    vocab.add(f)
        self._symptom_vocab = vocab
        return vocab

    def _get_neg_protected_phrases(self) -> list:
        """Các cụm triệu chứng CSV chứa 'không/chưa/chẳng' (tên nội tại) — dùng bảo vệ khỏi bộ che
        phủ định. Cache trên instance (tránh dựng lại mỗi lần gọi khớp bệnh danh)."""
        cached = getattr(self, "_neg_protected", None)
        if cached is not None:
            return cached
        cached = [f for f in self._get_symptom_vocab() if any(n in f for n in ("không", "chưa", "chẳng"))]
        self._neg_protected = cached
        return cached

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
                    p = re.escape(phrase)
                    # (1) phrase ĐẦU liệt kê sau ĐỘNG TỪ nhân-quả: "dẫn đến chóng mặt và X" -> GIỮ
                    # động từ, bỏ "chóng mặt và " -> "dẫn đến X" (không để lại 'và' mồ côi/mất vị ngữ).
                    text = re.sub(rf'(?i)(\b(?:gây ra|gây nên|dẫn đến|dẫn tới|gây|phát sinh|sinh ra)\s+){p}\s+(?:và|hoặc)\s+', r'\1', text)
                    # (2) phrase CUỐI/giữa liệt kê: "...và chóng mặt" -> bỏ " và chóng mặt" (phrase luôn
                    # là triệu chứng KHÔNG có trong input nên gỡ an toàn).
                    text = re.sub(rf'(?i)\s*(?:,|và|hoặc)\s+{p}\b', '', text)
                    # (3) chủ ngữ ĐẦU liệt kê trước động từ: "Chóng mặt và đau đầu là do..." -> bỏ "Chóng mặt và "
                    text = re.sub(rf'(?i)\b{p}\s+(?:và|hoặc)\s+', '', text)
                    # (4) connector/động từ + phrase đơn -> bỏ cả cụm
                    text = re.sub(rf'(?i)\b(?:gây ra|dẫn đến|gây|kèm|như)\s+{p}\b', '', text)
                    # (5) phrase trần còn sót -> bỏ
                    text = re.sub(rf'(?i)\b{p}\b', '', text)

        # [FIX CHỐNG BỊA TRIỆU CHỨNG — TỔNG QUÁT] Ngoài các luật cứng ở trên, quét theo TOÀN BỘ
        # từ vựng triệu chứng của CSV: thuật ngữ triệu chứng nào xuất hiện trong biện chứng mà
        # KHÔNG có trong danh sách đầu vào (kể cả sau quy đổi đồng nghĩa) sẽ bị gỡ cùng cơ chế.
        # Chặn triệt để kiểu LLM 7B chép gợi ý mẫu ('chóng mặt', 'hoa mắt'...) vào ca không có.
        input_terms = [t.strip().lower() for t in symptoms_str.split(",") if t.strip()]
        try:
            input_norm = set(self._normalize_symptoms(list(input_terms)))
        except Exception:
            input_norm = set(input_terms)

        def _is_input_symptom(term: str) -> bool:
            if term in symptoms_lower:
                return True
            for t in input_terms:
                if t in term or term in t:
                    return True
            try:
                n = self._normalize_symptoms([term])
            except Exception:
                n = [term]
            return bool(n) and (n[0] in input_norm or n[0] in symptoms_lower)

        # CHỈ gỡ khi triệu chứng lạ đứng sau ĐỘNG TỪ NHÂN-QUẢ ("dẫn đến chóng mặt", "gây hoa mắt",
        # "phát sinh X") — tức LLM đang KHẲNG ĐỊNH bệnh nhân có triệu chứng đó. Không gỡ danh từ
        # xuất hiện trong văn cơ chế thông thường (vd "vận hóa thức ăn") để tránh phá câu đúng.
        text_lower = text.lower()
        # 'cảm thấy|cảm nhận|cảm giác' = LLM KHẲNG ĐỊNH bệnh nhân CÓ cảm giác đó ('làm cho bệnh nhân
        # cảm thấy khó thở') -> nếu triệu chứng sau nó không có trong lời khai thì là bịa, gỡ như các
        # động từ nhân-quả khác. (Triệu chứng CÓ trong input đã được _is_input_symptom chừa.)
        _causal = r'(?:gây ra|gây nên|dẫn đến|dẫn tới|gây|phát sinh|sinh ra|biểu hiện qua|biểu hiện bằng|xuất hiện|thể hiện qua|thể hiện bằng|thể hiện ở|cảm thấy|cảm nhận|cảm giác)'
        # Mệnh đề nhân-quả MẠNH cho vị trí chủ ngữ ("Chóng mặt và đau đầu LÀ DO..."). Cố ý KHÔNG
        # dùng 'khiến/gây' đứng sau dấu phẩy và không cho lookahead băng qua ,.; — tránh gỡ nhầm
        # danh từ cơ chế ("vận hóa tân dịch và thức ăn, khiến cho...").
        _subj_causal = r'(?:là do|đều do|là vì|là bởi|(?:cũng\s+)?là(?:\s+một)?\s+(?:biểu hiện|dấu hiệu))'
        for term in self._get_symptom_vocab():
            if term not in text_lower or _is_input_symptom(term):
                continue
            pat = re.escape(term)
            # (a) sau động từ nhân-quả: "dẫn đến chóng mặt". Xử lý cả khi term nằm TRONG liệt kê
            # "verb X và Y" — nếu chỉ xóa "verb X" sẽ để lại "và Y" mồ côi / mất vị ngữ.
            #   (a1) term ĐẦU liệt kê -> GIỮ động từ, bỏ "X và": "dẫn đến chóng mặt và mệt" -> "dẫn đến mệt"
            new_text = re.sub(rf'(?i)(\b{_causal}\s+){pat}\s+(?:và|hoặc)\s+', r'\1', text)
            #   (a2) term CUỐI/giữa liệt kê ngay trong cụm nhân-quả -> bỏ " và X" (giữ phần trước)
            new_text = re.sub(rf'(?i)(\b{_causal}\b[^,.;]{{0,40}}?)\s*(?:,|và|hoặc)\s+{pat}(?![\wÀ-ỹ])', r'\1', new_text)
            #   (a3) term đơn lẻ sau động từ -> bỏ cả cụm "verb X"
            new_text = re.sub(rf'(?i)\b{_causal}\s+{pat}(?![\wÀ-ỹ])', '', new_text)
            # (b) chủ ngữ ĐẦU liệt kê: "Chóng mặt và đau đầu là do..." -> bỏ "Chóng mặt và "
            new_text = re.sub(
                rf'(?i)(?<![\wÀ-ỹ]){pat}\s*(?:,|và)\s*(?=[^,.;]{{0,60}}\b{_subj_causal})',
                '', new_text)
            # (c) chủ ngữ CUỐI liệt kê: "...và lưỡi nhợt là do..." -> bỏ " và lưỡi nhợt"
            new_text = re.sub(
                rf'(?i)(?:,|\bvà)\s+{pat}(?=\s*[^,.;]{{0,60}}\b{_subj_causal})',
                '', new_text)
            # (d) chủ ngữ ĐƠN đầu câu/mệnh đề: "Lưỡi nhợt cũng là dấu hiệu của huyết hư, vì..."
            # — không có động từ nhân-quả phía trước (a) cũng không nằm trong liệt kê (b)/(c) nên
            # 3 khối trên đều lọt. Cả câu là một khẳng định triệu chứng bịa -> gỡ TRỌN câu (giới
            # hạn [^.] để không ăn lan sang câu bên cạnh; nhánh $ xử lý câu cuối không có dấu chấm).
            new_text = re.sub(
                rf'(?i)(?<![\wÀ-ỹ]){pat}\s+[^.;]{{0,40}}?\b{_subj_causal}\b[^.]*?(?:\.\s*|$)',
                '', new_text)
            if new_text != text:
                text = new_text
                text_lower = text.lower()

        # Dọn mệnh đề nhân-quả MỒ CÔI còn trơ động từ sau khi gỡ triệu chứng bịa: 'làm cho bệnh
        # nhân cảm thấy khó thở và mệt mỏi' -> (gỡ khó thở/mệt mỏi) -> 'làm cho bệnh nhân .' -> gỡ
        # nốt cụm dẫn 'làm cho/khiến [bệnh nhân/người bệnh/cơ thể] [cảm thấy]' khi KHÔNG còn tân ngữ
        # (theo ngay sau là dấu câu/hết chuỗi). KHÔNG đụng 'làm cho Phế khí bế tắc' (còn tân ngữ).
        text = re.sub(
            r'(?i)[,;]?\s*(?:làm cho|khiến cho|khiến)\s+'
            r'(?:bệnh nhân|người bệnh|cơ thể|người)?\s*'
            r'(?:cảm thấy|cảm nhận|cảm giác)?\s*(?=[.;]|$)',
            '', text)

        # LIÊN TỪ MỒ CÔI cuối câu: gỡ triệu chứng bịa sau 'nhưng vẫn/và...' để lại đuôi cụt ('uống
        # nhiều nhưng vẫn.', 'mất ngủ và.') — cũng bắt trường hợp LLM 7B tự sinh câu cụt. Chỉ gỡ khi
        # liên từ đứng NGAY trước dấu chấm/phẩy/hết chuỗi (không còn vế sau) -> không phá 'A và B.'.
        text = re.sub(
            r'(?i)\s+(?:nhưng(?:\s+vẫn)?|và(?:\s+vẫn)?|mà|hoặc|cùng(?:\s+với)?|kèm(?:\s+theo)?)\s*(?=[.;]|$)',
            '', text)
        # MỆNH ĐỀ 'do/vì/khiến...' MỒ CÔI: tân ngữ bị gỡ để lại 'uống nhiều do cơ thể luôn.' / 'khiến
        # cơ thể.' -> gỡ khi sau liên từ chỉ còn ĐỘN TỪ (cơ thể/bệnh nhân/nó) + trạng từ (luôn/vẫn...)
        # rồi hết câu. KHÔNG đụng 'do <lý do thật>.' (vd 'do khí hư.') vì sau 'do' là danh từ thật.
        text = re.sub(
            r'(?i)\s+(?:do|vì|bởi(?:\s+vì)?|khiến(?:\s+cho)?|làm(?:\s+cho)?|gây(?:\s+ra)?|để)\s+'
            r'(?:cho\s+)?(?:cơ thể|bệnh nhân|người bệnh|nó)\s*'
            r'(?:luôn|vẫn|còn|mãi|rất|hơi|khá)?\s*(?=[.;]|$)',
            '', text)

        # Làm sạch các khoảng trắng và dấu câu thừa sau khi xóa
        text = re.sub(r'[ \t]+([,.])', r'\1', text)
        text = re.sub(r'[ \t]*,[ \t]*,', ',', text)
        text = re.sub(r'[ \t]*,[ \t]*\.', '.', text)
        text = re.sub(r'[ \t]*,[ \t]*và[ \t]+', ' và ', text)
        text = re.sub(r'[ \t]+và[ \t]+và[ \t]+', ' và ', text)
        text = re.sub(r'[ \t]+và[ \t]*,', ',', text)
        text = re.sub(r'[ \t]+,[ \t]*và[ \t]+', ' và ', text)
        
        # Làm sạch khoảng trắng thừa trên từng dòng và bảo toàn ký tự xuống dòng
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        text = "\n".join(lines).strip()
        # Viết hoa lại chữ đầu câu bị hở sau khi gỡ chủ ngữ — phủ cả 3 vị trí: sau [.!?], ĐẦU
        # chuỗi, và ĐẦU mỗi dòng (câu mở đầu bằng chữ thường khi triệu chứng bịa bị gỡ ở đầu).
        text = re.sub(r'(^|\n\s*|[.!?]\s+)([a-zà-ỹđ])',
                      lambda m: m.group(1) + m.group(2).upper(), text)
        return text.strip()

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

        # Mô tả từ VLM cloud (Qwen3-VL) đã là tiếng Việt sẵn -> không dịch nữa, tránh gọi LLM
        # thừa và lỗi "dịch" tiếng Việt sang tiếng Việt gây lặp từ. Nhận diện bằng dấu tiếng Việt
        # (mô tả tiếng Anh của LLaVA không bao giờ chứa các ký tự này).
        import re as _re
        if _re.search(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", english_text.lower()):
            return self._clean_translated_text(english_text)

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
        9. SPECIAL RULE FOR NORMAL/PHYSIOLOGICAL FINDINGS (MUST OBEY): Moist coating (rêu nhuận, rêu ẩm), evenly distributed coating, "not dry", "not peeled", "no cracks" are NORMAL physiological observations, NOT symptoms. You MUST NOT select 'rêu lưỡi nhuận', 'rêu nhuận', 'rêu lưỡi trắng nhuận' or any symptom for them. NEVER derive a positive symptom from the absence of an abnormality: "không thấy bong tróc hay khô" means the coating is normal in that aspect — it must NOT become 'rêu lưỡi nhuận' or any other selected symptom.
        10. CANONICAL TONGUE-COATING NAMES (CONSISTENCY RULE): For a thin white coating (rêu mỏng màu trắng / thin white coating), you MUST always select 'rêu trắng mỏng' (or 'rêu lưỡi trắng mỏng') — NEVER select the variant 'rêu lưỡi trắng nhạt' / 'rêu trắng nhạt' for this finding ("trắng nhạt" is a color nuance, not the canonical thin-white-coating symptom). Separately, if the coating is explicitly greasy/sticky (nhờn, nhớt, dính — kể cả 'hơi nhờn' / slightly greasy), you MUST ALSO select 'rêu lưỡi trắng nhớt' (or 'rêu trắng nhớt') if it is in the candidate list.
        11. SPECIAL RULE FOR TONGUE BODY SHAPE (MUST OBEY): If the tongue body is described as thick, slightly thick, enlarged, puffy, or swollen (e.g. 'hình thể lưỡi hơi dày', 'lưỡi dày', 'thân lưỡi hơi phồng', 'thick/swollen/enlarged tongue body'), you MUST select 'lưỡi bệu' (or 'thân lưỡi bệu' if 'lưỡi bệu' is absent from the list). Do NOT skip this finding.

        EXTERNAL-CAUSE EXCLUSION RULE (MUST OBEY):
        - If a feature is attributed to an EXTERNAL, NON-PATHOLOGICAL cause — lighting ("do ánh sáng", "due to indoor lighting", "because of the light", shadows), makeup ("do trang điểm", lipstick, blush), camera/photo quality or shooting angle — you MUST NOT select the corresponding symptom.
        - Example: "sắc mặt nhợt nhạt với một chút vàng do ánh sáng trong nhà" / "a slight yellow tint due to indoor lighting" -> you MUST NOT select "mặt vàng" or "da vàng" (the yellow is from the lamp, not the patient).

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
            # Lưới an toàn deterministic: LLM matcher hay bắt/bỏ thất thường vài dấu hiệu lưỡi
            # tùy cách diễn đạt của VLM -> chốt bằng code để mọi lần chạy cho cùng kết quả
            valid_mapped = self._force_add_tongue_signs(valid_mapped, description, candidate_symptoms)
            # Lưới an toàn deterministic: chặn triệu chứng bị map từ câu PHỦ ĐỊNH
            # ('không khô xỉn' -> Da khô) mà prompt NEGATION RULES thỉnh thoảng vẫn để lọt
            valid_mapped = self._drop_negated_signs(valid_mapped, description)
            # Lưới an toàn deterministic: chặn ĐẢO CỰC (mô tả 'môi nhợt' bị map thành 'môi thâm')
            valid_mapped = self._drop_polarity_conflicts(valid_mapped, description)
            # Lưới an toàn deterministic: LLM 7B hay bỏ qua luật ngoại cảnh trong prompt
            return self._filter_external_attributions(valid_mapped, description)
        except Exception as e:
            logger.error(f"Lỗi mapping triệu chứng bằng LLM: {e}")
            # [FALLBACK VỌNG CHẨN TẤT ĐỊNH] LLM matcher không gọi được (hết quota/mất mạng)
            # -> khớp từ điển theo biên từ để dấu lưỡi/mặt KHÔNG rơi trắng trong chế độ suy
            # giảm; vẫn đi qua đủ lưới an toàn tất định (chốt dấu lưỡi, phủ định, ngoại cảnh).
            fallback = self._map_desc_fallback_dict(description, candidate_symptoms)
            fallback = self._force_add_tongue_signs(fallback, description, candidate_symptoms)
            fallback = self._drop_negated_signs(fallback, description)
            fallback = self._drop_polarity_conflicts(fallback, description)
            fallback = self._filter_external_attributions(fallback, description)
            if fallback:
                logger.info(f"[FALLBACK VỌNG CHẨN] Khớp từ điển tất định: {fallback}")
            return fallback

    # [CHỐNG ĐẢO CỰC VỌNG CHẨN] Cặp (tập tên triệu chứng "cực NÀY", regex mô tả "cực ĐỐI LẬP",
    # regex mô tả "cực của chính nó"). LLM matcher đôi khi map mô tả 'môi nhợt' -> 'môi thâm' khi
    # danh sách ứng viên thiếu tên đúng cực -> đưa vào chẩn đoán một dấu hiệu NGƯỢC HƯỚNG (môi thâm
    # = huyết ứ/hàn, đối nghịch huyết hư). Chỉ LOẠI khi mô tả nêu RÕ cực đối lập và KHÔNG nêu cực
    # của chính triệu chứng đã map (giữ đúng ca 'môi tím tái' -> môi thâm là hợp lệ).
    _POLARITY_GUARDS = [
        # môi sẫm/tím bị map trong khi mô tả nói môi NHỢT/NHẠT (và không nói môi thâm/tím/tái)
        (("môi thâm", "môi tím", "môi tái", "môi bầm", "môi thâm tím", "môi tím tái", "môi thâm đen",
          "môi tía", "môi sẫm"),
         r"môi[^.,;\n]{0,24}(nhợt|nhạt|trắng)",
         r"môi[^.,;\n]{0,24}(thâm|tím|tái|bầm|sẫm|đen|tía)"),
        # môi nhợt bị map trong khi mô tả nói môi THÂM/TÍM (và không nói môi nhợt/nhạt)
        (("môi nhợt", "môi nhợt nhạt", "môi nhạt"),
         r"môi[^.,;\n]{0,24}(thâm|tím|tái|bầm|sẫm|tía)",
         r"môi[^.,;\n]{0,24}(nhợt|nhạt|trắng)"),
        # 'rêu ít' (thiểu rêu — dấu ÂM HƯ) bị map trong khi mô tả nói rêu MỎNG phân bố đều (gần
        # sinh lý). Rêu mỏng ≠ rêu ít: map nhầm tiếp bằng chứng âm hư giả cho ca thuần khí hư
        # (đã xảy ra thật: cốt lõi thành 'Khí âm lưỡng hư' dù lưỡi hồng nhạt ẩm, không dấu âm hư).
        (("rêu ít", "ít rêu", "rêu lưỡi ít", "lưỡi ít rêu"),
         r"rêu[^.,;\n]{0,30}mỏng|rêu[^.,;\n]{0,34}phân bố đều|rêu bình thường|thin coat",
         r"rêu[^.,;\n]{0,26}(rất\s+)?ít|ít\s+rêu|thiểu\s+rêu|gần như không|không\s+(có\s+)?rêu"
         r"|bong tróc|scanty|sparse|little coat|peeled"),
    ]

    def _drop_polarity_conflicts(self, mapped: list, description: str) -> list:
        desc_l = (description or "").lower()
        if not desc_l:
            return mapped
        kept = []
        for s in mapped:
            s_l = s.lower().strip()
            dropped = False
            for names, opposite_re, own_re in self._POLARITY_GUARDS:
                if s_l in names:
                    if re.search(opposite_re, desc_l) and not re.search(own_re, desc_l):
                        logger.info(f"[CHỐNG ĐẢO CỰC] Loại '{s}': mô tả nêu cực đối lập, không nêu cực của nó")
                        dropped = True
                    break
            if not dropped:
                kept.append(s)
        return kept

    def _map_desc_fallback_dict(self, description: str, candidate_symptoms: list) -> list:
        """Khớp trực tiếp tên triệu chứng chuẩn nằm TRONG mô tả VLM theo biên từ — chỉ dùng làm
        fallback khi LLM matcher lỗi. Bảo thủ: ứng viên >= 6 ký tự, bỏ mệnh đề chứa phủ định,
        khử cụm-con (giữ cụm dài đặc hiệu hơn khi cả hai cùng khớp)."""
        desc_l = (description or "").lower()
        if not desc_l:
            return []
        clauses = [cl for cl in re.split(r'[,.;\n]', desc_l) if cl.strip()]
        # Bỏ mệnh đề phủ định VÀ mệnh đề tả trạng thái sinh lý ('chất lưỡi hồng bình thường'
        # không phải triệu chứng — tương ứng luật NORMAL TONGUE COLOR của LLM matcher)
        pos_clauses = [cl for cl in clauses
                       if not re.search(r'\b(không|chưa|hết|no|not|without'
                                        r'|bình thường|khỏe mạnh|normal|healthy)\b', cl)]
        hits = []
        for cand in candidate_symptoms:
            c_l = (cand or "").lower().strip()
            if len(c_l) < 6:
                continue
            if any(self._kw_hit_clean(cl, [c_l]) for cl in pos_clauses) and cand not in hits:
                hits.append(cand)
        return [h for h in hits
                if not any(h != o and re.search(r"\b" + re.escape(h.lower()) + r"\b", o.lower())
                           for o in hits)]

    # (từ khóa nhận diện triệu chứng đã map, các từ gốc đặc trưng của nó trong mô tả VLM)
    _NEGATION_GUARDS = [
        (("da khô", "khô xỉn"), ("khô",)),
        (("quầng thâm", "quầng đen"), ("quầng",)),
        (("mặt phù", "sưng phù", "phù ở"), ("phù", "phù nề", "sưng", "húp", "mọng")),
        (("có ban", "ban đỏ", "xuất huyết"), ("ban đỏ", "mẩn", "xuất huyết")),
        (("vết nứt", "lưỡi nứt"), ("nứt",)),
        (("bong tróc",), ("bong tróc", "bong")),
        (("bọng mắt",), ("bọng",)),
        (("mặt đỏ", "vùng đỏ", "gò má đỏ", "đỏ bừng"), ("đỏ",)),
        (("nhớt", "nhờn"), ("nhớt", "nhờn", "dính")),
        (("lưỡi khô", "rêu khô"), ("khô",)),
    ]
    # Cụm chứa từ gốc nhưng KHÔNG phải dấu hiệu bệnh ('da phù HỢP', 'ban ĐÊM') -> xóa trước khi soi.
    _NEG_FALSE_FRIENDS = ("phù hợp", "ban đêm", "ban ngày", "ban đầu")
    _NEG_MARK = r"(?:không|chẳng|chưa|ko|no|not|without)"

    def _drop_negated_signs(self, mapped: list, description: str) -> list:
        """[CHỐNG MATCH TỪ CÂU PHỦ ĐỊNH] LLM matcher thỉnh thoảng map triệu chứng từ đặc điểm mà mô tả
        đã PHỦ ĐỊNH ('da không khô xỉn, ẩm mượt' -> Da khô). Chỉ LOẠI khi từ gốc bị phủ định ngay
        trước (trong ~2 từ) và KHÔNG có lần nào được khẳng định. Nếu mô tả KHÔNG nhắc tới từ gốc
        (matcher dùng đồng nghĩa/tiếng Anh: 'flushed cheeks' -> Gò má đỏ) thì TIN matcher, GIỮ lại —
        bản cũ đòi từ gốc literal dương tính nên xóa oan mọi ánh xạ đồng nghĩa và toàn bộ path tiếng Anh."""
        desc_l = (description or "").lower()
        for ff in self._NEG_FALSE_FRIENDS:
            desc_l = desc_l.replace(ff, " ")

        def _root_status(roots):
            """'pos' nếu có lần xuất hiện KHÔNG bị phủ định ngay trước; 'neg' nếu MỌI lần đều bị phủ
            định; 'absent' nếu không nhắc tới -> caller sẽ GIỮ (tin matcher)."""
            found = False
            for root in roots:
                # \b hai đầu: 'khô' KHÔNG được dính trong 'không' (bẫy âm tiết tiếng Việt kinh điển)
                for mt in re.finditer(r"\b" + re.escape(root) + r"\b", desc_l):
                    found = True
                    pre = desc_l[max(0, mt.start() - 28):mt.start()]
                    # phủ định phải đứng ngay trước (tối đa 2 âm tiết chen giữa: 'không thấy', 'không có')
                    if not re.search(self._NEG_MARK + r"(?:\s+\S+){0,2}\s*$", pre):
                        return "pos"
            return "neg" if found else "absent"

        kept = []
        for s in mapped:
            s_l = s.lower()
            dropped = False
            for keys, roots in self._NEGATION_GUARDS:
                if any(k in s_l for k in keys):
                    if _root_status(roots) == "neg":
                        logger.info(f"[CHỐNG PHỦ ĐỊNH] Loại '{s}': từ gốc chỉ xuất hiện ở dạng bị phủ định")
                        dropped = True
                    break
            if not dropped:
                kept.append(s)
        return kept

    # Nhóm triệu chứng lưỡi ĐỒNG NGHĨA -> gom về MỘT tên canonical (phần tử đầu nhóm) để mọi lần
    # chạy cho cùng tên node, điểm hội chứng không dao động theo cách LLM chọn biến thể tên.
    _TONGUE_SYNONYM_GROUPS = [
        ["rìa lưỡi có vết răng", "rìa lưỡi có hằn răng", "lưỡi bệu có dấu răng", "lưỡi có dấu răng",
         "lưỡi nhạt bệu có dấu răng", "thân lưỡi bệu có dấu răng", "lưỡi nhạt có dấu răng"],
        ["rêu trắng mỏng", "rêu lưỡi trắng mỏng", "rêu mỏng trắng", "lưỡi trắng mỏng", "rêu lưỡi trắng nhạt", "rêu trắng nhạt"],
        ["rêu lưỡi trắng nhớt", "rêu trắng nhớt", "rêu lưỡi trắng nhờn", "rêu trắng nhờn"],
        ["lưỡi bệu", "thân lưỡi bệu", "chất lưỡi bệu", "rêu lưỡi bệu", "lưỡi trắng bệu"],
        ["lưỡi hồng nhạt", "chất lưỡi hồng nhạt"],
    ]

    def _force_add_tongue_signs(self, mapped: list, description: str, candidate_symptoms: list) -> list:
        """[NHẤT QUÁN VỌNG CHẨN] Chốt deterministic các dấu hiệu lưỡi mà LLM matcher bắt/bỏ hoặc
        chọn biến thể tên thất thường giữa các lần chạy — cùng một mô tả phải luôn cho cùng một tập
        triệu chứng. Gồm 2 lớp: (1) bổ sung dấu hiệu theo regex trên mô tả, (2) chuẩn hóa đồng nghĩa."""
        import re as _re
        desc_l = (description or "").lower()
        cand_by_lower = {c.lower(): c for c in candidate_symptoms}

        def _add(names):
            for n in names:
                if n in cand_by_lower:
                    if cand_by_lower[n] not in mapped:
                        mapped.append(cand_by_lower[n])
                    return

        # 1. Rêu nhờn/nhớt/dính (kể cả 'hơi nhờn') -> rêu trắng nhớt (khi rêu trắng, không phủ định)
        greasy = _re.search(r"(?<!không )(?<!không có )(?<!không thấy )(nhờn|nhớt|dính)", desc_l)
        if greasy and "trắng" in desc_l and "vàng" not in desc_l:
            _add(["rêu lưỡi trắng nhớt", "rêu trắng nhớt"])

        # 2. Thân lưỡi dày/phồng/to -> lưỡi bệu
        if _re.search(r"(hình thể|thân) lưỡi\s+(hơi\s+)?(dày|phồng|to|bệu)|lưỡi\s+(hơi\s+)?(phồng|bệu)", desc_l):
            _add(["lưỡi bệu", "thân lưỡi bệu", "chất lưỡi bệu"])

        # 3. Dấu răng/vết lõm gợn sóng ở mép -> rìa lưỡi có vết răng
        # (lookbehind phải phủ cả 'không CÓ dấu răng' — bản cũ chỉ chặn 'không dấu răng')
        if _re.search(r"(?<!không )(?<!không có )(?<!không thấy )(dấu răng|hằn răng|vết răng|lõm gợn sóng)", desc_l):
            _add(["rìa lưỡi có vết răng", "rìa lưỡi có hằn răng", "lưỡi bệu có dấu răng"])

        # 4. Thân lưỡi hồng nhạt (màu nhạt bệnh lý nhẹ) -> lưỡi hồng nhạt
        if _re.search(r"(?<!không )(màu\s+)?hồng nhạt", desc_l):
            _add(["lưỡi hồng nhạt", "chất lưỡi hồng nhạt"])

        # 5. Rêu rất ít/gần như không rêu -> rêu ít (dấu âm hư quan trọng; 'rêu rất ít' không chứa
        # nguyên cụm 'rêu ít' nên khớp từ điển bỏ sót)
        if _re.search(r"rêu( lưỡi)?\s+(rất |hơi |khá )?ít|gần như không có rêu|không thấy rêu", desc_l):
            _add(["rêu ít", "rêu lưỡi ít", "ít rêu"])

        # Duyệt theo MỆNH ĐỀ, bỏ mệnh đề có phủ định — vì 'không có mặt phù'/'không có ... gò má'
        # có từ chen giữa nên lookbehind cố định (?<!không ) KHÔNG bắt được (đã bịa 'mặt phù' cho ca
        # mô tả 'không có mặt phù'). Tách câu rồi loại mệnh đề chứa không/chưa/no/not.
        _pos_clauses_fa = [_cl for _cl in _re.split(r'[,.;]', desc_l)
                           if not _re.search(r'\b(không|chưa|chẳng|no|not|without)\b', _cl)]

        # 6. Gò má ửng đỏ/đỏ (mô tả mặt; gated theo danh sách ứng viên nên vô hại với lưỡi)
        if any(_re.search(r"gò má[^,.;]{0,14}(ửng\s+)?đỏ", _cl) for _cl in _pos_clauses_fa):
            _add(["hai gò má đỏ", "gò má đỏ", "2 gò má đỏ"])

        # 7. Mặt phù/sưng nề (cả mặt) -> Mặt phù ('mặt hơi phù nề' không chứa nguyên cụm 'mặt phù')
        if any(_re.search(r"mặt[^,.;]{0,10}(phù|sưng húp|sưng nề)", _cl) for _cl in _pos_clauses_fa):
            _add(["mặt phù"])

        # 5. Chuẩn hóa đồng nghĩa: mỗi nhóm chỉ giữ MỘT tên canonical (ưu tiên phần tử đầu nhóm
        # nếu nó có trong danh sách ứng viên; nếu không giữ biến thể đã map đầu tiên)
        normalized = []
        for m in mapped:
            m_l = m.lower()
            target = m
            for group in self._TONGUE_SYNONYM_GROUPS:
                if m_l in group:
                    canonical = next((n for n in group if n in cand_by_lower), None)
                    if canonical:
                        target = cand_by_lower[canonical]
                    break
            if target not in normalized:
                normalized.append(target)
        return normalized

    # Đặc điểm bị mô tả quy cho NGUYÊN NHÂN NGOẠI CẢNH (không phải bệnh lý) — dùng cho bộ lọc dưới.
    # BỌC \b hai đầu: chặn token ngắn/đa nghĩa khớp NHẦM chuỗi con ('light' trong 'slightly',
    # 'son' trong 'season/reason', 'phấn' trong từ ghép) làm loại oan triệu chứng thị chẩn thật.
    _EXTERNAL_CAUSES = (r'\b(?:ánh sáng|ánh đèn|đèn|bóng đổ|bóng tối|trang điểm|son môi|son|phấn|má hồng'
                        r'|máy ảnh|camera|góc chụp|chất lượng ảnh|chất lượng hình'
                        r'|lighting|light|lamp|shadow|makeup|lipstick|blush|image quality|photo)\b')
    _FEATURE_TOKENS = ("vàng", "đỏ", "nhợt", "nhạt", "trắng", "xanh", "tím", "đen", "sạm", "thâm",
                       "sưng", "phù", "tối", "yellow", "red", "pale", "dark", "swollen", "puffy")
    # Nhóm tương đương VI<->EN: mô tả LLaVA gốc thường là tiếng Anh nhưng tên triệu chứng là
    # tiếng Việt ('mặt vàng' <- 'yellow tint due to lighting') — cần soi bằng chứng chéo ngôn ngữ.
    _FEATURE_GROUPS = (
        ("vàng", "yellow"),
        ("đỏ", "red"),
        ("nhợt", "nhạt", "trắng", "pale", "white"),
        ("xanh", "green"),
        ("tím", "purple"),
        ("đen", "sạm", "thâm", "tối", "dark", "black"),
        ("sưng", "phù", "swollen", "puffy"),
    )

    def _filter_external_attributions(self, mapped: list, description: str) -> list:
        """[CHỐNG VỌNG CHẨN NHIỄU NGOẠI CẢNH] Bỏ triệu chứng vision mà chính mô tả đã quy cho
        nguyên nhân KHÔNG bệnh lý. Vd LLaVA: 'sắc mặt nhợt nhạt với một chút vàng DO ÁNH SÁNG
        trong nhà' -> cấm lấy 'mặt vàng' (màu vàng là của đèn, không phải của bệnh nhân).
        Cách làm: token màu/đặc điểm mà MỌI lần xuất hiện trong mô tả đều đứng trước
        'do|vì|bởi|due to...' + nguyên nhân ngoại cảnh (trong cùng mệnh đề, không băng qua
        token đặc điểm khác) bị coi là 'nhiễm ngoại cảnh' -> loại triệu chứng chứa token đó."""
        if not mapped or not description:
            return mapped
        desc = description.lower()
        alt = "|".join(self._FEATURE_TOKENS)
        tainted = set()
        for tok in self._FEATURE_TOKENS:
            occs = {m.start() for m in re.finditer(re.escape(tok), desc)}
            if not occs:
                continue
            pat = re.compile(
                re.escape(tok)
                + r'(?:(?!(?:' + alt + r'))[^,.;]){0,45}?'
                + r'\b(?:do|vì|bởi|due to|because of|caused by|from)\b[^,.;]{0,40}?'
                + self._EXTERNAL_CAUSES
            )
            bad = {m.start() for m in pat.finditer(desc)}
            if bad and occs <= bad:  # mọi lần xuất hiện đều bị quy ngoại cảnh
                tainted.add(tok)
        if not tainted:
            return mapped
        kept = []
        for s in mapped:
            sl = s.lower()
            # Bằng chứng của triệu chứng = mọi token trong NHÓM tương đương (VI+EN) có mặt
            # trong mô tả. Chỉ loại khi TOÀN BỘ bằng chứng đều bị quy ngoại cảnh.
            present = []
            for grp in self._FEATURE_GROUPS:
                if any(t in sl for t in grp):
                    present.extend(t for t in grp if t in desc)
            present = present or [t for t in self._FEATURE_TOKENS if t in sl and t in desc]
            if present and all(t in tainted for t in present):
                logger.info(f"[VISION-FILTER] Bỏ '{s}' — mô tả quy đặc điểm cho nguyên nhân ngoại cảnh (ánh sáng/trang điểm/máy ảnh)")
                continue
            kept.append(s)
        return kept

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
                            # Cột tách mới (nếu CSV đã migrate) — dùng cho hiển thị & khớp riêng dấu
                            # lưỡi; matching bệnh danh vẫn dựa trên 'triệu_chứng' (giữ nguyên hành vi).
                            "chu_chung": row.get("chủ_chứng", "").strip(),
                            "luoi": row.get("lưỡi", "").strip(),
                            "mach": row.get("mạch", "").strip(),
                            "bai_thuoc": row.get("bài_thuốc", "").strip(),
                            "vi_thuoc": row.get("vị_thuốc", "").strip(),
                        })
                logger.info(f"Đã tải {len(self.csv_rows)} dòng dữ liệu từ file CSV ({csv_path}) để so khớp bệnh lý")
            else:
                logger.warning("Không tìm thấy file CSV dữ liệu bệnh lý (đã thử TCM_CSV_PATH và data/Medicine_clean.csv).")
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu CSV: {e}")
        # Precompute document-frequency của từng triệu chứng (số bệnh chứa nó) -> IDF cho khớp bệnh.
        self._build_symptom_idf()

    def _build_symptom_idf(self):
        """DF = số BỆNH khác nhau chứa triệu chứng đó; IDF = log(N/(1+df)). Triệu chứng phổ biến
        (mệt mỏi, chóng mặt, táo bón...) có IDF THẤP, triệu chứng đặc hiệu/hiếm có IDF CAO."""
        from collections import Counter
        df = Counter()
        for r in getattr(self, "csv_rows", []) or []:
            flds = {s.strip().lower() for s in r.get("triệu_chứng", "").split(",") if s.strip()}
            for f in flds:
                df[f] += 1
        self._symptom_df = df
        self._N_dis = len(getattr(self, "csv_rows", []) or []) or 1

    def _symptom_idf(self, field_lower: str) -> float:
        """IDF của 1 triệu chứng (0 nếu chưa build). Field lạ (không có trong corpus) -> IDF cao nhất."""
        import math
        df = getattr(self, "_symptom_df", None)
        if not df:
            return 0.0
        return math.log(self._N_dis / (1 + df.get(field_lower, 0)))

    # ------ Khớp CHỦ_CHỨNG (chủ chứng cardinal) — bù lỗ hổng bệnh 'bị chôn' ------
    # Cột chủ_chứng (tách riêng, xem [[data-driven-gates-and-test-harness]]) là các CHỦ CHỨNG định
    # danh bệnh nhưng _find_matching_diseases trước chỉ chấm triệu_chứng -> khai đúng chủ chứng
    # (vd 'liệt dương') vẫn trượt ngưỡng ratio (105/147 bệnh bị chôn). Nay khớp chủ_chứng ĐA-TỪ
    # (>=2 âm tiết, tránh đơn-âm 'kinh'/'ho'/'da' khớp bừa) và nạp bệnh làm ứng viên với ratio theo
    # ĐỘ ĐẶC HIỆU (IDF chủ chứng): 'liệt dương' (ít bệnh) ratio cao -> vượt match triệu chung chung.
    _CC_FLOOR = 0.50
    _CC_CEIL = 0.80

    def _build_chu_chung_index(self):
        from collections import defaultdict
        import math
        rows = getattr(self, "csv_rows", []) or []
        # Văn bản gộp (triệu_chứng + chủ_chứng) THEO BỆNH để đo ĐỘ ĐẶC HIỆU THẬT của chủ chứng:
        # df = số BỆNH mà keyword xuất hiện Ở BẤT KỲ ĐÂU (không chỉ cột chủ_chứng). 'ăn kém'/'mệt
        # mỏi' là chủ_chứng của 1 bệnh nhưng có mặt ở TRIỆU CHỨNG rất nhiều bệnh -> df cao -> ratio
        # THẤP (đúng: chúng generic). 'liệt dương' chỉ ~3 bệnh -> df thấp -> ratio cao (cardinal thật).
        dis_text = defaultdict(str)
        cc_keywords = set()
        for r in rows:
            b = r.get("benh_ly", "").strip().lower()
            if not b:
                continue
            dis_text[b] += " " + r.get("triệu_chứng", "").lower() + " " + r.get("chu_chung", "").lower()
            for k in r.get("chu_chung", "").split(","):
                k = k.strip().lower()
                if k and len(k.split()) >= 2:
                    cc_keywords.add(k)
        n = len(dis_text) or 1
        cc_df = {}
        for k in cc_keywords:
            cc_df[k] = sum(1 for b in dis_text if k in dis_text[b]) or 1
        self._cc_df = cc_df
        self._cc_ndis = n
        self._cc_idf_max = max((math.log(n / (1 + df)) for df in cc_df.values()), default=1.0) or 1.0

    def _cc_ratio(self, keyword: str) -> float:
        """Ratio nạp ứng viên theo độ đặc hiệu của chủ chứng khớp (0 nếu không phải chủ chứng đa-từ)."""
        import math
        cc_df = getattr(self, "_cc_df", None)
        if cc_df is None:
            self._build_chu_chung_index()
            cc_df = self._cc_df
        df = cc_df.get(keyword)
        if not df:
            return 0.0
        idf = math.log(self._cc_ndis / (1 + df))
        frac = min(1.0, idf / self._cc_idf_max) if self._cc_idf_max else 0.0
        return self._CC_FLOOR + (self._CC_CEIL - self._CC_FLOOR) * frac

    def _cc_match_ratio(self, row, patient_symptoms_lower, raw_text_lower):
        """Ratio chủ_chứng cao nhất khớp cho row (0 nếu không khớp). PHẠM VI: CHỈ nạp chủ chứng
        cardinal ĐẶC THÙ GIỚI (nam/phụ khoa — liệt dương/di tinh/kinh nguyệt/thống kinh...). Đây là
        các dấu định danh MẠNH (khớp -> gần như chắc chắn đúng họ bệnh) nên nạp an toàn KHÔNG hồi quy
        gold (đã đo: 78.6% == baseline). Chủ chứng chung chung (ợ chua/biếng ăn) KHÔNG nạp vì chúng là
        triệu chứng chia sẻ, dễ soán ngôi match triệu_chứng thật (đã đo Option A tụt 75%). Mở rộng sang
        chủ chứng cardinal khác -> thêm vào _cc_admit_set (curated) sau khi đo hồi quy."""
        cc_str = row.get("chu_chung", "")
        if not cc_str:
            return 0.0, None
        admit = getattr(self, "_cc_admit_set", None)
        if admit is None:
            admit = set(self._SEX_MARK_MALE) | set(self._SEX_MARK_FEMALE)
            self._cc_admit_set = admit
        best, hit = 0.0, None
        for k in cc_str.split(","):
            k = k.strip().lower()
            if len(k.split()) < 2 or k not in admit:
                continue
            present = (raw_text_lower and k in raw_text_lower) or \
                any(k == s or k in s or s in k for s in patient_symptoms_lower)
            if present:
                r = self._cc_ratio(k)
                if r > best:
                    best, hit = r, k
        return best, hit

    def _get_disease_gates(self) -> list:
        """Cổng an toàn bệnh danh nạp từ data/disease_gates.json (data-hóa từ luật hard-code cũ).
        Cache trên instance. Rỗng nếu thiếu file -> _validate_disease_safety cho qua tất (an toàn:
        không chặn oan, chỉ mất lớp lọc). Xây bởi scripts/build_disease_gates.py."""
        cached = getattr(self, "_disease_gates", None)
        if cached is not None:
            return cached
        import json
        import os
        gates = []
        path = os.getenv("TCM_DISEASE_GATES_PATH", "data/disease_gates.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    gates = json.load(f).get("gates", [])
                logger.info(f"Đã nạp {len(gates)} cổng an toàn bệnh danh từ {path}")
            else:
                logger.warning(f"Không thấy {path} — bỏ qua lớp lọc cổng bệnh danh.")
        except Exception as e:
            logger.error(f"Lỗi nạp cổng bệnh danh: {e}")
            gates = []
        self._disease_gates = gates
        return gates

    def _get_disease_sex(self) -> dict:
        """Nạp data/disease_sex.json -> {tên_bệnh_chuẩn_hoá: 'nam'|'nu'}. Cache trên instance.
        Rỗng nếu thiếu file (an toàn: không lọc oan). Bệnh không có trong map = cả 2 giới."""
        cached = getattr(self, "_disease_sex_map", None)
        if cached is not None:
            return cached
        import json
        import os
        m = {}
        path = os.getenv("TCM_DISEASE_SEX_PATH", "data/disease_sex.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                for _sx in ("nam", "nu"):
                    for _name in d.get(_sx, []):
                        m[self._norm_disease_name(_name)] = _sx
                logger.info(f"Đã nạp cổng giới tính: {len(m)} bệnh đặc thù giới từ {path}")
            else:
                logger.warning(f"Không thấy {path} — bỏ qua cổng giới tính.")
        except Exception as e:
            logger.error(f"Lỗi nạp cổng giới tính: {e}")
            m = {}
        self._disease_sex_map = m
        return m

    @staticmethod
    def _norm_disease_name(name: str) -> str:
        return re.sub(r"\s+", " ", (name or "").strip().lower())

    @staticmethod
    def _norm_sex(sex) -> str:
        """Chuẩn hoá giá trị giới về 'nam'|'nu'|None."""
        s = str(sex or "").strip().lower()
        if s in ("nam", "male", "m", "nam giới", "nam gioi", "trai"):
            return "nam"
        if s in ("nu", "nữ", "female", "f", "nữ giới", "nu gioi", "nữ giơi", "gái"):
            return "nu"
        return None

    # Dấu giới ĐẶC HIỆU (đa-từ/giải phẫu, tránh mơ hồ) để suy giới từ TRIỆU CHỨNG khi không khai form.
    _SEX_MARK_MALE = ("liệt dương", "dương nuy", "di tinh", "mộng tinh", "tảo tiết", "xuất tinh",
                      "hoạt tinh", "dương vật", "tinh trùng", "rối loạn cương", "cường dương",
                      "bìu", "tinh hoàn", "dịch hoàn")
    _SEX_MARK_FEMALE = ("kinh nguyệt", "thống kinh", "bế kinh", "rong kinh", "băng huyết", "đới hạ",
                        "huyết trắng", "âm đạo", "âm hộ", "tử cung", "mang thai", "có thai",
                        "thai nghén", "sản hậu", "hành kinh", "kinh trước kỳ", "kinh sau kỳ",
                        "buồng trứng", "tắc tia sữa")

    @classmethod
    def _infer_sex(cls, text: str):
        """Suy giới từ text lời khai: (1) cụm form 'nam giới'/'nữ giới' (ưu tiên), (2) dấu giới ĐẶC
        HIỆU trong triệu chứng ('liệt dương'->nam, 'thống kinh'->nữ). Nếu có dấu CẢ HAI giới (mơ hồ,
        vd nhắc người khác) -> None. Trả 'nam'|'nu'|None."""
        t = (text or "").lower()
        if "nữ giới" in t:
            return "nu"
        if "nam giới" in t:
            return "nam"
        male = any(k in t for k in cls._SEX_MARK_MALE)
        female = any(k in t for k in cls._SEX_MARK_FEMALE)
        if male and not female:
            return "nam"
        if female and not male:
            return "nu"
        return None

    def _sex_symptom_conflict(self, raw_text: str):
        """Giới KHAI BÁO (form, self._patient_sex) MÂU THUẪN với dấu đặc thù giới KHÁC trong lời khai
        (vd khai Nam nhưng có 'âm hộ'/'kinh nguyệt', hoặc khai Nữ nhưng 'liệt dương'/'dương vật').
        Trả danh sách dấu mâu thuẫn (để cảnh báo nhập liệu), None nếu không mâu thuẫn / không khai giới."""
        sex = getattr(self, "_patient_sex", None)
        if not sex or not raw_text:
            return None
        t = raw_text.lower()
        opp = self._SEX_MARK_FEMALE if sex == "nam" else self._SEX_MARK_MALE
        found = [k for k in opp if k in t]
        return found or None

    def _sex_conflict(self, disease_name: str, patient_sex: str) -> bool:
        """True khi bệnh thuộc giới KHÁC bệnh nhân đã khai -> phải LOẠI. patient_sex rỗng ->
        không lọc (False). Bệnh không đặc thù giới -> không lọc (False)."""
        if not patient_sex:
            return False
        ds = self._get_disease_sex().get(self._norm_disease_name(disease_name))
        return ds is not None and ds != patient_sex

    # [CỔNG TRẠNG THÁI SINH SẢN] Bệnh THAI SẢN / HẬU SẢN chỉ xảy ra khi ĐANG MANG THAI hoặc MỚI SINH.
    # Tên tự chỉ điểm nên khớp theo keyword (bền hơn liệt kê từng bệnh; bắt cả bệnh thêm sau này).
    _PREG_POSTPARTUM_DISEASE_KWS = (
        "động thai", "lưu sản", "tiểu sản", "hoạt thai", "quỷ thai", "thai phù", "thai lậu",
        "sản hậu", "hậu sản", "nhâm thần", "tử giản", "sản giật", "ố trở", "dọa sảy", "sảy thai")
    # Dấu ĐANG HÀNH KINH (đang có kinh -> chắc chắn KHÔNG mang thai). CỐ Ý KHÔNG gồm 'bế kinh/chậm
    # kinh/trễ kinh' — tắt/trễ kinh có thể là DẤU MANG THAI SỚM, không đủ khẳng định không mang thai.
    _MENSTRUATION_MARKS = (
        "kinh nguyệt", "hành kinh", "thống kinh", "rong kinh", "chu kỳ kinh",
        "kinh trước kỳ", "kinh sau kỳ", "kinh ra", "ra kinh")
    # Dấu ĐANG MANG THAI / HẬU SẢN -> TẮT cổng (giữ bệnh thai sản).
    _PREGNANCY_MARKS = (
        "có thai", "mang thai", "có bầu", "mang bầu", "thai nghén", "thai kỳ", "đang nghén",
        "ốm nghén", "mới sinh", "sau sinh", "mới đẻ", "sau đẻ", "hậu sản", "cho con bú", "sinh con")

    def _reproductive_state_conflict(self, disease_name: str, raw_user_text: str,
                                     patient_symptoms=None) -> bool:
        """True khi bệnh THAI SẢN/HẬU SẢN nhưng lời khai cho thấy đang HÀNH KINH + KHÔNG dấu mang
        thai/hậu sản -> phải LOẠI (đang có kinh thì không thể mang thai; 'Động thai/Lưu sản' vô lý,
        'Sản hậu ...' lạc bối cảnh). Bệnh không phải thai sản -> False. Xem _PREG_POSTPARTUM_DISEASE_KWS."""
        dl = (disease_name or "").lower()
        if not any(k in dl for k in self._PREG_POSTPARTUM_DISEASE_KWS):
            return False
        t = (raw_user_text or "").lower()
        if patient_symptoms:
            t += " " + " ".join(patient_symptoms).lower()
        is_menstruating = any(k in t for k in self._MENSTRUATION_MARKS)
        is_preg_postpartum = any(k in t for k in self._PREGNANCY_MARKS)
        return is_menstruating and not is_preg_postpartum

    # Dấu vọng/thiết + tên bệnh lọt nhãn triệu chứng — KHÔNG hỏi lại trong vấn chẩn chuyên sâu (người
    # bệnh không tự quan sát lưỡi/mạch). Đồng bộ với bộ lọc /api/related-symptoms.
    _DEEP_INQUIRY_EXCLUDE = ("lưỡi", "rêu", "mạch", "sắc mặt", "sắc da", "gò má", "má đỏ", "mặt đỏ",
                             "mặt nhợt", "mặt vàng", "mặt trắng", "vết răng", "hằn răng", "ung thư",
                             "u não", "khối u", "hội chứng", "bệnh ")

    # PHÁP TRỊ / công năng bài thuốc (dưỡng âm/bổ thận/hoạt huyết hóa ứ...) — data CSV/graph đôi khi
    # nhét lẫn vào field triệu_chứng. Đây là việc THẦY THUỐC LÀM, KHÔNG phải triệu chứng người bệnh
    # -> CẤM hiển thị như 'triệu chứng' ở vấn chẩn chuyên sâu / gợi ý liên quan. Nhận diện = cụm BẮT
    # ĐẦU bằng động từ pháp-trị (bigram đặc trưng, không mở đầu triệu chứng thật -> tránh bắt nhầm).
    _TREATMENT_PRINCIPLE_PREFIXES = (
        "dưỡng ", "bổ ", "tư âm", "tư bổ", "tư dưỡng", "thanh nhiệt", "thanh can", "thanh phế",
        "thanh tâm", "thanh vị", "thanh thấp", "ôn dương", "ôn trung", "ôn kinh", "ôn bổ", "ôn thận",
        "lương huyết", "hoạt huyết", "hóa ứ", "hóa đàm", "hóa thấp", "kiện tỳ", "kiện vận", "sơ can",
        "sơ phong", "ích khí", "ích âm", "ích thận", "cố thận", "cố tinh", "cố sáp", "liễm hãn",
        "sáp tinh", "thăng dương", "giáng nghịch", "giáng hỏa", "bình can", "tiềm dương", "trấn kinh",
        "trấn tâm", "an thần", "định thần", "khai khiếu", "sinh tân", "nhuận táo", "nhuận tràng",
        "táo thấp", "khu phong", "tán hàn", "tán ứ", "trừ thấp", "trừ đàm", "trừ phong", "lợi thủy",
        "lợi niệu", "lợi tiểu", "lý khí", "hành khí", "chỉ thống", "chỉ khái", "chỉ huyết", "chỉ tả",
        "giải biểu", "giải độc", "giải cơ", "tả hỏa", "hòa giải", "điều hòa", "bổ huyết", "bổ khí",
        "bổ trung", "bổ âm", "bổ dương", "trừ hàn", "khứ hàn", "tuyên phế", "túc giáng", "nạp khí",
    )

    @classmethod
    def _is_treatment_principle(cls, phrase: str) -> bool:
        """True nếu cụm là PHÁP TRỊ (dưỡng âm/bổ thận/hoạt huyết hóa ứ...) chứ KHÔNG phải triệu chứng.
        Bắt đầu bằng động từ pháp-trị -> loại khỏi mọi nơi hiển thị 'triệu chứng' cho người bệnh."""
        return (phrase or "").strip().lower().startswith(cls._TREATMENT_PRINCIPLE_PREFIXES)

    # Hậu tố PHƯƠNG DANH: tên bài có các từ này là PHƯƠNG DANH THẬT (dù mở đầu bằng động từ pháp-trị,
    # vd 'Ôn thận nạp khí phương', 'Dưỡng vị thang', 'Bổ Trung Ích Khí gia giảm') -> KHÔNG coi là 'trần'.
    _FORMULA_TYPE_WORD = re.compile(
        r'\b(thang|hoàn|hoàng|tán|tan|ẩm|đơn|đan|cao|tễ|phương|dịch|hợp|trà|cốm|thần)\b|gia giảm',
        re.IGNORECASE)

    @classmethod
    def _is_bare_treatment_principle(cls, name: str) -> bool:
        """True nếu TÊN BÀI thực chất chỉ là PHÁP TRỊ TRẦN (mở đầu động từ pháp-trị + KHÔNG có hậu tố
        phương/thang/hoàn/tán/ẩm/đơn/gia giảm...). Dùng để đổi nhãn Mục 5 'Dùng bài <X>' -> 'Dùng bài
        THEO PHÁP <X>' cho dễ hiểu (X là công năng, không phải phương danh). 'Ôn thận nạp khí phương',
        'Bổ Trung Ích Khí gia giảm' -> False (giữ 'Dùng bài')."""
        return cls._is_treatment_principle(name) and not cls._FORMULA_TYPE_WORD.search(name or "")

    def _relabel_bare_phaptri_bai(self, markdown: str) -> str:
        """Đổi 'Dùng bài **<pháp trị trần>**' -> 'Dùng bài theo pháp **<...>**' trên toàn Mục 5 (một
        chỗ, phủ mọi call-site). Chỉ đụng tên bài là pháp-trị TRẦN; phương danh thật giữ nguyên."""
        def _repl(m):
            bai = m.group(1)
            return f"Dùng bài theo pháp **{bai}**" if self._is_bare_treatment_principle(bai) else m.group(0)
        return re.sub(r'Dùng bài \*\*([^*]+)\*\*', _repl, markdown or "")

    def _compute_deep_inquiry(self, user_symptoms: str, patient_terms: list,
                              all_symptoms_list: list) -> dict:
        """[VẤN CHẨN CHUYÊN SÂU] Sau khi ĐÃ có chẩn đoán trực tiếp (KHÔNG chặn), gợi ý các yếu tố
        ĐÁNG HỎI để tinh chỉnh: (1) BỐI CẢNH quyết định (mang thai / giới) — làm ĐỔI bệnh danh/bài
        thuốc; (2) triệu chứng ĐẶC HIỆU của bệnh đang nghi mà người dùng CHƯA khai. Chỉ hiện khi
        thật sự đáng hỏi (có nhánh bối cảnh, hoặc nhiều bệnh hòa nhau, hoặc lời khai còn mỏng).
        Trả {show, factors:[{type,label,add,note}]}. Mỗi factor.add = cụm sẽ GỘP vào lời khai khi
        người dùng chọn -> hệ tự phân tích lại."""
        try:
            raw = (user_symptoms or "").lower()
            terms = [t for t in (patient_terms or []) if t and t.strip()]
            matched = self._find_matching_diseases(terms, raw_user_text=user_symptoms)
            if not matched:
                return {"show": False, "factors": []}
            gb = matched[0].get("ratio", 0.0)
            window = [m for m in matched if m.get("ratio", 0.0) >= gb - 0.15 and m.get("ratio", 0.0) >= 0.30]
            window_names = [m.get("benh_ly", "") for m in window]
            factors = []

            # --- (1a) BỐI CẢNH: MANG THAI — chỉ hỏi khi (i) trạng thái sinh sản CÒN MỞ (không dấu
            #     mang thai/hành kinh), (ii) không phải nam, VÀ (iii) chẩn đoán ĐANG XOAY quanh phụ
            #     khoa/sinh sản (cửa sổ top có bệnh kinh nguyệt/thai sản) HOẶC có dấu TRỄ/MẤT KINH
            #     (chỉ điểm mang thai kinh điển). KHÔNG dựa vào trùng triệu chứng CHUNG (sốt/đau mình
            #     của 'Sản hậu phát nhiệt') để tránh mời 'mang thai' cho ca hô hấp. Chọn 'đang mang
            #     thai' -> xét bệnh thai sản + tránh vị chống chỉ định thai kỳ; 'đang hành kinh' ->
            #     loại bệnh thai sản (xem _reproductive_state_conflict).
            sex = getattr(self, "_patient_sex", None) or self._infer_sex(user_symptoms)
            has_preg = any(k in raw for k in self._PREGNANCY_MARKS)
            has_mens = any(k in raw for k in self._MENSTRUATION_MARKS)
            _repro_dz_kw = ("kinh nguyệt", "nguyệt kinh", "thống kinh", "bế kinh", "băng lậu",
                            "canh niên", "tử cung", "buồng trứng", "đới hạ", "sản", "thai", "nhâm thần")
            repro_window = any(
                any(k in (m.get("benh_ly", "") or "").lower() for k in _repro_dz_kw)
                for m in window[:3])
            missed_period = any(k in raw for k in ("chậm kinh", "trễ kinh", "mất kinh", "tắt kinh"))
            if sex != "nam" and not has_preg and not has_mens and (repro_window or missed_period):
                factors.append({"type": "state", "key": "pregnancy", "label": "Đang mang thai",
                                "add": "đang mang thai",
                                "note": "Xét các bệnh thai sản (động thai...) và tránh vị thuốc chống chỉ định thai kỳ."})
                factors.append({"type": "state", "key": "menstruating",
                                "label": "Đang trong kỳ kinh (không mang thai)",
                                "add": "đang hành kinh",
                                "note": "Loại các bệnh thai sản không phù hợp."})

            # --- (1b) BỐI CẢNH: GIỚI — chỉ khi CHƯA rõ giới VÀ cửa sổ có CẢ bệnh nam LẪN nữ (giới
            #     thật sự phân biệt ứng viên). Nếu chỉ toàn bệnh một giới thì giới đã ngầm định -> bỏ.
            if not sex:
                dz_sex = self._get_disease_sex()
                sset = {dz_sex.get(self._norm_disease_name(n)) for n in window_names}
                if "nam" in sset and "nu" in sset:
                    factors.append({"type": "state", "key": "sex_nu", "label": "Nữ giới",
                                    "add": "nữ giới", "note": ""})
                    factors.append({"type": "state", "key": "sex_nam", "label": "Nam giới",
                                    "add": "nam giới", "note": ""})

            # --- (2) TRIỆU CHỨNG ĐẶC HIỆU của bệnh nghi (chưa khai) — gom từ CSV các bệnh trong cửa
            #     sổ, loại đã khai + vọng/thiết + generic, xếp theo IDF (đặc hiệu lên trước).
            input_join = (" ".join(terms) + " " + raw).lower()
            sym_idf, sym_disp = {}, {}
            for m in window[:4]:
                bl = m.get("benh_ly", "")
                for row in self.csv_rows:
                    if row.get("benh_ly") != bl:
                        continue
                    for s in (row.get("triệu_chứng", "") or "").split(","):
                        s = s.strip()
                        sl = s.lower()
                        if not sl or len(s) > 28:
                            continue
                        if any(k in sl for k in self._DEEP_INQUIRY_EXCLUDE) or self._is_pulse_field(sl):
                            continue
                        if self._is_treatment_principle(s) or self._is_generic_symptom(sl):
                            continue
                        if sl in input_join or any(sl in t or t in sl for t in terms):
                            continue
                        if sl not in sym_idf:
                            sym_idf[sl] = self._symptom_idf(sl)
                            sym_disp[sl] = s
            ranked = sorted(sym_idf.keys(), key=lambda k: sym_idf[k], reverse=True)
            kept_syms = []
            for k in ranked:
                if any(len(o.split()) >= 2 and o in k and o != k for o in kept_syms):
                    continue  # bỏ cụm dài trùng bộ phận cụm ngắn đã giữ
                kept_syms.append(k)
                factors.append({"type": "symptom", "key": k, "label": sym_disp[k],
                                "add": sym_disp[k], "note": ""})
                if len(kept_syms) >= 6:
                    break

            has_state = any(f["type"] == "state" for f in factors)
            ambiguous = len(window) >= 2
            thin = len(all_symptoms_list or []) <= 3
            show = bool(factors) and (has_state or ambiguous or thin)
            return {"show": show, "factors": factors[:9]}
        except Exception:
            logger.exception("Lỗi tính vấn chẩn chuyên sâu")
            return {"show": False, "factors": []}

    def _validate_disease_safety(self, disease_name: str, patient_symptoms: list, raw_user_text: str) -> bool:
        """Bộ lọc an toàn lâm sàng (DATA-HÓA): loại bệnh danh chuyên khoa nếu lời khai không có triệu
        chứng chỉ điểm tương ứng. Luật đọc từ data/disease_gates.json (thay ~460 dòng if/else cũ —
        xem _validate_disease_safety_legacy để đối chiếu). Ngữ nghĩa giữ nguyên: bệnh phải qua HẾT
        mọi cổng áp dụng; cổng 'requires' rỗng = luôn loại (bệnh Tây y thuần)."""
        disease_lower = disease_name.lower()
        # Che phủ định trên lời khai thô (đồng nhất với bản legacy): 'không đau đầu' không được rò
        # 'đau đầu' cho qua bệnh Thiên đầu thống. patient_symptoms đã lọc phủ định từ trước.
        _raw_l = raw_user_text.lower() if raw_user_text else ""
        if _raw_l:
            _raw_l = self._mask_negated_text(_raw_l, self._get_neg_protected_phrases())
        symptoms_str = " ".join(patient_symptoms).lower() + " " + _raw_l

        for gate in self._get_disease_gates():
            named = any(n in disease_lower for n in gate.get("names", ()))
            if not named:
                for tok in gate.get("names_regex", ()):
                    if re.search(r'\b' + re.escape(tok) + r'\b', disease_lower):
                        named = True
                        break
            if not named:
                continue
            reqs = gate.get("requires", ())
            if gate.get("kw_wb"):
                hit = any(re.search(r'\b' + re.escape(kw) + r'\b', symptoms_str) for kw in reqs)
            else:
                hit = any(kw in symptoms_str for kw in reqs)
            if not hit:
                return False
        return True


    # Ngưỡng IDF cho khớp bệnh danh — tinh chỉnh qua scripts/eval_disease_matching.py trên toàn 992 bệnh.
    # peak: bệnh chỉ được đặt tên nếu khớp ÍT NHẤT 1 triệu chứng đủ ĐẶC HIỆU (idf>=4.0 ~ xuất hiện ở
    # <=~18 bệnh). Đo được: chặn thêm over-match generic MÀ KHÔNG giảm recall (self-recall 86.4%) và
    # KHÔNG phá ca lâm sàng (4/4). Sum-IDF (_idf_min) hiệu quả kém hơn -> để tắt (0.0).
    _idf_peak_min = 4.0
    _idf_min = 0.0

    # Triệu chứng GENERIC (thể trạng chung + màu lưỡi/rêu/mạch cơ bản) — gần như hội chứng/bệnh nào
    # cũng có, KHÔNG đủ đặc hiệu để ĐẶT TÊN một bệnh danh. Bệnh chỉ khớp toàn các triệu chứng này bị
    # coi là khớp giả (vd "Áp xe gan" khớp 'mệt mỏi'+'lưỡi hồng', "Dương nuy" khớp 'mặt vàng'+'rêu trắng').
    _GENERIC_MATCH_KEYWORDS = (
        "mệt mỏi", "mệt", "uể oải", "vô lực", "không có sức", "đuối sức", "mỏi mệt",
        "mặt nhợt", "sắc mặt nhợt", "da nhợt", "mặt trắng nhợt", "mặt xanh", "da xanh",
        "mặt vàng", "da vàng", "sắc mặt vàng", "sắc mặt kém", "vàng xạm", "vàng sạm",
        "lưỡi hồng", "lưỡi nhợt", "chất lưỡi nhợt", "chất lưỡi hồng",
        "rêu lưỡi trắng", "rêu trắng", "rêu lưỡi mỏng", "rêu mỏng", "rêu lưỡi ít",
        "mạch tế", "mạch nhược", "mạch trầm", "mạch hư",
    )

    @classmethod
    def _is_generic_symptom(cls, ds_lower: str) -> bool:
        """True nếu triệu chứng chỉ là dấu thể trạng/lưỡi/mạch chung chung (không đặc hiệu bệnh)."""
        return any(kw in ds_lower for kw in cls._GENERIC_MATCH_KEYWORDS)

    # Field THIẾT CHẨN (mạch tượng) — hệ chỉ có Vấn + Vọng (lưỡi/mặt), KHÔNG bắt mạch, nên
    # field mạch không bao giờ quan sát được và phải bị LOẠI KHỎI MẪU SỐ khi tính tỷ lệ khớp
    # (nếu để trong mẫu số, thể bệnh mô tả mạch càng kỹ càng bị phạt ratio oan). Chỉ bắt field
    # BẮT ĐẦU bằng 'mạch' + phẩm chất mạch (mạch trầm/tế/sác/hoạt...), KHÔNG bắt 'mạch máu',
    # 'mạch lạc' (huyết quản/kinh mạch) hay field mở đầu bằng lưỡi/rêu rồi mới nhắc mạch (những
    # field 'mash' đó vẫn còn bằng chứng lưỡi quan sát được nên giữ lại để khớp phần lưỡi).
    _PULSE_FIELD_RE = re.compile(r'^mạch\s+(?!máu|lạc)', re.IGNORECASE)

    @classmethod
    def _is_pulse_field(cls, ds_lower: str) -> bool:
        """True nếu field CSV là mạch tượng thuần (thiết chẩn) — hệ không có kênh bắt mạch."""
        return bool(cls._PULSE_FIELD_RE.match((ds_lower or "").strip()))

    def _find_matching_diseases(self, patient_symptoms: list, raw_user_text: str = "") -> list:
        """Tìm các bệnh lý dựa trên tỷ lệ so khớp triệu chứng (mềm dẻo hơn thay vì bắt buộc 100%)"""
        if not hasattr(self, 'csv_rows') or not self.csv_rows:
            return []

        patient_symptoms_lower = [s.lower().strip() for s in patient_symptoms] if patient_symptoms else []
        raw_text_lower = raw_user_text.lower() if raw_user_text else ""
        # [CỔNG GIỚI TÍNH] Giới lấy từ tham số cấu trúc (self._patient_sex, do run_diagnosis đặt từ
        # form) hoặc suy từ text ('nam giới'/'nữ giới' đã ghép bởi compose_interview_text). Nếu biết
        # giới -> loại thẳng bệnh khác giới (phụ khoa cho nam, nam khoa cho nữ) bất kể triệu chứng.
        _patient_sex = getattr(self, "_patient_sex", None) or self._infer_sex(raw_user_text) \
            or self._infer_sex(" ".join(patient_symptoms_lower))
        # [CHỐNG PHỦ ĐỊNH KHỚP MỀM] Che vùng bị phủ định trên lời khai TRƯỚC khi tách phân đoạn, để
        # field CSV không soft-match nhầm triệu chứng người bệnh đã phủ nhận ('không sốt' -> field
        # 'sốt cao' không được khớp). Bảo vệ các cụm CSV vốn chứa 'không' ('tay chân không ấm'...)
        # để không cắt đứt chính chúng — dùng vocab triệu chứng CSV làm danh sách bảo vệ.
        if raw_text_lower:
            _protected = self._get_neg_protected_phrases()
            raw_text_lower = self._mask_negated_text(raw_text_lower, _protected)
        # Cách 2 khớp theo TỪNG PHÂN ĐOẠN (tách phẩy) của lời khai, không phải toàn chuỗi:
        # all-words-anywhere cho phép field 'đoản hơi' khớp nhờ ghép 'đoản' (của 'đoản khí') với
        # 'hơi' (của 'hắt hơi') — hai triệu chứng KHÁC NHAU — kéo bệnh danh lạc đề ('Bắp chân
        # xung đau' cho ca hô hấp). Từ của một field phải cùng nằm trong MỘT phân đoạn.
        raw_segments = [seg.strip() for seg in raw_text_lower.split(",") if seg.strip()] if raw_text_lower else []

        matched_candidates = []
        for row in self.csv_rows:
            db_symptoms_str = row.get("triệu_chứng", "")
            if not db_symptoms_str:
                continue
            db_symptoms = [s.strip() for s in db_symptoms_str.split(",") if s.strip()]
            if not db_symptoms:
                continue

            matched_count = 0
            specific_matched = 0   # số triệu chứng khớp KHÔNG phải generic (đặc hiệu)
            matched_idf = 0.0      # TỔNG IDF các triệu chứng khớp
            peak_idf = 0.0         # IDF CAO NHẤT trong các triệu chứng khớp (có ≥1 triệu chứng đặc hiệu?)
            observable_count = 0   # số field QUAN SÁT ĐƯỢC (loại mạch tượng) -> mẫu số tỷ lệ khớp
            matched_pulse = 0      # field mạch khớp được (chỉ khi người dùng TỰ gõ mạch vào lời khai)
            for ds in db_symptoms:
                ds_lower = ds.lower().strip()
                is_pulse = self._is_pulse_field(ds_lower)
                if not is_pulse:
                    observable_count += 1
                _hit = False
                # Cách 1: Khớp chính xác hoặc chứa trong tập triệu chứng đã chuẩn hóa
                if patient_symptoms_lower and any(ds_lower == ps or ds_lower in ps or ps in ds_lower for ps in patient_symptoms_lower):
                    _hit = True
                # Cách 2: Khớp mềm từ khóa trên lời khai — mọi từ của field phải cùng MỘT phân đoạn
                elif raw_segments:
                    ds_words = [w for w in ds_lower.split() if len(w) >= 2]
                    if ds_words and any(all(w in seg for w in ds_words) for seg in raw_segments):
                        _hit = True
                if _hit:
                    matched_count += 1
                    if is_pulse:
                        matched_pulse += 1
                    if not self._is_generic_symptom(ds_lower):
                        specific_matched += 1
                    _idf = self._symptom_idf(ds_lower)
                    matched_idf += _idf
                    if _idf > peak_idf:
                        peak_idf = _idf

            # Mẫu số = số field quan sát được (bỏ mạch). Field mạch CHỈ vào mẫu số nếu nó thực sự
            # khớp (người dùng có gõ mạch) — giữ tỷ lệ ≤ 1.0 và không phạt oan field mạch không khớp.
            denom = observable_count + matched_pulse
            if denom <= 0:
                denom = len(db_symptoms)   # phòng hờ: dòng toàn field mạch (không xảy ra trong CSV hiện tại)
            match_ratio = matched_count / denom

            # Ngưỡng chấp nhận: khớp >= 30% VÀ >= 2 triệu chứng VÀ có >= 1 triệu chứng ĐẶC HIỆU
            # VÀ tổng IDF khớp >= _idf_min. Ràng buộc "đặc hiệu"/IDF chặn bệnh danh vô lý bị gán chỉ vì
            # trùng triệu chứng thể trạng/lưỡi chung chung (vd hô hấp mà ra "Áp xe gan"/"Bạch tiển"):
            # triệu chứng phổ biến (mệt mỏi/chóng mặt) có IDF thấp nên khớp toàn chúng KHÔNG đủ ngưỡng,
            # còn triệu chứng đặc hiệu (IDF cao) chỉ cần ít vẫn qua. _idf_min tinh chỉnh qua eval harness.
            idf_min = getattr(self, "_idf_min", 0.0)
            peak_min = getattr(self, "_idf_peak_min", 0.0)
            _triage_ok = (match_ratio >= 0.30 and matched_count >= 2 and specific_matched >= 1
                          and matched_idf >= idf_min and peak_idf >= peak_min)
            # [KHỚP CHỦ_CHỨNG] Nạp bệnh có CHỦ CHỨNG (cardinal) khớp lời khai dù triệu_chứng chưa đủ
            # ngưỡng (bù 105/147 bệnh 'bị chôn' — nam khoa/phụ khoa chỉ tiếp cận được qua chủ chứng).
            _cc_ratio_v, _ = self._cc_match_ratio(row, patient_symptoms_lower, raw_text_lower)
            if _triage_ok or _cc_ratio_v > 0:
                # Áp dụng bộ lọc an toàn lâm sàng + cổng giới tính ngăn chẩn đoán sai lệch
                if self._validate_disease_safety(row["benh_ly"], patient_symptoms, raw_user_text) \
                        and not self._sex_conflict(row["benh_ly"], _patient_sex) \
                        and not self._reproductive_state_conflict(row["benh_ly"], raw_user_text, patient_symptoms):
                    matched_candidates.append({
                        "benh_ly": row["benh_ly"],
                        "hoi_chung": row["hoi_chung"],
                        "ratio": max(match_ratio, _cc_ratio_v),
                        "matched_count": max(matched_count, 1 if _cc_ratio_v > 0 else 0),
                        "matched_idf": round(matched_idf, 2)
                    })
        
        # Sắp xếp theo tỷ lệ khớp giảm dần, ưu tiên số lượng triệu chứng khớp nhiều hơn
        matched_candidates.sort(key=lambda x: (x["ratio"], x["matched_count"], x["matched_idf"]), reverse=True)

        # [FIX KHỬ TRÙNG BỆNH DANH] CSV có nhiều dòng/bệnh (mỗi dòng 1 hội chứng) nên một bệnh
        # khớp nhiều thể sẽ chiếm nhiều suất trong ứng viên (vd 'Ách nghịch' x5 chiếm cả top-3,
        # đẩy bệnh hợp lệ khác ra ngoài). Gộp về MỘT ứng viên/bệnh: đại diện = dòng khớp tốt nhất
        # (đã sort ở trên); 'hoi_chung' giữ hội chứng dòng đó (tương thích cũ), 'hoi_chung_all'
        # giữ ĐỦ hội chứng của MỌI dòng khớp — bước lọc chéo hội chứng phải soi hết danh sách này
        # để bệnh không bị loại oan chỉ vì thể tốt-nhất không trùng hội chứng cốt lõi.
        merged = {}
        for m in matched_candidates:
            key = m["benh_ly"].strip().lower()
            if key not in merged:
                m["hoi_chung_all"] = [m["hoi_chung"]]
                merged[key] = m
            elif m["hoi_chung"] not in merged[key]["hoi_chung_all"]:
                merged[key]["hoi_chung_all"].append(m["hoi_chung"])
        # [PHỔ HỘI CHỨNG ĐẦY ĐỦ] hoi_chung_all phải đại diện TOÀN BỘ các thể của bệnh trong CSV,
        # không chỉ các row vượt ngưỡng: bệnh đã qua ngưỡng nhờ MỘT thể là đủ tư cách ứng viên,
        # nhưng bước lọc chéo với hội chứng cốt lõi cần biết bệnh CÓ thể đúng hội chứng đó không.
        # Ca thật: 'Tỵ cứu' qua ngưỡng nhờ row Thận dương bất túc (0.31), row 'Phế khí hư hàn'
        # 12 trường giáo khoa chỉ đạt 0.25 — thiếu nó trong phổ, bệnh bị coi là không grounded
        # với cốt lõi 'Phế khí hư hàn' và rớt oan khỏi bệnh danh. KHÔNG hạ ngưỡng khớp (tránh
        # overmatch) — chỉ bổ sung metadata phổ thể cho bệnh ĐÃ vượt ngưỡng.
        if merged:
            _spectrum = {}
            for _r in (getattr(self, "csv_rows", None) or []):
                _k = _r.get("benh_ly", "").strip().lower()
                if _k in merged:
                    _hc = _r.get("hoi_chung", "").strip()
                    if _hc:
                        _spectrum.setdefault(_k, set()).add(_hc)
            for _k, _m in merged.items():
                for _hc in sorted(_spectrum.get(_k, ())):
                    if _hc not in _m["hoi_chung_all"]:
                        _m["hoi_chung_all"].append(_hc)
        return list(merged.values())

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
        # Khớp theo TOKEN (âm tiết) thay vì substring: mọi keyword nhóm đều là âm tiết đơn, còn
        # substring khiến 'âm' dính trong 'tâm/lâm' -> mọi hội chứng Tâm bị coi là liên quan mọi
        # hội chứng Âm hư (8.247 cặp giả trên KG+CSV thật), kéo sai bệnh danh và ánh xạ hội chứng.
        toks1 = set(re.findall(r'[^\W\d_]+', syn1_l))
        toks2 = set(re.findall(r'[^\W\d_]+', syn2_l))
        for group in concept_groups:
            if any(w in toks1 for w in group) and any(w in toks2 for w in group):
                return True
        return False

    def _kb_polarity_fallback_rows(self, diseases_lower, final_primary, final_concurrent,
                                   case_symptoms, printed_pairs):
        """[Mục 5 — FALLBACK THỂ KB CÙNG CỰC] Cốt lõi/kèm theo không trùng TÊN thể KB nào của
        bệnh danh đã chốt theo chiều NGƯỢC của thể-tổng-quát: nhãn biện chứng TỔNG QUÁT ('Khí
        hư') còn KB gán bài dưới thể ĐẶC HIỆU TẠNG ('Bĩ mãn × Tỳ vị hư nhược' — Hương sa lục
        quân) — token thể không là tập con của nhãn nên [FALLBACK THỂ TỔNG QUÁT] trượt và Mục 5
        trắng oan. Chọn hộ mỗi vai trò tối đa 1 dòng CSV nguyên gốc của đúng bệnh danh:
          Bản : CHỈ khi cốt lõi là hội chứng HƯ TỔNG QUÁT thuần bệnh lý (mọi token đều là token
                bệnh lý, không mang tạng phủ — có tạng thì tầng thể-tổng-quát mới là chuẩn) ->
                thể KB cùng cực Hư, KHÔNG xung trục tứ hư (thể mang 'âm/huyết/dương/khí' khác
                trục cốt lõi bị loại: 'Vị âm bất túc' không được thay 'Khí hư' — luật cấm nhầm
                Khí suy với Tân dịch hao tổn), và tả được >=1 triệu chứng của ca.
          Tiêu: hội chứng kèm theo THỰC -> thể KB cùng cực Thực LIÊN QUAN y lý
                (_are_syndromes_related) và tả được >=1 triệu chứng của ca.
        Trả [(is_core, row)] — có thể rỗng. Không bịa y lý: bài đều là dòng CSV của chính bệnh
        danh; caller in rõ vai trò 'thể KB' để không đọc nhầm thành bài đặc trị đích danh."""
        _patho_toks = {
            "hư", "suy", "nhược", "tổn", "nhiệt", "hàn", "thấp", "đàm", "đờm", "trọc",
            "ẩm", "hỏa", "hoả", "ứ", "trệ", "uất", "kết", "tích", "nghịch", "độc",
            "táo", "thử", "phong", "khí", "huyết", "âm", "dương",
        }
        # Token ĐỊNH LƯỢNG trung tính hay gặp trong nhãn tổng quát ('Khí huyết LƯỠNG hư',
        # 'Khí huyết ĐỀU hư') — không phải token bệnh lý nhưng cũng không phải tạng phủ.
        _neutral_toks = {"lưỡng", "đều", "song", "cả", "chứng"}
        _hu_axes = {"khí", "huyết", "âm", "dương"}
        rows = getattr(self, "csv_rows", None) or []
        case_syms = [str(s).lower().strip() for s in (case_symptoms or []) if str(s).strip()]

        def _overlap(row):
            txt = (row.get("triệu_chứng") or "").lower()
            return sum(1 for s in case_syms if s in txt)

        def _pick(target_syn, want_hu, need_related):
            best, best_score = None, 0
            t_axes = set(re.findall(r'[^\W\d_]+', (target_syn or "").lower())) & _hu_axes
            for row in rows:
                b = row.get("benh_ly", "").strip()
                hc = row.get("hoi_chung", "").strip()
                bt = row.get("bai_thuoc", "").strip()
                if not b or not hc or not bt or b.lower() not in diseases_lower:
                    continue
                if (b.lower(), bt.lower()) in printed_pairs:
                    continue
                if self._syndrome_is_hu(hc) != want_hu:
                    continue
                if want_hu:
                    hc_axes = set(re.findall(r'[^\W\d_]+', hc.lower())) & _hu_axes
                    if hc_axes - t_axes:
                        continue
                if need_related and not self._are_syndromes_related(target_syn, hc):
                    continue
                score = _overlap(row)
                if score > best_score:
                    best, best_score = row, score
            return best   # None nếu không thể KB nào tả được >=1 triệu chứng của ca

        picks = []
        fp_toks = set(re.findall(r'[^\W\d_]+', (final_primary or "").lower()))
        if fp_toks and fp_toks <= (_patho_toks | _neutral_toks) and self._syndrome_is_hu(final_primary):
            row = _pick(final_primary, want_hu=True, need_related=False)
            if row:
                picks.append((True, row))
        fc = (final_concurrent or "").strip()
        if fc and fc.lower() not in ("không có", "chưa rõ") and not self._syndrome_is_hu(fc):
            row = _pick(fc, want_hu=False, need_related=True)
            if row:
                picks.append((False, row))
        return picks

    _THIN_COAT_RE = re.compile(r'rêu(?:\s+lưỡi)?\s+trắng\s+mỏng')
    _DAMP_ATTR_RE = re.compile(r'\b(thủy thấp|thuỷ thấp|ẩm thấp|đàm|đờm|thấp trệ|thấp ứ|thấp tích|thủy ẩm)\b')
    # Động từ QUY NHÂN đứng ngay trước 'rêu trắng mỏng': 'khí huyết suy yếu ... TẠO RA rêu trắng
    # mỏng' — lấy cơ chế của CHẤT LƯỠI NHẠT (vinh nhuận) gán cho RÊU. Rêu trắng mỏng là rêu sinh
    # lý, bệnh lý không 'tạo ra' nó (trừ ngoại cảm biểu — caller chừa qua final_primary).
    _COAT_CAUSAL_RE = re.compile(
        r'(?:tạo ra|tạo thành|gây ra|gây nên|sinh ra|dẫn đến|hình thành|khiến(?:\s+cho)?|làm(?:\s+cho)?)'
        r'\s+(?:hiện tượng\s+|tình trạng\s+)?rêu(?:\s+lưỡi)?\s+trắng\s+mỏng')

    def _strip_thin_coating_damp_claims(self, text: str, final_primary: str = "") -> str:
        """[NHẤT QUÁN RÊU MỎNG] Luật 14 của prompt CẤM giải thích 'rêu trắng mỏng' bằng Đàm/Thủy
        thấp (thấp trọc bám lưỡi thì rêu bắt buộc DÀY/NHỚT) — LLM vẫn thỉnh thoảng vi phạm ở
        Mục 3 ('ẩm thấp tích tụ, gây ra rêu trắng mỏng') trong khi Mục 4 nói đúng ('vị khí còn
        tốt') -> hai mục tự mâu thuẫn ngay trong một kết quả. Chốt bằng code: quét từng CÂU,
        câu nào vừa nhắc rêu trắng mỏng vừa quy nó cho thấp/đàm thì thay bằng nhận định sinh lý
        chuẩn của luật 14 (chỉ chèn 1 lần; các câu vi phạm sau bị xóa hẳn).
        Mở rộng: câu QUY NHÂN trực tiếp ('khí huyết suy yếu ... tạo ra rêu trắng mỏng' — trộn cơ
        chế chất-lưỡi-nhạt sang rêu) cũng bị thay, NHƯNG chỉ ở ca KHÔNG phải ngoại cảm biểu —
        ca biểu được phép nói 'tà mới xâm nhập ... dẫn đến rêu trắng mỏng' theo chính luật 14."""
        if not text or not self._THIN_COAT_RE.search(text.lower()):
            return text
        allow_causal = self._syndrome_is_exterior_wind(final_primary)
        sanctioned = ("Riêng rêu trắng mỏng là rêu sinh lý, cho thấy vị khí còn tốt, "
                      "chính khí chưa suy kiệt, bệnh còn ở mức độ nhẹ.")
        has_sanctioned = "vị khí còn tốt" in text
        out_lines = []
        for line in text.split("\n"):
            sentences = re.split(r'(?<=[.!?])\s+', line)
            changed = False
            new_parts = []
            for sent in sentences:
                sl = sent.lower()
                _damp_hit = self._THIN_COAT_RE.search(sl) and self._DAMP_ATTR_RE.search(sl)
                _causal_hit = (not allow_causal) and self._COAT_CAUSAL_RE.search(sl)
                if _damp_hit or _causal_hit:
                    logger.info(f"[NHẤT QUÁN RÊU MỎNG] Gỡ câu quy nhân rêu trắng mỏng "
                                f"({'thấp/đàm' if _damp_hit else 'quy nhân trực tiếp'}): {sent[:90]!r}")
                    changed = True
                    if not has_sanctioned:
                        new_parts.append(sanctioned)
                        has_sanctioned = True
                    continue
                new_parts.append(sent)
            out_lines.append(" ".join(p for p in new_parts if p) if changed else line)
        return "\n".join(out_lines)

    # Lưỡi KHÔNG/ÍT rêu (kính diện thiệt, bong rêu) = ÂM HƯ / TÂN DỊCH KHUY. LLM Mục 4 hay quy SAI
    # cho huyết hư/khí hư/thấp/huyết ứ (over-fit theo cốt lõi đã chốt).
    _NO_COAT_RE = re.compile(
        r'(?:không|hầu như không|gần như không)\s*(?:có\s+)?rêu'
        r'|(?:^|[^\w])ít\s+rêu|rêu(?:\s+lưỡi)?\s+ít|lưỡi\s+(?:nhẵn\s+)?(?:trơn\s+)?bóng'
        r'|kính\s+diện\s+thiệt|bong\s+(?:tróc\s+)?rêu')
    _NO_COAT_WRONG_ATTR_RE = re.compile(
        r'\b(huyết hư|khí hư|khí huyết|dương hư|thủy thấp|thuỷ thấp|ẩm thấp|thấp trệ|đàm|đờm|huyết ứ|khí trệ)\b')
    # Cốt lõi liên quan âm/tân/táo -> quy lưỡi-không-rêu cho nó là ĐÚNG (không gỡ).
    _CORE_YIN_RE = re.compile(r'\b(âm|tân dịch|dịch|táo|khô)\b')

    def _strip_no_coating_yin_claims(self, text: str, final_primary: str = "", bat_cuong_hint: str = "") -> str:
        """[NHẤT QUÁN LƯỠI-KHÔNG-RÊU] Lưỡi không/ít rêu (kính diện thiệt) = ÂM HƯ / TÂN DỊCH KHUY,
        KHÔNG phải huyết hư/khí hư/thấp/huyết ứ. LLM Mục 4 hay over-fit (vd core Huyết hư -> 'lưỡi
        không có rêu là biểu hiện của huyết hư' — SAI; chính KG map lưỡi-không-rêu vào Can thận âm
        hư/Khí âm/Khí huyết hư chứ KHÔNG phải Huyết hư đơn thuần). Cùng lớp với luật cracks (lưỡi
        nứt = âm hư). Nếu cốt lõi KHÔNG liên quan âm/tân/táo, gỡ câu quy sai + chèn nhận định chuẩn
        (chỉ 1 lần). Cốt lõi âm/tân -> giữ nguyên (quy cho nó là đúng)."""
        if not text or not self._NO_COAT_RE.search(text.lower()):
            return text
        if self._CORE_YIN_RE.search((final_primary or "").lower()):
            return text
        # [THERMAL-AWARE] Bát Cương HÀN / core khí-dương hư -> lưỡi ít rêu là VỊ KHÍ HƯ TỔN (không đủ
        # huân chưng sinh rêu), KHÔNG phải âm dịch hao tổn/âm hư (âm hư = hư nhiệt, trái Bát Cương Hàn).
        _bc = (bat_cuong_hint or "").lower()
        _fp = (final_primary or "").lower()
        if "nhiệt" not in _bc and ("hàn" in _bc
                                   or re.search(r'\b(khí hư|dương hư|tỳ hư|vị hư|khí huyết)\b', _fp)):
            sanctioned = ("Lưỡi không (ít) rêu ở thể hư-hàn phản ánh VỊ KHÍ hư tổn không đủ huân chưng "
                          "sinh rêu, KHÔNG phải âm hư/nội nhiệt; là dấu nền nên theo dõi thêm.")
        elif "nhiệt" in _bc and ("thực" in _bc or "biểu" in _bc):
            # THỰC NHIỆT / ngoại cảm nhiệt: lưỡi ít rêu do NHIỆT HAO TÂN DỊCH (cấp), KHÔNG phải âm hư
            # sẵn có (âm hư = HƯ chứng, mâu thuẫn 'Thực chứng thuần túy' ở Mục 3); là dấu nền theo dõi.
            sanctioned = ("Lưỡi không (ít) rêu trong thể thực-nhiệt phản ánh NHIỆT làm hao tân dịch nhẹ, "
                          "KHÔNG phải âm hư sẵn có; là dấu nền nên theo dõi thêm.")
        else:
            sanctioned = ("Lưỡi không (ít) rêu phản ánh âm dịch/tân dịch hao tổn nhẹ (dấu âm hư), "
                          "không phải biểu hiện của huyết hư; là dấu nền nên theo dõi thêm.")
        has_sanctioned = ("âm dịch" in text.lower() or "vị khí hư tổn" in text.lower()
                          or "nhiệt làm hao tân dịch" in text.lower())
        out_lines = []
        for line in text.split("\n"):
            sentences = re.split(r'(?<=[.!?])\s+', line)
            changed = False
            new_parts = []
            for sent in sentences:
                sl = sent.lower()
                if self._NO_COAT_RE.search(sl) and self._NO_COAT_WRONG_ATTR_RE.search(sl):
                    logger.info(f"[NHẤT QUÁN LƯỠI-KHÔNG-RÊU] Gỡ câu quy sai lưỡi-không-rêu: {sent[:90]!r}")
                    changed = True
                    if not has_sanctioned:
                        new_parts.append(sanctioned)
                        has_sanctioned = True
                    continue
                new_parts.append(sent)
            out_lines.append(" ".join(p for p in new_parts if p) if changed else line)
        return "\n".join(out_lines)

    @staticmethod
    def _syndrome_thermal_sign(name):
        """Cực HÀN / NHIỆT của hội chứng theo TÊN. Trả 'han', 'nhiet', hoặc None (trung tính hoặc
        lẫn cả hai — vd 'Phong hàn hóa nhiệt', 'Thượng nhiệt hạ hàn' -> None, không coi là xung).
        Táo/ôn/thử xếp phía NHIỆT: phong táo (Tang hạnh thang tân lương nhuận táo) đối lập pháp trị
        với phong hàn (tân ôn giải biểu) — dùng để chặn mượn chéo giữa hai cực."""
        nl = (name or "").lower()
        han = bool(re.search(r'\b(hàn|lạnh)\b', nl))
        nhiet = bool(re.search(r'\b(nhiệt|hỏa|hoả|ôn|thử|táo)\b', nl))
        if han and not nhiet:
            return "han"
        if nhiet and not han:
            return "nhiet"
        return None

    @classmethod
    def _syndromes_thermal_conflict(cls, a, b):
        """True khi hai hội chứng ĐỐI CỰC hàn-nhiệt rõ ràng (một 'han', một 'nhiet'). Hội chứng
        trung tính/lẫn cực -> False (không chặn). Dùng để KHÔNG mượn metadata Bát Cương lẫn bài
        thuốc từ ứng viên trái cực (vd cốt lõi Phong hàn không được mượn Phong táo/Phong nhiệt —
        chúng 'liên quan' qua chữ 'phong' nhưng pháp trị ngược nhau)."""
        sa, sb = cls._syndrome_thermal_sign(a), cls._syndrome_thermal_sign(b)
        return sa is not None and sb is not None and sa != sb

    @classmethod
    def _gate_concurrent_by_disease(cls, final_primary, final_concurrent, matched_diseases,
                                    disease_names, overridden):
        """[CỔNG CÙNG-BỆNH cho HỘI CHỨNG KÈM THEO] Trả (final_concurrent_đã_lọc, reason|None).

        Hội chứng kèm theo (final_concurrent) chọn theo rank KG + khử trùng-tên, KHÔNG kiểm nó có
        thuộc bệnh danh đã chốt hay không -> ứng viên rank-2 khớp triệu chứng CHUNG (sợ gió, rêu
        trắng mỏng) của MỘT BỆNH KHÁC lọt vào (vd 'Doanh vệ bất hòa' — vốn CHỈ thuộc 'Tiểu nhi hãn
        chứng', bài Quế chi thang gia Hoàng kỳ tính ẤM — ghép nhầm vào 'Viêm yết hầu × Phong nhiệt
        phạm phế'). Cổng _syndromes_thermal_conflict KHÔNG cứu được vì tên 'Doanh vệ bất hòa' vô
        dấu-nhiệt (thermal_sign=None). Chặn theo CẤU TRÚC: kèm-theo phải là hội chứng của ÍT NHẤT
        một bệnh trong CỬA SỔ chief-complaint (ratio cao nhất).

        Vì sao neo theo CỬA SỔ RATIO (tính lại tại chỗ) chứ KHÔNG theo disease_names: disease_names
        được lọc qua valid_syndromes vốn CHỨA chính final_concurrent, nên concurrent giả có thể tự
        kéo bệnh-nhà của nó vào rồi tự-hợp-thức (vòng lặp). Cửa sổ ratio miễn nhiễm.

        UNION theo >=1 bệnh cửa sổ (KHÔNG bó cùng-bệnh-chính) để VẪN cho phép kèm-theo chéo bệnh
        HỢP LỆ kiểu biểu-lý đồng bệnh (vd Cảm mạo Phong hàn + Thực tích của Thương thực — cả hai
        cùng trong cửa sổ). Thành viên khớp EXACT/SUBSTRING (KHÔNG _are_syndromes_related — nhóm
        khái niệm quá lỏng, rò qua 1 âm tiết chung 'phong/khí/huyết'). Cổng thermal giữ làm phụ trợ
        (bắt ca kèm-theo có dấu cực đối nghịch rõ trong tên). Chừa 8 nhánh hardcode (overridden=True
        — cố ý ghép tiêu-bản chéo bệnh). Giữ nguyên khi disease_names/matched_diseases rỗng (bay mù).
        """
        fc = (final_concurrent or "").strip().lower()
        if overridden or fc in ("không có", "", "chưa rõ") or not disease_names or not matched_diseases:
            return final_concurrent, None
        gb = matched_diseases[0].get("ratio", 0.0)
        window = [m for m in matched_diseases
                  if m.get("ratio", 0.0) >= gb - 0.15 and m.get("ratio", 0.0) >= 0.30]
        union_hc = set()
        for m in window:
            for hc in m.get("hoi_chung_all", [m.get("hoi_chung", "")]):
                hc = (hc or "").strip().lower()
                if hc:
                    union_hc.add(hc)
        member = any(fc == hc or fc in hc or hc in fc for hc in union_hc)
        thermal_bad = cls._syndromes_thermal_conflict(final_primary, final_concurrent)
        if (not member) or thermal_bad:
            reason = (f"loại kèm-theo {final_concurrent!r} khỏi cốt lõi {final_primary!r} "
                      f"(member={member}, thermal_bad={thermal_bad}; "
                      f"cửa sổ={[m.get('benh_ly') for m in window]})")
            return "Không có", reason
        return final_concurrent, None

    _synonym_clusters = None  # cache cụm đồng nghĩa (list[set[str]] tên đã chuẩn hoá)

    @staticmethod
    def _norm_syn_name(s):
        """Chuẩn hoá tên hội chứng để so cụm đồng nghĩa: chữ thường, bỏ nội dung ngoặc, gộp space."""
        s = re.sub(r'\([^)]*\)', ' ', (s or '').lower())
        return re.sub(r'\s+', ' ', s).strip()

    @classmethod
    def _load_synonym_map(cls):
        """Nạp data/syndrome_synonyms.json -> list các set tên chuẩn hoá (>=2 thành viên). Cache."""
        if cls._synonym_clusters is not None:
            return cls._synonym_clusters
        import json as _json
        import os as _os
        path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "data", "syndrome_synonyms.json")
        clusters = []
        try:
            data = _json.load(open(path, encoding="utf-8"))
            for cl in data.get("clusters", []):
                members = {cls._norm_syn_name(x) for x in cl if x and str(x).strip()}
                if len(members) >= 2:
                    clusters.append(members)
        except Exception as e:
            logger.warning(f"Không nạp được syndrome_synonyms.json: {e}")
        cls._synonym_clusters = clusters
        return clusters

    @classmethod
    def _syndromes_are_synonyms(cls, a, b):
        """True khi a,b ĐỒNG NGHĨA (cùng một cụm curated data/syndrome_synonyms.json) — dùng để
        bắc cầu tra bài Mục 5 khi nhãn core lệch tên với hội chứng của bệnh đã chốt (vd 'Can vị bất
        hòa' vs 'Can khí phạm vị'). KHÔNG dùng _are_syndromes_related (lỏng: 46% cặp liên quan lệch
        trục Âm/Dương -> kê trái cực). Chỉ xét khi tên KHÁC nhau (đồng nhất đã do tầng exact lo)."""
        na, nb = cls._norm_syn_name(a), cls._norm_syn_name(b)
        if not na or not nb or na == nb:
            return False
        return any(na in cl and nb in cl for cl in cls._load_synonym_map())

    @classmethod
    def _core_grounded_in_window(cls, syn, window):
        """True khi hội chứng `syn` THỰC SỰ thuộc một bệnh trong cửa sổ chief-complaint (khớp
        EXACT/SUBSTRING/ĐỒNG NGHĨA với hoi_chung_all). KHÔNG dùng _are_syndromes_related (lỏng: 46%
        cặp lệch Âm/Dương, và 'khí trệ huyết ứ' ~ 'can khí phạm vị' qua 'khí' -> tưởng grounded oan).
        Substring có chủ đích để nhận biến thể tạng: core 'Phế khí hư' grounded ở bệnh có 'Khí hư'."""
        sl = (syn or "").lower().strip()
        if not sl:
            return False
        for m in (window or []):
            for hc in m.get("hoi_chung_all", [m.get("hoi_chung", "")]):
                hcl = (hc or "").lower().strip()
                if hcl and (sl == hcl or sl in hcl or hcl in sl or cls._syndromes_are_synonyms(syn, hc)):
                    return True
        return False

    @classmethod
    def _reground_core(cls, final_primary, all_syndromes, window):
        """[CỔNG GROUNDING CORE] all_syndromes[0] có thể là hội chứng PROMISCUOUS của bệnh KHÁC (vd
        'Khí trệ huyết ứ' xuất hiện ở 19 bệnh) thắng oan core dù KHÔNG thuộc bệnh trong cửa sổ chief-
        complaint -> Mục 5 trắng + biện chứng lệch (scorer _score_syndromes_grounded gộp triệu chứng
        qua MỌI bệnh, không scope). Nếu core KHÔNG grounded ở cửa sổ, re-rank sang ứng viên
        all_syndromes CAO NHẤT (giữ thứ hạng scorer) có grounded. Trả (core_mới, reason|None).

        AN TOÀN: chỉ THU HẸP pool -> NO-OP nếu core vốn grounded (kể cả biến thể tạng 'Phế khí hư' ⊃
        'Khí hư') -> không phá ca tốt, chỉ ground lại ca ngoại lai. Nếu KHÔNG ứng viên nào grounded
        -> giữ core cũ (thà 'chưa có bài' còn hơn đổi bừa)."""
        if not window or not final_primary:
            return final_primary, None
        if cls._core_grounded_in_window(final_primary, window):
            return final_primary, None
        reg = next((s for s in (all_syndromes or []) if cls._core_grounded_in_window(s, window)), None)
        if reg and reg.lower().strip() != final_primary.lower().strip():
            return reg, f"core ngoại lai {final_primary!r} (promiscuity) -> re-rank grounded {reg!r}"
        return final_primary, None

    # Cụm "bạn hữu giả": chứa âm tiết trùng keyword bệnh lý nhưng vô hại ('sốt' trong 'sốt ruột',
    # 'thực' trong 'thực sự'). Gỡ khỏi text TRƯỚC khi khớp keyword Hàn/Nhiệt/Hư/Thực ở MỌI tầng
    # (Bát Cương lẫn run_diagnosis) để hai tầng không mâu thuẫn nhau.
    _FALSE_FRIEND_PHRASES = ["thực sự", "quả thực", "thực ra", "thực phẩm", "sốt ruột", "sốt sắng", "nôn nóng"]

    # Che phủ định trên TEXT THÔ (cổng hàn-nhiệt + khớp mềm bệnh danh). Từ khóa các cổng đều là
    # dấu DƯƠNG TÍNH ('sốt', 'sợ lạnh', 'khát nước'...) nên xóa vùng sau 'không/chưa/chẳng' là đúng
    # hướng: 'không sốt' không được tính là dấu nhiệt. 'không những/chỉ' = nhấn mạnh CÓ -> bỏ qua.
    _NEG_RAW_CUE = re.compile(r'\b(?:không|chẳng|chả|chưa|ko)\b')
    _NEG_RAW_STOP = re.compile(r'[,.;:!?]|\b(?:nhưng|mà|còn|song|tuy)\b')

    @classmethod
    def _mask_negated_text(cls, text_lower: str, protected_phrases=()) -> str:
        """Thay bằng dấu cách vùng bị phủ định (từ 'không/chưa/chẳng' TỰ DO tới ranh giới mệnh đề
        gần nhất, tối đa 30 ký tự). protected_phrases = các cụm chứa 'không' vốn là TÊN triệu chứng
        ('tay chân không ấm', 'miệng nhạt không khát'): chữ 'không' bên trong chúng KHÔNG bị coi là
        phủ định (che chở để khớp mềm không đứt tên triệu chứng thật)."""
        if not text_lower or "không" not in text_lower and "chưa" not in text_lower \
                and "chẳng" not in text_lower and "chả" not in text_lower and "ko" not in text_lower:
            return text_lower
        protected_spans = []
        for ph in protected_phrases:
            if not any(n in ph for n in ("không", "chưa", "chẳng")) or ph not in text_lower:
                continue
            for m in re.finditer(re.escape(ph), text_lower):
                protected_spans.append((m.start(), m.end()))

        def _in_protected(pos):
            return any(a <= pos < b for a, b in protected_spans)

        out = list(text_lower)
        for m in cls._NEG_RAW_CUE.finditer(text_lower):
            if _in_protected(m.start()):
                continue
            tail = text_lower[m.end():m.end() + 8].lstrip()
            if tail.startswith("những") or tail.startswith("chỉ"):
                continue
            region = text_lower[m.end():m.end() + 30]
            stop = cls._NEG_RAW_STOP.search(region)
            end = m.end() + (stop.start() if stop else len(region))
            for i in range(m.start(), end):
                out[i] = " "
        return "".join(out)

    # Tiền tố bào chế (Sinh/Sao/Chích...) và alias chính tả để KHỬ TRÙNG vị thuốc: KG gộp nhiều dòng
    # CSV gần trùng của cùng bài -> danh sách lặp biến thể ('Sinh mẫu lệ'+'Mẫu lệ', 'Màn kinh'+'Mạn
    # kinh tử'). Gộp về một khi hiển thị Mục 5.
    _HERB_PREFIXES = ("sinh ", "sao ", "chích ", "thán ", "than ", "bào ", "nướng ", "tẩm ", "chế ")
    _HERB_ALIASES = {"màn kinh": "mạn kinh tử", "mạn kinh": "mạn kinh tử"}

    # Ghi chú ĐUÔI trong field vị_thuốc (bài thay thế / gia giảm điều kiện) — phải CẮT trước khi tách
    # theo dấu phẩy, nếu không _dedupe_herbs xé ngoặc (dấu phẩy nội bộ) thành token rác ('gia: Trần
    # bì', 'Nếu có đờm trệ') + hỏng vị base cuối ('Sinh Cam thảo. (Hoặc dùng...'). Ngoặc BÀO CHẾ (a:
    # chích/tẩm muối/sao... dính liền tên vị, KHÔNG sau '. ') và ALIAS (e: '(Đương quy)') KHÔNG bị cắt.
    # Marker gia:/nếu bắt buộc dạng 'gia:'(có dấu hai chấm) hoặc 'nếu' để KHÔNG cắt nhầm 'Sa tiền
    # (gia thêm)'/'Thạch cao (có thể)'. Không có ngoặc lồng trong data (đã quét) nên [^)]* an toàn.
    _HERB_TAIL_CUT = re.compile(
        r'\.\s*\('
        r'|[,;]\s*\(\s*(?:hoặc\s+dùng|kết\s+hợp|nếu\b|gia\s*:|tam\s+tử|chân\s+vũ|tham\s+khảo|sau\s+khi|không\s+liệt|có\s+thể\s+phối)'
        r'|\(\s*(?:hoặc\s+dùng|kết\s+hợp\s+bột|nếu\b|gia\s*:)'
        r'|[.;]\s*(?:nếu\b|gia\s*:|thường\s+gia|có\s+thể\s+phối\s+hợp|sau\s+khi\b|thể\s+(?:thực|hư|âm|dương)\b)'
        # header bài thứ 2 sau '.'/'; ': '<cụm ngắn>:' (kể cả tên bài KHÔNG kết thúc formula-word,
        # vd 'Thập toàn đại bổ:'); an toàn vì vị thuốc base không có 'dấu hai chấm'.
        r'|[.;]\s*[^.;,():]{2,30}:'
        # bất kỳ ';' ở đuôi có kèm cụm điều kiện (nếu/gia/thêm/hàn/nhiệt/kiêm) -> note gia giảm không ngoặc
        r'|;\s*[^;]*?\b(?:nếu|gia\b|thêm\b|bỏ\b|kiêm\b|hàn\s+thấp|thấp\s+nhiệt)',
        re.IGNORECASE)
    # thay-vị/loại-vị inline: 'X (hoặc/nay thay bằng/thay/bỏ/bớt ...)' -> 'X'. Lookahead loại 'hoặc dùng/bài'.
    _HERB_INLINE_SUB = re.compile(
        r'\s*\(\s*(?:hoặc(?!\s+dùng)(?!\s+bài)|nay\s+thay\s+bằng|thay|bỏ|bớt)\b[^)]*\)', re.IGNORECASE)
    # ngoặc CHÚ THÍCH đứng ĐẦU field ('(Tùy bài chọn) Nhân sâm...') -> bỏ ngoặc dẫn đầu.
    _HERB_LEAD_NOTE = re.compile(r'^\s*\([^)]*\)\s*', re.IGNORECASE)

    @classmethod
    def _strip_herb_tail_notes(cls, vi_str: str) -> str:
        """Cắt ghi chú đuôi (bài thay thế/gia giảm điều kiện) + rút gọn thay-vị inline 'X (hoặc Y)'->'X'
        + bỏ ngoặc chú thích đầu field. GIỮ ngoặc bào chế + alias. Xem _HERB_TAIL_CUT."""
        if not vi_str or ("(" not in vi_str and "." not in vi_str and ";" not in vi_str):
            return vi_str
        s = cls._HERB_LEAD_NOTE.sub("", vi_str)       # bỏ ngoặc note dẫn đầu (nếu có)
        m = cls._HERB_TAIL_CUT.search(s)
        s = s[:m.start()] if m else s
        s = cls._HERB_INLINE_SUB.sub("", s)
        return s.strip().rstrip(".,; ").strip()

    @classmethod
    def _dedupe_herbs(cls, vi_str: str) -> str:
        """Khử trùng vị thuốc (gộp biến thể bào chế/chính tả), GIỮ VỊ TRÍ đầu tiên nhưng ưu tiên
        hiển thị dạng CHUẨN (không tiền tố bào chế, không phải alias-sai)."""
        if not vi_str:
            return vi_str
        vi_str = cls._strip_herb_tail_notes(vi_str)   # cắt ghi chú đuôi TRƯỚC khi split ',' (chống blob)
        order, best = [], {}
        for h in vi_str.split(","):
            h = h.strip()
            if not h:
                continue
            key = re.sub(r"\s+", " ", h.lower()).strip()
            # Bỏ NGOẶC chú thích/bào chế và HẬU TỐ bào chế để gộp biến thể cùng vị: 'Tri mẫu
            # (tẩm muối)' == 'Tri mẫu Tẩm muối' == 'Tri mẫu'; '(kết hợp bột X)' -> rỗng -> bỏ (là
            # ghi chú, không phải vị). Trước đây chỉ hạ hoa-thường nên các biến thể này lọt trùng.
            key = re.sub(r"\s*\([^)]*\)", "", key).strip()
            key = re.sub(r"\s+(tẩm\s+\S+|sao(?:\s+\S+)?|thán|chế|sống|tươi|phi|nướng|bào)$", "", key).strip()
            if not key:
                continue
            prefixed = False
            for p in cls._HERB_PREFIXES:
                if key.startswith(p):
                    key = key[len(p):].strip()
                    prefixed = True
                    break
            aliased = key in cls._HERB_ALIASES
            key = cls._HERB_ALIASES.get(key, key)
            is_canon = not prefixed and not aliased
            if key not in best:
                best[key] = h
                order.append(key)
            elif is_canon:                 # thay bản đang lưu bằng dạng chuẩn hơn
                best[key] = h
        return ", ".join(best[k] for k in order)

    @classmethod
    def _kw_hit_clean(cls, text_lower: str, kws) -> bool:
        """Khớp keyword theo ranh giới từ (\\b) sau khi gỡ cụm bạn-hữu-giả VÀ che vùng phủ định."""
        for _ff in cls._FALSE_FRIEND_PHRASES:
            text_lower = text_lower.replace(_ff, " ")
        text_lower = cls._mask_negated_text(text_lower)
        return any(re.search(r"\b" + re.escape(k) + r"\b", text_lower) for k in kws)

    @staticmethod
    def _syndrome_is_hu(name: str) -> bool:
        """Hội chứng thuộc HƯ chứng (bản chất suy yếu/bất túc) theo từ khóa trong tên.
        Khớp theo RANH GIỚI TỪ: bản substring cũ dính 'hư' TRONG 'thượng' khiến các hội chứng
        thực thuần như 'Can dương thượng cang', 'Vị khí thượng nghịch' bị coi là Hư (sai cả luật
        đảo Bản Hư Tiêu Thực lẫn phân loại dọn tag Bát Cương)."""
        return bool(re.search(r'\b(hư|suy|bất túc|nhược|khuy|tổn|thiếu)\b', (name or "").lower()))

    @classmethod
    def _syndrome_is_thuc_pure(cls, name: str) -> bool:
        """Hội chứng THỰC THUẦN (tà khí tích tụ: đàm/hỏa/nhiệt/ứ/trệ/nghịch...) và KHÔNG kèm chữ Hư.
        Hội chứng vừa Hư vừa Thực (vd 'Khí hư huyết ứ') coi là Hư (đã là Bản Hư Tiêu Thực nội tại)."""
        nl = (name or "").lower()
        # Tà khí thực: đàm/hỏa/nhiệt/ứ/trệ/uất/kết/ngưng/tích/độc/nghịch/phong + hàn/thấp (ngoại tà,
        # vd "Hàn thấp", "Thấp nhiệt", "Phong hàn"). "Hư hàn" có chữ "hư" nên vẫn được coi là Hư.
        # Ranh giới từ: tránh 'ứ' dính trong 'chứng' ("Chứng tỳ thận lưỡng hư"), 'táo' trong tên bệnh ghép.
        # 'đờm/trọc/ẩm': biến thể chính tả và tứ ẩm (Đờm trọc ngăn trở, Huyền ẩm, Thủy ẩm) đều là tà thực.
        has_thuc = bool(re.search(r'\b(đàm|đờm|trọc|ẩm|hỏa|hoả|nhiệt|ứ|trệ|uất|kết|ngưng|tích|thực|độc|nghịch|phong|hàn|thấp|thử|táo)\b', nl))
        return has_thuc and not cls._syndrome_is_hu(nl)

    @staticmethod
    def _syndrome_is_exterior_wind(name: str) -> bool:
        """Hội chứng NGOẠI CẢM BIỂU (phong tà phạm biểu/vệ/phế): Phong hàn, Phong nhiệt, Phong thấp,
        Biểu hàn/nhiệt, Thương phong, *phạm phế/phạm vệ... Đây là BIỂU THỰC cấp — gốc bệnh là tà khí
        ngoại cảm ở biểu, KHÔNG có Bản Hư nội thương → KHÔNG được đảo hội chứng Hư lên làm cốt lõi
        (nếu không sẽ dựng sai cơ chế 'Bản Hư' cho một cảm mạo cấp)."""
        nl = (name or "").lower()
        # \b: 'hư' substring dính trong 'THƯƠNG phong' khiến chính tên ngoại cảm kinh điển bị loại
        if re.search(r'\bhư\b', nl):                       # 'Phế khí hư', 'Biểu hư tự hãn' = nội thương
            return False
        if re.search(r'phong\s*(hàn|nhiệt|thấp|táo|thử|ôn)', nl):   # Phong hàn/nhiệt/thấp...
            return True
        if re.search(r'biểu\s*(hàn|nhiệt|thực|chứng)', nl):          # Biểu hàn/nhiệt/thực
            return True
        if any(k in nl for k in ("phạm phế", "phạm vệ", "thương phong", "ngoại cảm")):
            return True
        return False

    # Dấu hiệu SINH LÝ bình thường (rêu nhuận = tân dịch tốt, lưỡi hồng = sắc khỏe...) — KHÔNG được
    # dùng làm bằng chứng bệnh khi chấm điểm hội chứng, dù KG có node trùng tên (vd 'rêu lưỡi nhuận'
    # nối với 'Ứ huyết' khiến bệnh nhân khỏe bị cộng oan điểm Ứ huyết).
    _PHYSIOLOGICAL_TERMS = {
        "rêu lưỡi nhuận", "rêu nhuận", "rêu lưỡi trắng nhuận", "rêu trắng nhuận",
        "rêu trắng mỏng nhuận", "rêu lưỡi bình thường", "rêu bình thường",
        "lưỡi bình thường", "lưỡi hồng", "lưỡi hồng nhuận", "lưỡi hồng hào",
        # 'lưỡi hồng nhạt' = đạm hồng thiệt — MÀU LƯỠI SINH LÝ chuẩn theo Đông y. KG có node trùng
        # tên gắn với Huyết hư/Khí trệ đờm ngưng... khiến bệnh nhân lưỡi bình thường bị cộng oan điểm.
        "lưỡi hồng nhạt", "chất lưỡi hồng nhạt",
        "mạch bình thường", "mạch hoãn", "sắc mặt hồng hào", "da hồng hào",
        "mặt bình thường", "sắc mặt bình thường",
    }

    # [NHÓM ĐỒNG NGHĨA CHẤM ĐIỂM] Các biến thể cùng MỘT phát hiện lâm sàng: gộp thành MỘT pattern
    # alternation khi chấm điểm hội chứng, để (1) không đếm trùng 1 phát hiện 2 lần ('rêu trắng mỏng'
    # + 'rêu trắng mỏng hoặc trắng vàng'), và (2) bắc cầu từ vựng bệnh nhân -> từ vựng node KG
    # ('rìa lưỡi có vết răng' -> node 'thể lưỡi bệu có hằn răng...', 'khó thở' -> node 'đoản khí').
    # Thiết kế đã kiểm chứng trên Neo4j thật với 4 ca chuẩn (ca khí hư+đàm thấp, ca Tỳ khí hư,
    # ca Huyết hư điển hình, ca Phong hàn biểu chứng) — không phá ca nào.
    _SCORING_TERM_GROUPS = [
        {"khó thở", "đoản khí", "hơi thở ngắn", "thở ngắn", "hụt hơi", "đoản hơi"},
        {"rìa lưỡi có vết răng", "rìa lưỡi có hằn răng", "lưỡi có vết răng",
         "lưỡi có hằn răng", "hằn răng", "vết răng", "dấu răng"},
        {"tinh thần mệt mỏi", "mệt mỏi", "mệt mỏi vô lực"},
        {"rêu trắng mỏng", "rêu lưỡi trắng mỏng", "rêu trắng mỏng hoặc trắng vàng"},
        {"mặt nhợt", "mặt nhợt nhạt", "mặt trắng nhợt", "sắc mặt nhợt nhạt", "sắc mặt trắng nhợt",
         "sắc mặt trắng bệch"},
    ]

    def _score_syndromes_grounded(self, patient_terms: list, top_n: int = 6) -> list:
        """[GROUNDED SYNDROME-SCORING] Chấm điểm mỗi HoiChung = SỐ triệu chứng bệnh nhân khớp với
        triệu chứng của hội chứng đó trên graph (cạnh CÓ_BIỂU_HIỆN), khớp theo ranh giới từ để bắt
        cả node triệu chứng ghép. Ranking DETERMINISTIC, ổn định, không phụ thuộc LLM 7B chọn bừa.

        Trả về list [(syndrome_name, matched_count)] xếp điểm giảm dần (tối đa top_n)."""
        # Chuẩn hóa đồng nghĩa TRƯỚC khi chấm: _preprocess_question trả tên node DB thô ('mặt nhợt',
        # 'rìa lưỡi có vết răng'...) KHÔNG qua _resolve_symptom_conflicts, nên nếu không quy đổi thì
        # term trượt khỏi _SCORING_TERM_GROUPS và ranking lệch giữa các đường gọi (đã tái hiện thật).
        terms = self._normalize_symptoms([t for t in (patient_terms or []) if t and t.strip()])
        _physio = [t for t in terms if t in self._PHYSIOLOGICAL_TERMS]
        if _physio:
            logger.info(f"[GROUNDED] Loại dấu hiệu sinh lý khỏi bằng chứng chấm điểm: {_physio}")
            terms = [t for t in terms if t not in self._PHYSIOLOGICAL_TERMS]
        # [CHỐNG RÒ CROSS-REGION LƯỠI-NỨT] 'vết nứt/lưỡi nứt' (tongue crack) KHÔNG có node đúng chỗ
        # trên KG -> bare 'vết nứt' khớp NHẦM node 'Vết nứt hậu môn' (bệnh Hậu môn nứt kẽ × Khí trệ
        # huyết ứ), đẩy core sang 'huyết ứ' TRÁI luật (cracks = âm hư/tân dịch khuy, KHÔNG phải huyết
        # ứ — xem quy tắc SPECIAL RULE FOR CRACKS trong prompt vọng chẩn). Loại khỏi bằng chứng chấm
        # core. GIỮ term 'nứt' CÓ định vị vùng KHÁC lưỡi (vd 'vết nứt hậu môn') để khớp đúng bệnh vùng.
        def _is_tongue_crack(_t):
            _tl = (_t or "").lower()
            return ("nứt" in _tl) and not any(_r in _tl for _r in (
                "hậu môn", "môi", "gót", "bàn chân", "kẽ chân", "vú", "núm", "gan bàn", "kẽ tay", " da"))
        _bg_crack = [t for t in terms if _is_tongue_crack(t)]
        if _bg_crack:
            logger.info(f"[GROUNDED] Loại dấu nền lưỡi-nứt khỏi chấm core (chống rò cross-region): {_bg_crack}")
            terms = [t for t in terms if not _is_tongue_crack(t)]
        if not terms:
            return []
        # Java-regex khớp từ độc lập (tái dùng của qa_system, bắt cả triệu chứng nằm trong node ghép).
        # Term thuộc _SCORING_TERM_GROUPS -> MỘT pattern alternation cho cả nhóm (đếm 1 lần/phát hiện).
        patterns = []
        _seen_groups = set()
        for t in terms:
            _gi = next((i for i, g in enumerate(self._SCORING_TERM_GROUPS) if t in g), None)
            if _gi is None:
                patterns.append(self.qa_pipeline._word_boundary_pattern(t))
            elif _gi not in _seen_groups:
                _seen_groups.add(_gi)
                _alts = "|".join(r'\Q' + x + r'\E' for x in sorted(self._SCORING_TERM_GROUPS[_gi]))
                patterns.append(r'.*(^|[^\p{L}])(' + _alts + r')($|[^\p{L}]).*')
        # Điểm hiển thị = matched + idf_sum, nhưng XẾP HẠNG theo (matched DESC, idf DESC):
        #   - matched (số triệu chứng khớp) -> COVERAGE là yếu tố CHÍNH (khớp nhiều thắng).
        #   - idf_sum = tổng 1/df (df = số hội chứng có triệu chứng đó) -> CHỈ phá hoà khi coverage
        #     bằng nhau. KHÔNG cộng gộp vào khoá xếp hạng: triệu chứng df=1 đóng góp trọn 1.0 điểm
        #     khiến hội chứng hẹp khớp 2 triệu chứng vượt mặt hội chứng khớp 3 (đã tái hiện trên
        #     Neo4j thật: 'Huyết ứ' 2 khớp = 3.5 điểm > 'Thận dương hư' 2 khớp = 2.51 và suýt vượt
        #     'Huyết hư' 3 khớp = 3.58) — phá bất biến coverage-first mà không tầng nào sau cứu được.
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
        ORDER BY matched DESC, idf DESC, syndrome LIMIT 40
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
        # [PHÁ HÒA THEO ĐẶC HIỆU] Hai ứng viên HÒA ĐIỂM TUYỆT ĐỐI mà tên bên này CHỨA tên bên kia
        # (biến thể định vị tạng phủ của hội chứng chung, vd 'Phế khí hư' ⊃ 'khí hư') -> xếp biến
        # thể ĐẶC HIỆU lên trước. Không để thứ tự chữ cái của Cypher quyết định hội chứng cốt lõi
        # (đã xảy ra thật: Khí hư 5.129 = Phế khí hư 5.129, 'K' < 'P' nên cốt lõi thành Khí hư
        # chung chung dù bệnh cảnh khó thở + chảy mũi chỉ đích danh tạng Phế).
        for _i in range(len(picked) - 1):
            for _j in range(_i + 1, len(picked)):
                if abs(picked[_i][1] - picked[_j][1]) < 1e-6 \
                        and picked[_i][0].lower() in picked[_j][0].lower():
                    picked[_i], picked[_j] = picked[_j], picked[_i]
                    break
        # [KHỬ ĐỒNG NGHĨA TÊN] KG có node đồng nghĩa khác chữ ('Thận dương bất túc' = 'Thận dương
        # hư') mà dedupe-casefold không gộp được — nếu để cả hai, ranking sẽ trưng "cốt lõi X +
        # kèm theo X'" là cùng một hội chứng (đã xảy ra thật). Chuẩn hóa bất túc/nhược/suy yếu -> hư
        # rồi giữ bản xếp hạng cao nhất của mỗi tên chuẩn.
        _seen_norm = set()
        _dedup = []
        for name, score, matched in picked:
            _norm = re.sub(r'\bhư(\s+hư)+\b', 'hư', re.sub(
                r'\b(bất túc|hư nhược|suy yếu|nhược)\b', 'hư', name.lower()))
            _norm = re.sub(r'\s+', ' ', _norm).strip()
            if _norm in _seen_norm:
                continue
            _seen_norm.add(_norm)
            _dedup.append((name, score, matched))
        picked = _dedup
        logger.info(f"[GROUNDED] Điểm hội chứng (IDF): {[(n, round(s, 3), m) for n, s, m in picked]}")
        return [(name, round(score, 3)) for name, score, _matched in picked]

    # ------ CỔNG ÂM-HƯ KHÔNG NHIỆT (GUARD RULE 4 deterministic) ------
    # Âm hư = hư NHIỆT: định nghĩa phải kèm dấu nhiệt. Quét KG: 87% (40/46) hội chứng âm-hư có dấu
    # nhiệt trong triệu chứng -> ca âm-hư THẬT gần như luôn khai dấu nhiệt. Nếu core chốt ra là âm-hư
    # mà lời khai KHÔNG có dấu nhiệt/khô nào VÀ CÓ dấu hư-hàn/thấp (mệt/nặng nề/sợ lạnh/lưỡi trắng-nhợt)
    # -> gần như chắc chắn KHÔNG phải âm hư -> hạ bậc, chọn ứng viên non-âm-hư kế. 'Mồ hôi trộm'/'đạo
    # hãn' đơn độc KHÔNG tính là nhiệt (có thể do khí/dương hư). Trước đây luật này chỉ nằm trong prompt
    # LLM (mềm, bị 'mồ hôi trộm' kéo lệch); nay là cổng CỨNG.
    _AMHU_HEAT_SIGNS = (
        "sốt", "phát nhiệt", "triều nhiệt", "cốt chưng", "khát", "khô họng", "họng khô", "khô miệng",
        "miệng khô", "khô mũi", "lưỡi đỏ", "chất lưỡi đỏ", "đầu lưỡi đỏ", "lưỡi thon đỏ", "ít rêu",
        "không rêu", "rêu vàng", "rêu lưỡi vàng", "gò má đỏ", "má đỏ", "hai gò má đỏ", "bốc hỏa",
        "ngũ tâm phiền nhiệt", "lòng bàn tay nóng", "bàn tay chân nóng", "nóng trong", "phiền nhiệt",
        "tâm phiền", "nước tiểu vàng", "tiểu vàng", "nước tiểu đỏ", "táo bón", "đại tiện táo",
    )
    # CHỈ dấu hư-hàn/thấp ĐẶC HIỆU (hàn: sợ lạnh/tay chân lạnh/tiểu trong; thấp: nặng nề/đàm/phù/rêu
    # nhớt; khí-dương hư: đại tiện lỏng/hụt hơi...). KHÔNG gồm 'mệt mỏi/mệt/uể oải' — mệt là dấu TRUNG
    # TÍNH (mọi thể hư đều có, kể cả âm hư), dùng nó làm bằng chứng NGHỊCH âm-hư sẽ loại oan thể âm-hư
    # THẬT chỉ vì bệnh nhân kêu mệt (ca 'tiểu nhiều, mệt mỏi, ít ngủ' -> Thận âm hư của Đái tháo nhạt bị
    # loại, core rơi về Huyết hư ngoại lai + Mục 5 trắng). Thêm dấu ĐÀM-THẤP (lưỡi bệu/rêu nhớt/rêu dày)
    # để cổng vẫn bắt ca đàm-thấp bị ép âm-hư (vd Nhĩ minh lưỡi bệu/rêu nhớt) mà KHÔNG cần dựa vào 'mệt'.
    _AMHU_HUHAN_SIGNS = (
        "người nặng nề", "nặng nề", "nặng đầu", "không muốn hoạt động",
        "sợ lạnh", "sợ gió", "tay chân lạnh", "chân tay lạnh", "rêu trắng", "rêu lưỡi trắng",
        "lưỡi nhợt", "lưỡi nhạt", "lưỡi hồng nhạt", "chất lưỡi nhợt", "đàm", "đờm", "phù",
        "lưỡi bệu", "lưỡi to bè", "rêu nhớt", "rêu dày", "rêu dính", "rêu nhờn",
        "đại tiện lỏng", "tiểu trong", "nước tiểu trong", "hụt hơi", "đoản khí",
    )

    @staticmethod
    def _is_amhu_syndrome(name: str) -> bool:
        nl = (name or "").lower()
        return ("âm hư" in nl or "âm hỏa" in nl or "âm hoả" in nl) and "dương" not in nl

    def _demote_amhu_without_heat(self, syndromes: list, case_text: str):
        """Khi core là âm-hư mà lời khai KHÔNG có dấu nhiệt + CÓ dấu hư-hàn/thấp -> LOẠI HẲN mọi hội
        chứng âm-hư (ca này không phải âm hư). Phải loại hẳn chứ không chỉ hạ xuống kèm-theo: Đờm
        trọc/thực lên core thì logic Bản-Tiêu sẽ kéo âm-hư (Hư duy nhất) TRỞ LẠI làm Bản -> 'Nhiệt'
        oan vẫn còn + Mục 3 vẫn biện âm hư. Trả (danh sách mới, lý do|None)."""
        if not syndromes:
            return syndromes, None
        core = syndromes[0]
        if not self._is_amhu_syndrome(core):
            return syndromes, None
        t = (case_text or "").lower()
        if any(k in t for k in self._AMHU_HEAT_SIGNS):
            return syndromes, None                       # có dấu nhiệt/khô -> âm hư có thể đúng
        if not any(k in t for k in self._AMHU_HUHAN_SIGNS):
            return syndromes, None                       # không có dấu hư-hàn -> không đủ cơ sở, để yên
        kept = [s for s in syndromes if not self._is_amhu_syndrome(s)]
        if not kept:
            return syndromes, None                       # toàn âm-hư -> không có gì thay, để yên
        removed = [s for s in syndromes if self._is_amhu_syndrome(s)]
        return kept, (f"loại hội chứng âm-hư {removed} (lời khai không dấu nhiệt + có dấu hư-hàn/thấp) "
                      f"-> core '{kept[0]}'")

    # ------ CỔNG THERMAL-POLARITY: hạ core NHIỆT khi lời khai toàn dấu HÀN ------
    # Song song cổng âm-hư nhưng cho trục HÀN/NHIỆT ngoại cảm+nội. Quét KG: 90% (136/151) hội chứng
    # NHIỆT thuần (không dấu lạnh) — NHƯNG phong nhiệt SỚM có thể kèm 恶寒/sợ lạnh chính đáng -> luật
    # "có dấu lạnh -> không nhiệt" KHÔNG chuẩn. Discriminator an toàn = "KHÔNG có dấu nhiệt NÀO":
    # phong nhiệt THẬT gần như luôn có họng đỏ/sốt/khát/rêu vàng. Nếu core NHIỆT mà lời khai 0 dấu
    # nhiệt + CÓ dấu hàn (tay chân lạnh/rêu trắng) -> gần chắc là HÀN -> loại nhiệt, chọn non-nhiệt.
    _NHIET_HEAT_SIGNS = (
        "sốt", "phát nhiệt", "triều nhiệt", "cốt chưng", "khát", "khô họng", "họng khô", "khô miệng",
        "miệng khô", "lưỡi đỏ", "chất lưỡi đỏ", "đầu lưỡi đỏ", "rêu vàng", "rêu lưỡi vàng", "gò má đỏ",
        "má đỏ", "mặt đỏ", "bốc hỏa", "ngũ tâm phiền nhiệt", "lòng bàn tay nóng", "nóng trong",
        "phiền nhiệt", "tâm phiền", "nước tiểu vàng", "tiểu vàng", "nước tiểu đỏ", "táo bón",
        "đại tiện táo", "họng đỏ", "họng sưng", "sưng đau họng", "đau rát họng", "amidan sưng",
        "hạnh nhân sưng", "đờm vàng", "đờm đặc vàng", "mũi vàng", "nước mũi vàng", "sưng đỏ",
        "sưng nóng đỏ", "mụn đỏ", "mụn viêm", "nóng rát", "nhiệt miệng", "loét miệng",
    )
    _NHIET_COLD_SIGNS = (
        "tay chân lạnh", "chân tay lạnh", "tay chân quyết lạnh", "chi lạnh", "sợ lạnh", "úy hàn",
        "người lạnh", "rêu trắng", "rêu lưỡi trắng", "lưỡi nhợt", "chất lưỡi nhợt", "đờm trắng",
        "đờm loãng trắng", "đờm loãng", "nước mũi trong", "chảy nước mũi trong", "tiểu trong",
        "nước tiểu trong", "đại tiện lỏng",
    )

    @staticmethod
    def _is_nhiet_syndrome(name: str) -> bool:
        nl = (name or "").lower()
        # có 'nhiệt/hỏa' trong tên; loại thể HÀN-NHIỆT-tạp ('hàn' trong tên) và âm-hư (đã có cổng riêng)
        return (("nhiệt" in nl or "hỏa" in nl or "hoả" in nl)
                and "hàn" not in nl and "âm hư" not in nl and "âm hoả" not in nl)

    def _demote_nhiet_without_heat(self, syndromes: list, case_text: str):
        """Core NHIỆT mà lời khai KHÔNG có dấu nhiệt + CÓ dấu hàn -> LOẠI HẲN hội chứng nhiệt (giữ
        >=1 non-nhiệt). Trả (danh sách mới, lý do|None)."""
        if not syndromes:
            return syndromes, None
        core = syndromes[0]
        if not self._is_nhiet_syndrome(core):
            return syndromes, None
        t = (case_text or "").lower()
        if any(k in t for k in self._NHIET_HEAT_SIGNS):
            return syndromes, None                       # có dấu nhiệt -> nhiệt có thể đúng
        if not any(k in t for k in self._NHIET_COLD_SIGNS):
            return syndromes, None                       # không dấu hàn -> không đủ cơ sở, để yên
        kept = [s for s in syndromes if not self._is_nhiet_syndrome(s)]
        if not kept:
            return syndromes, None                       # toàn nhiệt -> không có gì thay
        removed = [s for s in syndromes if self._is_nhiet_syndrome(s)]
        return kept, (f"loại hội chứng NHIỆT {removed} (lời khai không dấu nhiệt + có dấu hàn) "
                      f"-> core '{kept[0]}'")

    def _matched_terms_by_syndrome(self, terms: list, syndromes: list) -> dict:
        """{syndrome: [term...]} — term khớp (ranh giới từ + bắc cầu nhóm đồng nghĩa như chấm điểm)
        với >=1 biểu hiện của hội chứng. Nuôi ĐỒ THỊ LẬP LUẬN: chỉ nối triệu chứng THẬT SỰ khớp
        vào hội chứng, thay vì nối tất cả vào tất cả."""
        terms = [t for t in dict.fromkeys(terms or []) if t and t.strip()]
        syndromes = [s for s in (syndromes or []) if s]
        if not terms or not syndromes:
            return {}
        pairs = []
        for t in terms:
            tl = t.strip().lower()
            _g = next((g for g in self._SCORING_TERM_GROUPS if tl in g), None)
            if _g:
                _alts = "|".join(r'\Q' + x + r'\E' for x in sorted(_g))
                pat = r'.*(^|[^\p{L}])(' + _alts + r')($|[^\p{L}]).*'
            else:
                pat = self.qa_pipeline._word_boundary_pattern(tl)
            pairs.append({"term": t, "pat": pat})
        cypher = """
        UNWIND $pairs AS p
        MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
        WHERE h.name IN $syns AND toLower(t.name) =~ p.pat
        RETURN h.name AS syn, collect(DISTINCT p.term) AS terms
        """
        try:
            rows = self.qa_pipeline.run_cypher(cypher, {"pairs": pairs, "syns": syndromes})
            return {r["syn"]: r["terms"] for r in rows if r.get("syn")}
        except Exception as e:
            logger.error(f"Lỗi tính triệu chứng khớp theo hội chứng: {e}")
            return {}

    def _syndromes_matching_terms(self, terms: list) -> set:
        """Tập hội chứng có >= 1 biểu hiện khớp (theo ranh giới từ) với các triệu chứng cho trước.
        Dùng làm CỔNG CHỦ CHỨNG: hội chứng kèm theo phải khớp ít nhất 1 triệu chứng bệnh nhân tự khai.
        Trả set rỗng khi không có term hoặc truy vấn lỗi (caller phải coi rỗng = không lọc)."""
        terms = [t.strip().lower() for t in (terms or []) if t and t.strip()]
        if not terms:
            return set()
        patterns = [self.qa_pipeline._word_boundary_pattern(t) for t in terms]
        cypher = """
        UNWIND $patterns AS pat
        MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
        WHERE toLower(t.name) =~ pat
        RETURN DISTINCT h.name AS name
        """
        try:
            rows = self.qa_pipeline.run_cypher(cypher, {"patterns": patterns})
            return {r["name"] for r in rows if r.get("name")}
        except Exception as e:
            logger.error(f"Lỗi truy vấn hội chứng theo chủ chứng: {e}")
            return set()

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
                            
                # [CHỐNG RÒ VÍ DỤ PROMPT] LLM bung đồng nghĩa hay CHÉP các cặp ví dụ trong prompt
                # vào ca không có triệu chứng nguồn ('tay chân lạnh', 'trướng bụng', 'đoản khí'...).
                # Từ khóa bung nằm trong bảng dưới chỉ được giữ khi triệu chứng NGUỒN thật sự có
                # trong text — nếu không sẽ thành "đầu vào" giả: bệnh danh khớp oan (Động mạch viêm
                # tắc qua 'tay chân lạnh' bịa) và Mục 3 bị ép giải thích triệu chứng không tồn tại.
                _SYNONYM_SOURCES = {
                    "huyễn vựng": ["chóng mặt", "hoa mắt", "huyễn vựng", "choáng váng", "xây xẩm"],
                    "úy hàn": ["sợ lạnh", "úy hàn"],
                    "thần bì": ["mệt", "thần bì", "uể oải"],
                    "lực kiệt": ["mệt", "kiệt sức", "lực kiệt", "vô lực", "uể oải"],
                    "tứ chi quyết nghịch": ["tay chân lạnh", "chân tay lạnh", "chi lạnh", "lạnh tay", "lạnh chân"],
                    "tứ chi bất ôn": ["tay chân lạnh", "chân tay lạnh", "chi lạnh", "lạnh tay", "lạnh chân"],
                    "chi lạnh": ["tay chân lạnh", "chân tay lạnh", "chi lạnh", "lạnh tay", "lạnh chân"],
                    "tay chân lạnh": ["tay chân lạnh", "chân tay lạnh", "chi lạnh", "lạnh tay", "lạnh chân"],
                    "ách nghịch": ["nấc", "ách nghịch"],
                    "tiết tả": ["tiêu chảy", "phân lỏng", "đại tiện lỏng", "đi lỏng", "ỉa chảy", "tiết tả", "phân nát"],
                    "đại tiện lỏng": ["tiêu chảy", "phân lỏng", "đại tiện lỏng", "đi lỏng", "ỉa chảy", "phân nát"],
                    "trướng bụng": ["đầy bụng", "trướng bụng", "bụng trướng", "chướng bụng"],
                    "bụng trướng": ["đầy bụng", "trướng bụng", "bụng trướng", "chướng bụng"],
                    "thất miên": ["mất ngủ", "khó ngủ", "thất miên", "ngủ kém", "trằn trọc", "khó vào giấc"],
                    "đoản khí": ["khó thở", "hụt hơi", "thở ngắn", "hơi thở ngắn", "đoản khí", "đoản hơi", "thiểu khí"],
                    "hơi thở ngắn": ["khó thở", "hụt hơi", "thở ngắn", "hơi thở ngắn", "đoản khí", "đoản hơi"],
                }
                _text_l2 = text.lower()
                _leaked = [k for k in keywords
                           if k in _SYNONYM_SOURCES and not any(s in _text_l2 for s in _SYNONYM_SOURCES[k])]
                if _leaked:
                    logger.info(f"[CHỐNG RÒ VÍ DỤ PROMPT] Loại từ khóa bung không có triệu chứng nguồn: {_leaked}")
                    keywords = [k for k in keywords if k not in _leaked]

                # [CHỐNG NHIỄM SYNONYM BỆNH LÝ] LLM bung đồng nghĩa đôi khi biến màu lưỡi SINH LÝ
                # thành BỆNH LÝ ('lưỡi hồng nhạt' [đạm hồng bình thường] -> tự thêm 'lưỡi nhợt'),
                # rồi bộ khử mâu thuẫn ưu tiên màu bệnh lý mà XÓA màu sinh lý thật — kết quả biện
                # chứng nói bệnh nhân 'lưỡi nhợt' dù không ai khai (censor không gỡ được vì term đã
                # thành "đầu vào"). Chỉ giữ màu lưỡi nhợt/nhạt nếu text gốc THẬT SỰ chứa nó.
                _pale_terms = {"lưỡi nhợt", "lưỡi nhạt", "chất lưỡi nhợt", "chất lưỡi nhạt",
                               "lưỡi đạm", "lưỡi sắc nhợt", "đạm bạch"}
                if not re.search(
                        r'(?<![\wÀ-ỹ])(lưỡi nhợt|lưỡi nhạt|chất lưỡi nhợt|chất lưỡi nhạt|lưỡi đạm|lưỡi sắc nhợt)(?![\wÀ-ỹ])',
                        text.lower()):
                    keywords = [k for k in keywords if k not in _pale_terms]

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
                
            # Rêu trắng dày / Thấp Hàn Rule: Rêu trắng DÀY/NHỚT biểu hiện Đàm thấp/Hàn thấp, không đi với
            # Thấp nhiệt. KHÔNG dùng 'rêu trắng'/'rêu lưỡi trắng' trần: substring dính 'rêu trắng mỏng'
            # (gần sinh lý) khiến ca rêu mỏng bị phạt oan Thấp nhiệt -20 và cộng oan Đàm thấp +20.
            has_white_thick_coating = any(kw in cq_lower for kw in [
                "rêu lưỡi trắng dày", "rêu trắng dày", "rêu trắng dầy",
                "rêu trắng nhớt", "rêu lưỡi trắng nhớt", "rêu trắng nhờn", "rêu trắng dính"])
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

    # Từ điển giải thích y lý dự phòng cho triệu chứng Thực chứng — dùng chung cho
    # _patch_missing_symptoms (vá triệu chứng sót) và _sync_tieu_thuc_with_bat_cuong (đồng bộ Mục 4).
    _THUC_TEMPLATES = {
        "nốt mụn đỏ": "Nốt mụn đỏ xuất hiện do nhiệt độc hoặc thấp nhiệt uẩn kết ở kinh lạc bốc lên bề mặt da.",
        "tiếng nấc nhanh mà không liên tục": "Tiếng nấc nhanh mà không liên tục là do Vị khí thượng nghịch, khí của Vị bốc ngược lên trên gây ra nấc.",
        "tiếng nấc": "Tiếng nấc là do Vị khí thượng nghịch, khí của Vị bốc ngược lên trên gây ra nấc.",
        "nấc": "Tiếng nấc là do Vị khí thượng nghịch, khí của Vị bốc ngược lên trên gây ra nấc.",
        "rêu lưỡi trắng dày": "Rêu lưỡi trắng dày phản ánh tình trạng đàm thấp hoặc thủy thấp tích tụ ở trung tiêu.",
        "rêu trắng dày": "Rêu lưỡi trắng dày phản ánh tình trạng đàm thấp hoặc thủy thấp tích tụ ở trung tiêu.",
        "rêu lưỡi trắng nhớt": "Rêu lưỡi trắng nhớt là dấu Thủy thấp / đàm trọc ứ đọng ở trung tiêu (yếu tố Tiêu Thực) — Tỳ vận hóa kém, thấp trọc bám lên mặt lưỡi.",
        "rêu trắng nhớt": "Rêu trắng nhớt là dấu Thủy thấp / đàm trọc ứ đọng (yếu tố Tiêu Thực), phản ánh Tỳ mất kiện vận, thấp trọc nội đình.",
        "rêu nhớt": "Rêu nhớt phản ánh đàm trọc / thủy thấp ứ đọng ở trung tiêu (yếu tố Tiêu Thực).",
        "tiêu chảy": "Tỳ dương hư, không vận hóa được thủy thấp, thanh trọc không phân, gây ra đại tiện lỏng, tiêu chảy.",
        "buồn nôn": "Vị khí nghịch lên, Tỳ vị thất hòa không thể giáng trọc khí gây ra buồn nôn.",
        "ăn kém": "Tỳ khí hư, chức năng vận hóa suy giảm, vị không thụ nạp được thức ăn gây ra ăn uống kém.",
        "đổ mồ hôi": "Vệ dương hư suy, không đủ sức cố nhiếp tân dịch, mồ hôi tự ra (tự hãn) do vệ khí bất cố.",
        # Dấu NHIỆT cục bộ trên nền hư (Hàn Nhiệt Thác Tạp) — tuyệt đối không giải thích bằng cơ chế dương hư
        "tiểu vàng": "Thấp nhiệt uất ở hạ tiêu, bức bách tân dịch khiến nước tiểu vàng sẻn — đây là yếu tố nhiệt cục bộ (Tiêu) trên nền bệnh.",
        "đờm vàng": "Đàm thấp uất lâu hóa nhiệt, nhiệt chưng luyện tân dịch thành đờm vàng đặc — yếu tố đàm nhiệt (Tiêu) cần thanh hóa.",
        "mũi vàng": "Nhiệt uất ở phế khiếu, chưng đốt tân dịch thành dịch mũi vàng đục — yếu tố nhiệt cục bộ (Tiêu) trên nền bệnh.",
    }

    @staticmethod
    def _muc4_denies_tieu_thuc(body_l: str) -> bool:
        """Thân Mục 4 có PHẢI câu phủ nhận Tiêu Thực không? Chỉ True khi:
          - chứa cụm 'không có tiêu thực' (bắt cả biến thể có tiền tố: 'Hiện tại/Nhìn chung không có Tiêu Thực...'), HOẶC
          - là câu cụt 'Không có.' đứng một mình.
        KHÔNG tính câu phân tích Thực hợp lệ mở đầu bằng 'Không có ngoại tà xâm nhập, song đàm trọc...'
        (điều kiện cũ startswith('không có') nuốt nhầm cả câu này rồi xóa mất giải thích triệu chứng)."""
        body_l = (body_l or "").lower().strip().lstrip("-*• ").strip()
        if re.search(r"không\s+có\s+tiêu\s+thực", body_l):
            return True
        if re.fullmatch(r"không\s+có[\s.]*", body_l):
            return True
        return False

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
        thuc_templates = self._THUC_TEMPLATES

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
            # Chỉ GHI ĐÈ khi thân mục 4 thực sự là câu phủ nhận Tiêu Thực (kể cả biến thể có tiền tố
            # 'Hiện tại không có Tiêu Thực...'); nếu là phân tích hợp lệ thì APPEND để không nuốt nội dung.
            # startswith('không có') cũ vừa bỏ sót biến thể có tiền tố (-> nối template sau câu chối,
            # tự mâu thuẫn) vừa nuke nhầm câu 'Không có ngoại tà..., song...'.
            _p3_body = part3[len(tieu_thuc_marker):].strip().lstrip("-*• ").strip().lower()
            if self._muc4_denies_tieu_thuc(_p3_body):
                part3 = tieu_thuc_marker + "\n- " + " ".join(missing_thuc_patches) + "\n"
            else:
                part3 = part3.rstrip() + " " + " ".join(missing_thuc_patches) + "\n"
                
        return part1 + part2 + part3

    def _strip_unfounded_cold_mechanism(self, llm_text: str, bat_cuong_hint: str,
                                        primary: str, concurrent: str, symptoms_str: str) -> str:
        """[NHẤT QUÁN HÀN] LLM đôi khi bịa cơ chế 'Âm hàn ngưng trệ' / 'hàn ngưng' để giải thích đau
        đầu/đau nhức trong ca THUẦN HƯ không hề có căn cứ Hàn (vd Huyết hư: 'kinh mạch vùng đầu cổ bị
        Âm hàn ngưng trệ' — sai; đau đầu huyết hư là do huyết không nuôi dưỡng được thanh khiếu). Đây
        là dao động ngẫu nhiên của LLM (temp>0) nên chốt bằng lưới TẤT ĐỊNH: CHỈ viết lại cụm hàn-ngưng
        khi KHÔNG có bất kỳ căn cứ Hàn nào — Bát Cương không 'Hàn', hội chứng đã chốt không 'Hàn'/'dương
        hư', lời khai không dấu lạnh. Ở ca có Hàn thật (phong hàn, dương hư…) hàm này KHÔNG đụng tới."""
        if not llm_text:
            return llm_text
        bc = (bat_cuong_hint or "").lower()
        syn = ((primary or "") + " " + (concurrent or "")).lower()
        sym = (symptoms_str or "").lower()
        if ("hàn" in bc or "hàn" in syn or "dương hư" in syn
                or any(k in sym for k in ("sợ lạnh", "tay chân lạnh", "chân tay lạnh",
                                          "úy hàn", "sợ rét", "lạnh người", "người lạnh"))):
            return llm_text  # có căn cứ Hàn -> giữ nguyên biện luận
        before = llm_text
        _repl = "khí huyết hư nhược không nuôi dưỡng được"
        # 'âm hàn ngưng trệ' / 'hàn (tà) ngưng trệ|kết|tụ' (± 'huyết ứ') -> cơ chế hư
        llm_text = re.sub(r'(?i)(?:âm\s+)?hàn\s+(?:tà\s+)?ngưng\s+(?:trệ|kết|tụ)(?:\s+huyết\s+ứ)?',
                          _repl, llm_text)
        llm_text = re.sub(r'(?i)(?:âm\s+)?hàn\s+ngưng\b', _repl, llm_text)
        # 'do hàn tà ...' / 'bởi hàn tà ...' -> do khí huyết hư nhược
        llm_text = re.sub(r'(?i)\b(do|bởi|vì)\s+hàn\s+tà\b', r'\1 khí huyết hư nhược', llm_text)
        if llm_text != before:
            logger.info("[NHẤT QUÁN HÀN] Viết lại cơ chế 'hàn ngưng' bịa trong ca không có căn cứ Hàn.")
        return llm_text

    def _strip_unfounded_heat_mechanism(self, llm_text: str, bat_cuong_hint: str,
                                        primary: str, concurrent: str, symptoms_str: str) -> str:
        """[NHẤT QUÁN NHIỆT] MIRROR của _strip_unfounded_cold_mechanism cho trục NHIỆT. LLM hay viện
        'âm hư sinh nội nhiệt / nhiệt bức tân dịch' để giải thích MỒ HÔI (đạo hãn/tự hãn) trong ca
        THUẦN HƯ-HÀN không hề có căn cứ Nhiệt (vd core Khí hư/Dương hư + Bát Cương Hàn: mồ hôi là TỰ
        HÃN do VỆ KHÍ BẤT CỐ, không phải hư nhiệt bức tân dịch — viện nội nhiệt là trái Bát Cương Hàn
        và trái bài ôn dương). Chốt bằng lưới TẤT ĐỊNH: CHỈ viết lại khi KHÔNG có bất kỳ căn cứ Nhiệt
        nào — Bát Cương không 'Nhiệt', core không nhiệt/âm-hư, lời khai không dấu nhiệt. Ca âm-hư/nhiệt
        thật (Bát Cương Nhiệt) hàm này KHÔNG đụng."""
        if not llm_text:
            return llm_text
        bc = (bat_cuong_hint or "").lower()
        syn = ((primary or "") + " " + (concurrent or "")).lower()
        sym = (symptoms_str or "").lower()
        if ("nhiệt" in bc or self._syndrome_thermal_sign(primary) == "nhiet" or "âm hư" in syn
                or "âm hoả" in syn or "âm hỏa" in syn
                or any(k in sym for k in ("sốt", "khát", "rêu vàng", "rêu lưỡi vàng", "họng đỏ",
                                          "họng sưng", "mặt đỏ", "gò má đỏ", "đờm vàng", "tiểu vàng",
                                          "nước tiểu vàng", "lưỡi đỏ", "ngũ tâm phiền nhiệt", "táo bón"))):
            return llm_text  # có căn cứ Nhiệt -> giữ nguyên biện luận
        before = llm_text
        # 'âm hư sinh/gây/làm nội nhiệt' -> khí (dương) hư khiến vệ biểu bất cố
        llm_text = re.sub(r'(?i)âm\s+hư\s+(?:sinh(?:\s+ra)?|gây(?:\s+ra)?|làm|dẫn\s+đến)?\s*nội\s+nhiệt',
                          "khí (dương) hư khiến vệ biểu bất cố", llm_text)
        # 'nhiệt bức tân dịch (tiết ra ngoài)' -> tân dịch không được cố nhiếp mà tự thoát
        llm_text = re.sub(r'(?i)nhiệt\s+bức\s+(?:tân\s+dịch|mồ\s+hôi|tấu\s+lý)\s*(?:tiết(?:\s+ra\s+ngoài)?|thoát(?:\s+ra)?)?',
                          "tân dịch không được cố nhiếp mà tự thoát", llm_text)
        # 'nội nhiệt' còn sót -> khí (dương) hư không cố nhiếp
        llm_text = re.sub(r'(?i)\bnội\s+nhiệt\b', "khí (dương) hư không cố nhiếp", llm_text)
        if llm_text != before:
            logger.info("[NHẤT QUÁN NHIỆT] Viết lại cơ chế 'âm hư nội nhiệt' bịa trong ca không căn cứ Nhiệt.")
        return llm_text

    def _sync_tieu_thuc_with_bat_cuong(self, llm_text: str, bat_cuong_hint: str, symptoms_str: str) -> str:
        """[ĐỒNG BỘ BÁT CƯƠNG <-> MỤC 4] Bát Cương ở Mục 2 là kết quả deterministic (đồ thị + từ khóa)
        còn thân Mục 4 do LLM viết, nên hai bên thỉnh thoảng vênh nhau theo cả 2 chiều:
          - Bát Cương thuần Hư mà Mục 4 vẫn có phân tích Tà khí (LLM bịa hoặc bị patch chèn vào).
          - Bát Cương có 'Bản Hư Tiêu Thực'/'Thực' mà Mục 4 lại chốt 'Không có Tiêu Thực'.
        Chốt bằng code sau mọi bước hậu xử lý: Mục 4 phải nói cùng chiều với Bát Cương đã hiển thị."""
        m = re.search(r"### 4\.[^\n]*", llm_text)
        if not m:
            return llm_text
        marker, idx = m.group(0), m.start()
        head, body = llm_text[:idx], llm_text[idx + len(marker):]

        hint_l = (bat_cuong_hint or "").lower()
        # \b để 'hư' không dính trong 'chưa rõ' ('c-hư-a')
        hint_has_hu = bool(re.search(r"\bhư\b", hint_l))
        # 'Hàn Nhiệt Thác Tạp' = CÓ tà khí (nhiệt/hàn cục bộ) — phải coi như có Tiêu, nếu không
        # chiều 1 sẽ ép Mục 4 về 'thuần Hư' mâu thuẫn với chính nhãn Thác Tạp (đã xảy ra thật).
        hint_has_thuc = bool(re.search(r"\bthực\b", hint_l)) or ("thác tạp" in hint_l)
        body_stripped = body.strip().lstrip("-*• ").strip()
        body_l = body_stripped.lower()
        if body_l.startswith("không xác định"):
            return llm_text  # fallback lỗi LLM, không có gì để đồng bộ
        body_says_none = self._muc4_denies_tieu_thuc(body_l)

        # Chiều 1: Bát Cương thuần Hư nhưng Mục 4 có nội dung -> dồn nội dung về Mục 3
        # (giữ luật phủ lấp 100% triệu chứng) rồi trả Mục 4 về câu chuẩn thuần Hư.
        if hint_has_hu and not hint_has_thuc and body_stripped and not body_says_none:
            prose = re.sub(r"\s+", " ", re.sub(r"^[\s\-\*•]+", "", body, flags=re.MULTILINE)).strip()
            logger.info("[ĐỒNG BỘ MỤC 4] Bát Cương thuần Hư nhưng Mục 4 có phân tích -> dồn về Mục 3.")
            head = self._append_prose_to_muc3(head, prose)
            return head.rstrip() + "\n\n" + marker + "\n- Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy.\n"

        # Chiều 2: Bát Cương khẳng định có Thực nhưng Mục 4 chối 'Không có' -> dựng lại thân
        # Mục 4 deterministic từ template Thực chứng khớp với triệu chứng thực sự có mặt.
        if hint_has_thuc and body_says_none:
            symptoms_l = (symptoms_str or "").lower()
            # Khớp key template; bỏ key CON nằm trong key dài hơn ('tiếng nấc' ⊂ 'tiếng nấc nhanh
            # mà không liên tục') để không chèn 2 câu gần trùng cho cùng một triệu chứng.
            matched = [(k, t) for k, t in self._THUC_TEMPLATES.items() if k in symptoms_l]
            matched = [(k, t) for k, t in matched
                       if not any(k != k2 and k in k2 for k2, _ in matched)]
            parts, seen = [], set()
            for _k, tmpl in matched:
                if tmpl not in seen:
                    parts.append(tmpl)
                    seen.add(tmpl)
            if not parts:
                # Không khớp template nào -> câu chung chung, nhưng KHÔNG khẳng định 'Bản Hư' nếu
                # Bát Cương là Thực thuần túy (không có chữ Hư) — tránh mâu thuẫn với Mục 3.
                if hint_has_hu:
                    parts = ["Trên nền chính khí hư suy (Bản Hư), định vị Bát Cương cho thấy còn tồn tại "
                             "yếu tố Thực (tà khí/đàm thấp ứ trệ) chưa được giải quyết; các biểu hiện "
                             "liên quan đã được biện giải ở phần trên."]
                else:
                    parts = ["Định vị Bát Cương cho thấy còn yếu tố Thực (tà khí/đàm thấp ứ trệ) "
                             "chi phối bệnh cảnh; các biểu hiện liên quan đã được biện giải ở phần trên."]
            logger.info("[ĐỒNG BỘ MỤC 4] Bát Cương có Thực nhưng Mục 4 ghi 'Không có' -> dựng lại từ template.")
            return head.rstrip() + "\n\n" + marker + "\n- " + " ".join(parts) + "\n"

        return llm_text

    @staticmethod
    def _append_prose_to_muc3(head: str, prose: str) -> str:
        """Đưa prose (nội dung Thực bị đặt nhầm ở Mục 4) vào Mục 3. Nếu thân Mục 3 rỗng hoặc chỉ là
        câu stub ('Không có Bản Hư...', 'Không xác định') thì THAY luôn thân bằng prose; nếu đã có
        phân tích thật thì nối tiếp — TUYỆT ĐỐI không dán prose vào dòng tiêu đề '### 3.' (bug cũ
        head.rstrip()+prose biến cả đoạn thành một H3 khổng lồ khi thân Mục 3 rỗng)."""
        m3 = re.search(r"### 3\.[^\n]*", head)
        if not m3:  # không thấy Mục 3 -> nối an toàn ở cuối (không có heading để hỏng)
            return head.rstrip() + f"\n- {prose}\n"
        pre3, marker3 = head[:m3.start()], m3.group(0)
        body3 = head[m3.end():]
        body3_l = body3.strip().lstrip("-*• ").strip().lower()
        is_stub = (not body3_l) or body3_l.startswith("không có bản hư") or body3_l.startswith("không xác định")
        if is_stub:
            return pre3 + marker3 + "\n- " + prose + "\n"
        return pre3 + marker3 + body3.rstrip() + " Ngoài ra, " + prose + "\n"

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
        # [FIX THỨ BẬC THEO BẰNG CHỨNG] Chỉ khử hội chứng CON khi hội chứng cha-tổ-hợp đứng TRƯỚC
        # nó trong danh sách (danh sách đã xếp theo bằng chứng giảm dần). Bản cũ khử VÔ ĐIỀU KIỆN:
        # 'Khí huyết đều hư' hạng 5 (3.06 điểm) xóa luôn 'Huyết hư' hạng 1 (5.33 điểm) khiến cốt lõi
        # rơi vào tay hội chứng hạng 3 chỉ khớp 2 triệu chứng (đã xảy ra thật với ca chóng mặt +
        # đau đầu + mặt nhợt + da khô -> chốt sai 'Tỳ thận dương hư' thay vì 'Huyết hư').
        lower_index = {}
        for i, s in enumerate(syndromes):
            lower_index.setdefault(s.lower().strip(), i)   # giữ vị trí ĐẦU (hạng cao nhất) nếu trùng tên
        to_remove = set()
        for parent, children in redundancy_map.items():
            p_idx = lower_index.get(parent.lower().strip())
            if p_idx is None:
                continue
            for child in children:
                c_idx = lower_index.get(child.lower().strip())
                if c_idx is not None and c_idx > p_idx:
                    to_remove.add(syndromes[c_idx])
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
        # [CHUẨN NARRATIVE — TRIỆU CHỨNG GỐC] symptoms_arr nuôi prompt Mục 3-4 (LUẬT PHỦ LẤP 100%),
        # bộ censor và khớp bệnh danh trong hàm này -> PHẢI là triệu chứng GỐC (lời khai khớp DB
        # trực tiếp + vọng chẩn), KHÔNG dùng search_terms đã bung synonym: từ bung thiếu căn cứ sẽ
        # bị luật phủ lấp ép LLM "giải thích" như triệu chứng thật và được censor bảo vệ như đầu vào.
        s_standardized = self.qa_pipeline._preprocess_question(user_symptoms) if user_symptoms else []
        if detected_symptoms:
            s_standardized.extend(detected_symptoms)
        if not s_standardized and search_terms:
            s_standardized.extend(search_terms)
        if not s_standardized and user_symptoms:
            s_standardized.extend([s.strip() for s in user_symptoms.split(",") if s.strip()])

        symptoms_arr = self._resolve_symptom_conflicts(list(dict.fromkeys(s_standardized)))
        
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
        # [ỔN ĐỊNH MÀNG LỌC] symptoms_arr giờ đã là triệu chứng GỐC ổn định (xem trên) — các màng
        # lọc gold-standard + luật Bát Cương dùng chung nền này, không còn biến thiên theo lần chạy.
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
        # [GỠ HARDCODE — grounded thay thế] 3 case Vị hàn / Đờm Trọc Ngưng Kết(âm hành) / Phong hàn(bế
        # kinh) đã bỏ: grounded scoring tự sinh đúng hội chứng cốt lõi (kiểm chứng trên Neo4j live
        # + scripts/test_hardcode_vs_grounded.py). Biện chứng Mục 3-4 nay do LLM sinh như mọi ca thường.
        import re
        # Loại trừ khi có dấu HƯ MẠN (đồng bộ _chronic_hu_signs của run_diagnosis): 'mệt mỏi lâu
        # ngày + đoản khí + sợ lạnh + sổ mũi' là HƯ NHÂN CẢM MẠO — không được ép thành phong hàn
        # thuần biểu (mất hội chứng hư nền, Mục 3 nói 'không có Bản Hư' sai với bệnh cảnh).
        # [PHÂN BIỆT PHONG HÀN vs PHONG NHIỆT] 'Sợ gió/sợ lạnh' lúc mới cảm có ở CẢ hai thể ngoại
        # cảm; yếu tố phân định là DẤU NHIỆT. Ca có khát nước, họng đau/đỏ/sưng, nước mũi\đờm VÀNG
        # đặc, rêu VÀNG, chất lưỡi ĐỎ... là PHONG NHIỆT phạm biểu — TUYỆT ĐỐI không được ép thành
        # 'Phong hàn phạm biểu' (kéo theo pháp trị tân ôn giải biểu ngược chứng, và dán nhãn Bát
        # Cương 'Hàn' hoặc 'Hàn Nhiệt Thác Tạp' giả). Phong hàn thật: nước mũi TRONG, không khát,
        # rêu TRẮNG, không đau họng.
        _phong_nhiet_signs = [
            "khát nước", "khát", "họng đau", "đau họng", "rát họng", "đau rát họng", "hầu thống",
            "họng đỏ", "họng sưng", "sưng họng", "viêm họng", "yết hầu sưng",
            "nước mũi vàng", "mũi vàng", "đờm vàng", "rêu vàng", "rêu lưỡi vàng",
            "lưỡi đỏ", "rìa lưỡi đỏ", "chất lưỡi đỏ", "đầu lưỡi đỏ", "biên lưỡi hồng đỏ",
        ]
        has_phong_nhiet_sign = self._kw_hit_clean(symptoms_lower, _phong_nhiet_signs)
        is_ngoai_cam_phong_han_case = (
            any((x in symptoms_lower if x != "ho" else bool(re.search(r'\bho\b', symptoms_lower))) for x in ["sổ mũi", "chảy nước mũi", "ngạt mũi", "hắt hơi", "ho"]) and
            any(x in symptoms_lower for x in ["sợ lạnh", "sợ gió", "rét run"]) and
            not has_phong_nhiet_sign and
            not any(x in symptoms_lower for x in ["bệnh lâu ngày", "mãn tính", "lâu ngày", "đau lưng mỏi gối",
                                                  "mạch vi nhược", "tiểu đêm", "đoản khí", "hụt hơi",
                                                  "hay cảm", "dễ cảm", "tái phát", "gầy sút", "tự hãn"])
        )
        
        if is_an_duong_case:
            all_syndromes = ["Tỳ khí hư", "Khí huyết đều hư", "Thận âm hư", "Can khí uất kết"]
        elif is_ban_hu_tieu_thuc_case:
            all_syndromes = ["Tỳ khí hư", "Khí huyết đều hư", "Thấp nhiệt"]
        elif is_ngoai_cam_phong_han_case:
            all_syndromes = ["Phong hàn"]
        else:
            all_syndromes = [item["syndrome"] for item in detailed_kg_data]
            
        all_syndromes = self._filter_hierarchical_redundancies(all_syndromes)
        # [CỔNG ÂM-HƯ KHÔNG NHIỆT] GUARD RULE 4 deterministic: core âm-hư mà lời khai KHÔNG có dấu
        # nhiệt/khô (sốt/khát/lưỡi đỏ/rêu vàng/gò má đỏ/ngũ tâm phiền nhiệt...; 'mồ hôi trộm' đơn độc
        # KHÔNG tính) VÀ CÓ dấu hư-hàn/thấp -> hạ bậc, chọn ứng viên non-âm-hư kế. Âm hư THẬT gần như
        # luôn kèm dấu nhiệt (87% KG) nên cổng không đụng nhầm. Chèn TRƯỚC final_primary để định vị
        # bệnh danh + Bát Cương + Mục 5 dùng core đã sửa.
        all_syndromes, _amhu_reason = self._demote_amhu_without_heat(all_syndromes, symptoms_lower)
        if _amhu_reason:
            logger.info(f"[CỔNG ÂM-HƯ KHÔNG NHIỆT] {_amhu_reason}")
        # [CỔNG THERMAL-POLARITY] core NHIỆT (nhiệt/hỏa/phong nhiệt) mà lời khai 0 dấu nhiệt + có dấu
        # hàn (tay chân lạnh/rêu trắng) -> loại nhiệt, chọn non-nhiệt (vd Phong nhiệt phạm phế -> Phong
        # hàn cho ca rêu trắng + tay chân lạnh). Discriminator '0 dấu nhiệt' để không đụng phong-nhiệt thật.
        all_syndromes, _nhiet_reason = self._demote_nhiet_without_heat(all_syndromes, symptoms_lower)
        if _nhiet_reason:
            logger.info(f"[CỔNG THERMAL-POLARITY] {_nhiet_reason}")
        # Các Guard rules đặc biệt - Khởi tạo sớm để tránh lỗi UnboundLocalError
        overridden = False
        final_primary = all_syndromes[0] if all_syndromes else "Chưa rõ"
        # [KHỬ KÈM-THEO TRÙNG Ý] Bỏ qua ứng viên kèm theo mà tên CHỨA hoặc BỊ CHỨA trong tên cốt
        # lõi (cha-con/đồng nghĩa mở rộng: 'Thận dương hư' + 'Tỳ thận dương hư', 'Phế khí hư' +
        # 'Khí hư') — hiển thị cặp đó là lặp cùng một ý chẩn đoán. Lấy ứng viên khác họ đầu tiên.
        final_concurrent = "Không có"
        _fp_l = final_primary.lower().strip()
        _fp_fold = _fold_vn(final_primary)
        for _cand in all_syndromes[1:]:
            _cl = _cand.lower().strip()
            _cl_fold = _fold_vn(_cand)
            # Trùng cùng ý cốt lõi: khớp theo chuỗi thường HOẶC theo bản bỏ dấu (chặn cặp chỉ khác
            # cách đặt dấu thanh như 'Can Hoả Phạm Phế' vs 'Can hỏa phạm phế' — vốn lọt lưới cũ).
            if (_cl in _fp_l or _fp_l in _cl or _cl_fold in _fp_fold or _fp_fold in _cl_fold):
                continue
            final_concurrent = _cand
            break
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
            # Soi TOÀN BỘ hội chứng của bệnh (hoi_chung_all — sau khử trùng mỗi bệnh 1 ứng viên)
            #
            # [FIX CỬA SỔ-TRƯỚC] Cửa sổ ratio phải tính theo match TỐT NHẤT TOÀN CỤC (chief complaint
            # mạnh nhất) TRƯỚC, rồi mới ưu tiên hội chứng TRONG cửa sổ. Bản cũ lọc chéo hội chứng
            # TRƯỚC rồi mới lấy cửa sổ trong tập đã lọc -> một bệnh khớp-hội-chứng nhưng ratio thua
            # xa (khớp toàn dấu thể trạng/lưỡi chung: 0.50) SOÁN NGÔI match chief-complaint thật
            # (chảy mũi/ho: 0.75) chỉ vì trùng 1 âm tiết hội chứng ('khí'/'tỳ' của "Phế khí hư" nối
            # sang "Khí trệ huyết ứ"/"Tỳ hư..."). Đã xảy ra thật: ca 'chảy nước mũi, sổ mũi, ho' ra
            # bệnh danh 'Trưng hà, Long bế, Tĩnh mạch viêm tắc' (u bụng/bí tiểu/viêm tĩnh mạch).
            _global_best = matched_diseases[0]["ratio"] if matched_diseases else 0.0
            _window = [
                m for m in matched_diseases
                if m["ratio"] >= _global_best - 0.15 and m["ratio"] >= 0.30
            ]
            # [CỔNG GROUNDING CORE] Ground lại core NGOẠI LAI (hội chứng promiscuous của bệnh khác
            # thắng oan) về ứng viên grounded ở cửa sổ chief-complaint. Chèn TRƯỚC core_matched để
            # toàn bộ định vị bệnh danh + Bát Cương + Mục 5 dùng core đã ground. No-op nếu core vốn
            # grounded (không đụng ca tốt).
            _new_core, _reg_reason = self._reground_core(final_primary, all_syndromes, _window)
            if _reg_reason:
                logger.info(f"[CỔNG GROUNDING CORE] {_reg_reason}")
                final_primary = _new_core
                # Re-chọn hội chứng kèm theo (loại trùng ý với core mới) + cập nhật valid_syndromes.
                _fpl = final_primary.lower().strip()
                _fpf = _fold_vn(final_primary)
                final_concurrent = "Không có"
                for _c in all_syndromes:
                    if _c.lower().strip() == _fpl:
                        continue
                    _clx = _c.lower().strip()
                    _cfx = _fold_vn(_c)
                    if (_clx in _fpl or _fpl in _clx or _cfx in _fpf or _fpf in _cfx):
                        continue
                    final_concurrent = _c
                    break
                valid_syndromes = [final_primary.lower().strip()]
                if final_concurrent and final_concurrent != "Không có":
                    valid_syndromes.append(final_concurrent.lower().strip())
            # [CỔNG ÂM-HƯ/THERMAL SAU RE-GROUND] _reground_core có thể ÉP core về thể âm-hư/nhiệt của
            # bệnh danh cho ca KHÔNG dấu nhiệt (vd Nhĩ minh chỉ có thể 'Can thận âm hư' -> ép core 'Âm
            # hư' dù lưỡi bệu/rêu nhớt/mặt nhợt = đàm-thấp/khí-huyết-hư). Cổng chạy sớm không bắt vì
            # âm-hư được promote SAU nó. Re-áp: core mới âm-hư/nhiệt + lời khai 0 dấu nhiệt + có hư-hàn
            # -> hạ sang ứng viên non-âm-hư/non-nhiệt (chấp nhận core 'ngoại lai' bệnh danh — đúng cực
            # nhiệt quan trọng hơn grounded sai cực). Chỉ đụng ca no-heat; âm-hư/nhiệt THẬT không sao.
            _pg = [final_primary] + [s for s in all_syndromes
                                     if s.lower().strip() != final_primary.lower().strip()]
            _pg, _pgr1 = self._demote_amhu_without_heat(_pg, symptoms_lower)
            _pg, _pgr2 = self._demote_nhiet_without_heat(_pg, symptoms_lower)
            if (_pgr1 or _pgr2) and _pg and _pg[0].lower().strip() != final_primary.lower().strip():
                logger.info(f"[CỔNG ÂM-HƯ/THERMAL SAU RE-GROUND] {_pgr1 or ''} {_pgr2 or ''}")
                final_primary = _pg[0]
                all_syndromes = _pg
                _fpl3 = final_primary.lower().strip()
                _fpf3 = _fold_vn(final_primary)
                final_concurrent = "Không có"
                for _c in all_syndromes:
                    _clx3 = _c.lower().strip()
                    if _clx3 == _fpl3:
                        continue
                    if (_clx3 in _fpl3 or _fpl3 in _clx3 or _fold_vn(_c) in _fpf3 or _fpf3 in _fold_vn(_c)):
                        continue
                    final_concurrent = _c
                    break
                valid_syndromes = [_fpl3] + ([final_concurrent.lower().strip()]
                                            if final_concurrent != "Không có" else [])
            core_matched = [
                m for m in _window
                if any(self._are_syndromes_related(final_primary.lower().strip(), hc)
                       for hc in m.get("hoi_chung_all", [m["hoi_chung"]]))
            ]
            any_matched = [
                m for m in _window
                if any(self._are_syndromes_related(vs, hc)
                       for vs in valid_syndromes
                       for hc in m.get("hoi_chung_all", [m["hoi_chung"]]))
            ]
            # Trong CỬA SỔ chief-complaint: ưu tiên bệnh khớp cốt lõi -> khớp hội chứng bất kỳ ->
            # còn lại giữ nguyên cửa sổ (không rơi về TOÀN BỘ matched, tránh kéo lại match ratio thấp).
            syndrome_matched_diseases = core_matched if core_matched else any_matched

            # [FIX TRUY HỒI NHẤT QUÁN] Đánh dấu bệnh danh có thực sự grounded với hội chứng đã
            # biện hay không. Nếu KHÔNG có bệnh nào (trong cửa sổ) liên quan hội chứng cốt lõi/kèm
            # theo, ta buộc phải rơi về danh sách khớp-theo-triệu-chứng (ungrounded) -> bài thuốc
            # truy hồi exact sẽ rỗng, nên phải ghi rõ để bệnh danh (Mục 1) không mâu thuẫn Mục 5.
            disease_grounded = bool(syndrome_matched_diseases)
            # Ưu tiên các bệnh khớp cả hội chứng; nếu không có, giữ chính CỬA SỔ chief-complaint.
            target_matches = syndrome_matched_diseases if syndrome_matched_diseases else _window

            if target_matches:
                filtered_matches = target_matches[:3]

                disease_names = list(dict.fromkeys([m["benh_ly"].strip() for m in filtered_matches]))
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

        # [CỔNG CÙNG-BỆNH] Loại hội chứng kèm theo lọt từ BỆNH KHÁC (vd 'Doanh vệ bất hòa' của Tiểu
        # nhi hãn chứng ghép nhầm vào Viêm yết hầu). Chèn tại đây (sau chuỗi hardcode, trước khi in
        # Mục 1) để tự lan xuống Bát Cương (Biểu-Lý đồng bệnh, L3077) và Mục 5 (active_syndromes).
        final_concurrent, _gate_reason = self._gate_concurrent_by_disease(
            final_primary, final_concurrent, matched_diseases, disease_names, overridden)
        if _gate_reason:
            logger.info(f"[CỔNG CÙNG-BỆNH] {_gate_reason}")

        final_markdown += f"- **Hội chứng cốt lõi:** {final_primary}\n"
        final_markdown += f"- **Hội chứng kèm theo (nếu có):** {final_concurrent}\n\n"

        # Thu thập thông tin Tạng Phủ và Bát Cương từ đồ thị để định hướng biện chứng.
        # [FIX NHẤT QUÁN BÁT CƯƠNG] CHỈ lấy metadata của hội chứng ĐƯỢC CHỌN (cốt lõi + kèm theo),
        # KHÔNG union cả ~6 ứng viên grounded — tránh chữ 'Nhiệt'/'Thực' của ứng viên hạng 3-6
        # (không được chọn) rò vào Bát Cương rồi kích hoạt nhầm 'Bản Hư Tiêu Thực', mâu thuẫn
        # trực tiếp với Mục 4 ('Hư chứng thuần túy').
        all_organs = set()
        all_bat_cuong = set()
        _chosen_syns = {final_primary.strip().lower()}
        if final_concurrent and final_concurrent.strip().lower() not in ("không có", "", "chưa rõ"):
            _chosen_syns.add(final_concurrent.strip().lower())
        _meta_entries = [d for d in detailed_kg_data
                         if d.get("syndrome", "").strip().lower() in _chosen_syns]
        # Hội chứng được chọn (vd bị override bởi màng lọc) không nằm trong detailed_kg_data
        # -> tra metadata trực tiếp từ đồ thị để không mất định hướng Tạng phủ/Bát cương.
        _found_syns = {d.get("syndrome", "").strip().lower() for d in _meta_entries}
        for _syn in (final_primary, final_concurrent):
            _key = (_syn or "").strip().lower()
            if _key and _key in _chosen_syns and _key not in _found_syns:
                try:
                    _meta = self.qa_pipeline.get_syndrome_metadata(_syn)
                    if _meta.get("organs") or _meta.get("bat_cuong"):
                        _meta_entries.append({"organs": _meta.get("organs", []),
                                              "bat_cuong": _meta.get("bat_cuong", [])})
                    else:
                        # Tên hội chứng (thường do màng lọc hardcode, vd 'Phong hàn phạm biểu',
                        # hoặc tên có ngoặc) KHÔNG có node graph -> log để lộ mismatch, rồi bù bằng
                        # metadata của ứng viên grounded LIÊN QUAN (thay vì bỏ trắng Tạng phủ/Bát cương).
                        logger.info(f"[BÁT CƯƠNG] '{_syn}' không có metadata node graph — bù từ ứng viên grounded liên quan.")
                        for d in detailed_kg_data:
                            _cand = d.get("syndrome", "")
                            # Cốt lõi NGOẠI CẢM đã chốt thuần biểu: không mượn tag của ứng viên
                            # NỘI THƯƠNG ('Phế khí hư hàn' [Hư, Lý]...) — rò 'Hư' vào union sinh
                            # nhãn 'Bản Hư Tiêu Thực' mâu thuẫn Mục 3 'không có Bản Hư'. Tag Biểu/
                            # Hàn/Nhiệt/Thực đã được khối seed theo tên phía dưới lo.
                            if self._syndrome_is_exterior_wind(_syn) and not self._syndrome_is_exterior_wind(_cand):
                                continue
                            # [CHỐNG RÒ NHIỆT TRÁI CỰC] Cốt lõi Phong hàn (hàn) không được mượn
                            # metadata của ứng viên Phong táo/Phong nhiệt (nhiet) — cùng là ngoại
                            # cảm biểu (_are_syndromes_related qua 'phong') nhưng tag Nhiệt của nó
                            # rò vào Bát Cương thành 'Hàn Nhiệt Thác Tạp' giả, mâu thuẫn Mục 4 thuần hàn.
                            if self._syndromes_thermal_conflict(_syn, _cand):
                                continue
                            if self._are_syndromes_related(_key, _cand) and \
                                    (d.get("organs") or d.get("bat_cuong")):
                                _meta_entries.append(d)
                                break
                except Exception as _e:
                    logger.warning(f"[BÁT CƯƠNG] Lỗi tra metadata '{_syn}': {_e}")
        for data in _meta_entries:
            for o in data.get("organs", []):
                all_organs.add(o)
            for bc in data.get("bat_cuong", []):
                all_bat_cuong.add(bc)
        
        # Check if patient symptoms have both cold and heat indicators
        # [ỔN ĐỊNH BÁT CƯƠNG] Mọi luật dò theo triệu chứng (Hàn/Nhiệt/Hư/Thực, biểu/lý, nấc, sắc mặt,
        # mạch/lưỡi, lời khuyên Mục 5) dùng chung chuỗi triệu chứng GỐC đã dựng ở symptoms_lower
        # (không synonym LLM — xem chú thích [ỔN ĐỊNH MÀNG LỌC] phía trên).
        symptoms_lower_all = symptoms_lower
        
        # 'rêu trắng' trần bị BỎ khỏi cold_kws: substring dính 'rêu trắng mỏng' (rêu SINH LÝ mỏng
        # bình thường) khiến ca lành + khô miệng bị dựng thành 'Hàn Nhiệt Thác Tạp'. Chỉ giữ rêu
        # DÀY/NHỚT (dấu hàn thấp/đàm thấp thật). Toàn bộ khớp qua _kw_hit_clean (\b + gỡ bạn-hữu-giả
        # 'sốt ruột'/'thực sự'...) — trước đây cold/heat khớp substring còn hu/thuc khớp \b nên hai
        # nhóm có thể mâu thuẫn nhau trên cùng một input.
        cold_kws = ["sợ lạnh", "úy hàn", "sợ gió", "rét run", "tay chân lạnh", "rêu lưỡi trắng dày",
                    "rêu trắng dày", "rêu trắng nhớt", "rêu lưỡi trắng nhớt", "mạch trì"]
        # 'đờm vàng/mũi vàng/vàng đục' = dấu NHIỆT quan trọng hay gặp (đàm nhiệt, tỵ uyên) — thiếu
        # chúng thì ca 'chảy mũi vàng đục + sợ lạnh' bị dựng thành 'Hư thuần túy' không dấu nhiệt.
        heat_kws = ["mạch sác", "tế sác", "sác", "mạch trầm sác", "khát nước", "sốt", "đỏ bừng", "khô miệng",
                    "họng ráo", "lưỡi đỏ", "chất lưỡi đỏ", "mụn đỏ", "nốt mụn đỏ", "rêu vàng",
                    "đờm vàng", "mũi vàng", "vàng đục", "tiểu vàng", "họng đỏ"]

        has_cold_indicator = self._kw_hit_clean(symptoms_lower_all, cold_kws)
        has_heat_pulse_indicator = self._kw_hit_clean(symptoms_lower_all, heat_kws)
        # [GÁC NGOẠI CẢM] Với cốt lõi ngoại cảm biểu, 'sợ gió/sợ lạnh/rét run' là Ố HÀN phần biểu
        # bình thường (chính-tà giao tranh ở biểu, có ở CẢ phong hàn lẫn phong nhiệt lúc mới cảm),
        # KHÔNG phản ánh nội hàn -> loại khỏi việc xét xung đột Hàn-Nhiệt. Nếu không, ca phong NHIỆT
        # phạm biểu (sốt + sợ gió + rêu vàng + lưỡi đỏ) bị dựng nhãn 'Hàn Nhiệt Thác Tạp' giả. Chỉ
        # dấu hàn NỘI thực sự (tay chân lạnh, rêu trắng dày/nhớt, mạch trì) mới tính ở ca ngoại cảm.
        if self._syndrome_is_exterior_wind(final_primary):
            _interior_cold_kws = [k for k in cold_kws if k not in ("sợ lạnh", "sợ gió", "úy hàn", "rét run")]
            has_cold_indicator = self._kw_hit_clean(symptoms_lower_all, _interior_cold_kws)
        # [DẤU NHIỆT YẾU — NƯỚC TIỂU VÀNG] Nước tiểu vàng/sẫm là dấu nhiệt YẾU, không đặc hiệu (rất hay
        # do cô đặc/mất nước, nhất là ca cấp/người trẻ). Khi nó là DẤU NHIỆT DUY NHẤT trên bệnh cảnh hư
        # hàn rõ (>=2 dấu hàn), KHÔNG đủ dựng 'Hàn Nhiệt Thác Tạp' đè lên chứng dương hư — nếu không sẽ
        # kéo LLM bịa 'Thấp nhiệt hạ tiêu' ở Mục 4. Cần một dấu nhiệt THỰC khác corroborate mới tính
        # xung đột. Gỡ dấu nhiệt yếu này -> để khối [NỘI HÀN DO DƯƠNG HƯ] phía dưới dán đúng 'Hàn'.
        _urine_heat_kws = ("tiểu vàng", "nước tiểu vàng", "tiểu sẫm", "nước tiểu sẫm")
        if has_heat_pulse_indicator and has_cold_indicator:
            _heat_nonurine = self._kw_hit_clean(symptoms_lower_all, [k for k in heat_kws if k not in _urine_heat_kws])
            _cold_n_conflict = sum(1 for _k in cold_kws if self._kw_hit_clean(symptoms_lower_all, [_k]))
            if not _heat_nonurine and _cold_n_conflict >= 2:
                logger.info(f"[DẤU NHIỆT YẾU] Nước tiểu vàng là dấu nhiệt duy nhất trên bệnh cảnh hư hàn "
                            f"({_cold_n_conflict} dấu hàn) -> không tính xung đột Hàn-Nhiệt.")
                has_heat_pulse_indicator = False
        has_yinyang_conflict = has_cold_indicator and has_heat_pulse_indicator

        hu_kws = ["nhợt", "nhợt nhạt", "mệt mỏi", "chóng mặt", "hoa mắt", "tế", "hư", "vô lực", "đau lưng", "mỏi gối", "khô miệng", "họng ráo"]
        # Rêu NHỚT/NHỜN (đàm thấp/thủy thấp trọc bám lưỡi) là dấu THỰC (Tiêu) — không tồn tại 'thuần
        # Hư' nào có rêu nhớt. Thiếu chúng thì ca lưỡi bệu+hằn răng+rêu nhớt bị chốt 'thuần Hư' trong
        # khi Mục 3-4 vẫn (đúng) nói 'Thủy thấp ứ đọng' -> nhãn tự mâu thuẫn. KHÔNG thêm 'lưỡi bệu'
        # trần (bệu nhạt đơn thuần là Tỳ khí/dương hư, thiên Hư) — chỉ RÊU nhớt mới chốt Thực.
        # 'ho' TRẦN bị gỡ khỏi thuc_kws: ho đơn thuần KHÔNG định Hư/Thực (Phế khí hư ho, Phế âm hư
        # ho khan — đều hư chứng); từng ép ca thuần Hư (ho khan + mệt mỏi + mặt nhợt) thành
        # 'Bản Hư Tiêu Thực' vô căn cứ. Chỉ ho CÓ ĐỜM mới là bằng chứng tà thực (đàm trọc).
        thuc_kws = ["mụn đỏ", "nốt mụn đỏ", "tiếng nấc", "nấc", "rêu lưỡi trắng dày", "rêu trắng dày", "rêu dày",
                    "khạc đờm", "ho có đờm", "ho đờm", "đờm nhiều", "nhiều đờm", "sốt", "thực", "hữu lực",
                    "đờm vàng", "mũi vàng", "vàng đục",
                    "rêu lưỡi trắng nhớt", "rêu trắng nhớt", "rêu nhớt", "rêu nhờn", "rêu lưỡi nhớt", "rêu vàng nhớt"]

        # Khớp theo RANH GIỚI TỪ (\b): check substring cũ khiến 'ho' dính trong 'hoa mắt' (triệu chứng
        # Hư kinh điển) -> has_thuc bật sai -> Bát Cương thành 'Bản Hư Tiêu Thực' trong khi Mục 4
        # (đúng luật thuần Hư) ghi 'Không có Tiêu Thực' -> hai mục mâu thuẫn nhau.
        # \b của Python là ranh giới KÝ TỰ, còn tiếng Việt tách âm tiết bằng dấu cách -> keyword đơn
        # âm đa nghĩa vẫn dính cụm vô hại ('sốt' trong 'sốt ruột', 'thực' trong 'thực sự'). Cụm bạn
        # hữu được gỡ sẵn trong _kw_hit_clean để không bịa Tiêu Thực cho ca thuần Hư.
        has_hu = self._kw_hit_clean(symptoms_lower_all, hu_kws)
        has_thuc = self._kw_hit_clean(symptoms_lower_all, thuc_kws)
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
        # Dấu BIỂU đặc hiệu (mũi/hầu họng ngoại cảm) tự nó đủ; còn sợ lạnh/sợ gió ĐƠN ĐỘC là úy hàn
        # nội thương thường gặp của dương hư/vệ hư — chỉ tính là biểu khi kèm dấu ngoại cảm khác
        # (mũi hoặc phát sốt/đau mình): giáo khoa định nghĩa biểu chứng bằng Ố HÀN PHÁT NHIỆT đồng thời.
        _bieu_specific = ["ngạt mũi", "hắt hơi", "sổ mũi", "chảy nước mũi", "chảy mũi", "nghẹt mũi", "rét run"]
        _bieu_cold = ["sợ gió", "sợ lạnh", "úy phong", "úy hàn"]
        # [CHỐNG BIỂU OAN CHO CORE NỘI THƯƠNG HƯ] 'sợ lạnh + đau mình' (KHÔNG sốt, KHÔNG dấu mũi họng)
        # KHÔNG phải biểu khi CỐT LÕI là nội thương HƯ: với hư core, sợ lạnh = úy hàn do dương/vệ khí
        # hư, đau mình = cơ nhục thất dưỡng (khí huyết hư) — đều LÝ. Tính là biểu -> dán 'Biểu - Hư'
        # mâu thuẫn (Khí huyết hư vốn Lý; Mục 3 tự nói 'sợ lạnh do dương khí hư'). VẪN giữ biểu cho:
        # (a) dấu mũi họng đặc hiệu (ngoại cảm chắc chắn, kể cả trên nền hư -> biểu-lý đồng bệnh);
        # (b) sợ lạnh + PHÁT SỐT đồng thời (ố hàn phát nhiệt kinh điển) — kể cả core hư.
        _core_internal_def = self._syndrome_is_hu(final_primary) \
            and not self._syndrome_is_exterior_wind(final_primary)
        _has_bieu_cold = any(kw in symptoms_lower_all for kw in _bieu_cold)
        _has_fever = any(kw in symptoms_lower_all for kw in ["sốt", "phát nhiệt"])
        _has_bieu_ache = any(kw in symptoms_lower_all for kw in ["đau mình", "mình mẩy đau", "nhức mỏi toàn thân"])
        has_bieu_indicator = (
            any(kw in symptoms_lower_all for kw in _bieu_specific)
            or (_has_bieu_cold and _has_fever)
            or (_has_bieu_cold and _has_bieu_ache and not _core_internal_def)
        )
        # Dấu LÝ: chỉ điểm nội thương/tạng phủ CỤ THỂ. KHÔNG đếm 'mệt mỏi'/'chóng mặt' — triệu chứng
        # phổ quát có cả trong cảm mạo biểu chứng cấp; bản cũ đếm chúng khiến ca thuần biểu (Phong
        # hàn phạm biểu, Mục 3 'Thực chứng thuần túy') bị dán nhãn 'Biểu - Lý đồng bệnh' tự mâu thuẫn.
        _ly_kws = ["lưỡi bệu", "rìa lưỡi có hằn răng", "mạch trầm", "tiểu đêm", "đại tiện lỏng",
                   "phân lỏng", "phân nát", "ăn kém", "đầy bụng", "đau lưng", "mỏi gối"]
        has_ly_indicator = any(x in all_bat_cuong for x in ["Lý"]) or len(all_organs) > 0 \
            or any(kw in symptoms_lower_all for kw in _ly_kws)
        # Chẩn đoán đã chốt THUẦN BIỂU (cốt lõi ngoại cảm, không hội chứng kèm theo nội thương)
        # -> tag 'Lý'/tạng phủ rơi rớt từ metadata không đủ nâng 'đồng bệnh'; chỉ dấu Lý THẬT
        # trên lời khai mới tính.
        _concurrent_noi_thuong = bool(final_concurrent) and \
            final_concurrent.strip().lower() not in ("không có", "", "chưa rõ") and \
            not self._syndrome_is_exterior_wind(final_concurrent)
        if self._syndrome_is_exterior_wind(final_primary) and not _concurrent_noi_thuong:
            has_ly_indicator = any(kw in symptoms_lower_all for kw in _ly_kws)

        # [NGOẠI CẢM] Node hội chứng ngoại cảm thường KHÔNG có tag metadata ('Phong hàn phạm biểu'
        # rỗng hoàn toàn) -> bổ sung Hàn/Nhiệt/Thực từ chính TÊN hội chứng cốt lõi để nhãn không cụt.
        if self._syndrome_is_exterior_wind(final_primary):
            _fp_l = final_primary.lower()
            all_bat_cuong.add("Thực")
            # không re-add Hàn/Nhiệt nếu ca đã chốt 'Hàn Nhiệt Thác Tạp' ở luật xung đột phía trên
            if not has_yinyang_conflict:
                if re.search(r'\bhàn\b', _fp_l):
                    all_bat_cuong.add("Hàn")
                elif re.search(r'\b(nhiệt|ôn)\b', _fp_l):
                    all_bat_cuong.add("Nhiệt")

        # [NỘI HÀN DO DƯƠNG HƯ] Hội chứng dương hư / hư hàn nội thương (Tỳ/Thận dương hư, Tỳ vị hư
        # hàn...) bản chất sinh NỘI HÀN vì dương khí suy không ôn ấm được cơ thể — nhưng node KG lắm
        # khi chỉ gắn tag 'Hư, Lý' mà thiếu 'Hàn', khiến Bát Cương cụt (Lý-Hư) không phản ánh được
        # tính hàn của bệnh. Khi cốt lõi NỘI THƯƠNG là dương hư/hư hàn VÀ có dấu hàn thật (tay chân
        # lạnh, sợ lạnh, bụng lạnh, đại tiện lỏng...) mà KHÔNG có dấu nhiệt -> bổ sung 'Hàn'.
        _fp_low = final_primary.lower()
        _core_is_cold_def = (
            (re.search(r'\bdương\b', _fp_low) and self._syndrome_is_hu(final_primary) and 'âm' not in _fp_low)
            or 'hư hàn' in _fp_low or re.search(r'\bhàn\b', _fp_low)
        )
        if (not self._syndrome_is_exterior_wind(final_primary) and _core_is_cold_def
                and has_cold_indicator and not has_heat_pulse_indicator):
            all_bat_cuong.add("Hàn")

        # [NỘI HÀN THEO BẰNG CHỨNG] Ngay cả khi cốt lõi là hội chứng HƯ TỔNG QUÁT (Khí hư/Huyết hư/
        # Khí huyết đều hư) — tên không mang 'dương'/'hàn' nên _core_is_cold_def ở trên bỏ sót — nếu
        # lời khai hội đủ NHIỀU dấu hư hàn nội thương (>=2: sợ lạnh, tay chân lạnh, đại tiện lỏng,
        # bụng lạnh, tiểu đêm, phân sống, rêu trắng dày/nhớt...) mà KHÔNG một dấu nhiệt nào, bệnh cảnh
        # là LÝ HƯ HÀN. Bổ sung 'Hàn' THEO BẰNG CHỨNG để Bát Cương không rụng trục Hàn chỉ vì nhãn
        # cốt lõi chung chung (dương khí hư suy sinh nội hàn — bằng chứng đủ mạnh mới kích hoạt).
        _interior_cold_signs = [
            "sợ lạnh", "úy hàn", "rét run", "lạnh run", "tay chân lạnh", "chân tay lạnh", "chi lạnh",
            "lưng lạnh", "bụng lạnh", "lạnh bụng", "tiểu đêm", "ngũ canh", "phân sống",
            "đại tiện lỏng", "phân lỏng", "rêu trắng dày", "rêu trắng nhớt", "mạch trì"]
        _interior_cold_n = sum(1 for _k in _interior_cold_signs
                               if self._kw_hit_clean(symptoms_lower_all, [_k]))
        if (not self._syndrome_is_exterior_wind(final_primary) and self._syndrome_is_hu(final_primary)
                and not _core_is_cold_def and _interior_cold_n >= 2
                and has_cold_indicator and not has_heat_pulse_indicator):
            logger.info(f"[NỘI HÀN THEO BẰNG CHỨNG] Cốt lõi Hư tổng quát '{final_primary}' + {_interior_cold_n} "
                        f"dấu hư hàn, không dấu nhiệt -> bổ sung 'Hàn' (Lý Hư Hàn).")
            all_bat_cuong.add("Hàn")

        # [NHIỆT THEO BẰNG CHỨNG] Nội thương có dấu NHIỆT RÕ (rêu vàng, lưỡi đỏ, mắt/mặt đỏ, khát,
        # họng đỏ, đờm\mũi\tiểu vàng...) mà KHÔNG có dấu hàn -> Bát Cương PHẢI có 'Nhiệt', kể cả khi
        # node hội chứng cốt lõi mang tag Hư/Lý và tên không chứa chữ nhiệt/hỏa (vd 'Can dương thượng
        # kháng' — Can hỏa thực nhiệt nhưng node tag Hư/Lý), tránh để Nhiệt phụ thuộc LLM (bất ổn).
        _strong_heat_kws = ["rêu vàng", "rêu lưỡi vàng", "lưỡi đỏ", "chất lưỡi đỏ", "đầu lưỡi đỏ",
                            "rìa lưỡi đỏ", "mắt đỏ", "mặt đỏ", "đỏ bừng", "khát nước", "họng đỏ",
                            "đờm vàng", "mũi vàng", "vàng đục", "tiểu vàng", "mụn đỏ", "nốt mụn đỏ", "sốt"]
        has_strong_heat = self._kw_hit_clean(symptoms_lower_all, _strong_heat_kws)
        if (not self._syndrome_is_exterior_wind(final_primary)
                and has_strong_heat and not has_cold_indicator):
            all_bat_cuong.add("Nhiệt")

        # [BỔ SUNG LÝ] Không có bất kỳ dấu BIỂU CHỨNG nào -> bệnh thuộc Lý theo phép loại trừ Bát
        # Cương (nội thương tạng phủ), bất kể node metadata có tag 'Lý' hay không — tránh nhãn cụt
        # (chỉ 'Hư'). Nhánh Biểu/đồng bệnh phía dưới không ảnh hưởng vì chỉ chạy khi has_bieu_indicator.
        if not has_bieu_indicator:
            all_bat_cuong.add("Lý")

        if has_bieu_indicator:
            if has_ly_indicator:
                all_bat_cuong.discard("Biểu")
                all_bat_cuong.discard("Lý")
                all_bat_cuong.add("Biểu - Lý đồng bệnh")
            else:
                all_bat_cuong.discard("Lý")
                all_bat_cuong.add("Biểu")

        # [GÁC THUẦN BIỂU] Chẩn đoán đã CHỐT thuần biểu (cốt lõi ngoại cảm, không hội chứng kèm
        # theo nội thương) và lời khai KHÔNG có dấu Lý thật (_ly_kws) -> trục Biểu-Lý phải là 'Biểu'.
        # Không có guard này, chữ 'Lý' rơi rớt — từ luật [BỔ SUNG LÝ] phía trên (bật vì 'sợ gió'
        # đơn độc không đủ chuẩn ố-hàn-phát-nhiệt nên has_bieu_indicator=False) hoặc từ metadata
        # node — đứng cạnh 'Biểu' của node ngoại cảm rồi bị khối gộp dưới dựng thành
        # 'Biểu - Lý đồng bệnh', mâu thuẫn trực tiếp Mục 3/4 'tà còn ở phần Biểu nông'.
        if self._syndrome_is_exterior_wind(final_primary) and not _concurrent_noi_thuong \
                and not any(kw in symptoms_lower_all for kw in _ly_kws):
            if "Lý" in all_bat_cuong:
                logger.info("[GÁC THUẦN BIỂU] Cốt lõi ngoại cảm thuần biểu, không dấu Lý thật -> gỡ 'Lý' rơi rớt.")
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
        # [CHỐNG NHIỄU METADATA] Chỉ hợp nhất khi có CĂN CỨ Thực thật: hoặc danh sách hội chứng có
        # hội chứng THỰC thuần theo tên (Phong hàn, Đàm thấp...), hoặc triệu chứng có dấu Thực
        # (has_thuc). Metadata đồ thị gộp theo ngữ cảnh bệnh nên node thuần Hư vẫn dính tag 'Thực'
        # (vd node 'Khí hư' mang ['Hư','Thực']) — thiếu căn cứ thì tag 'Thực' là nhiễu, bỏ đi và giữ
        # Hư (trước đây nhãn nhảy 'Lý - Hư' <-> 'Bản Hư Tiêu Thực' chỉ vì hội chứng kèm theo đổi).
        if "Hư" in all_bat_cuong and "Thực" in all_bat_cuong:
            _syn_names = [d.get("syndrome", "") for d in (detailed_kg_data or [])]
            all_bat_cuong.discard("Thực")
            if has_thuc or any(self._syndrome_is_thuc_pure(n) for n in _syn_names):
                all_bat_cuong.discard("Hư")
                all_bat_cuong.add("Bản Hư Tiêu Thực")

        # [ĐỐI CHIẾU CUỐI — BẤT BIẾN NHÃN] Sau mọi luật cộng/trừ ở trên, nhãn có thể tự mâu thuẫn
        # (đã xảy ra thật: '... - Hàn - Nhiệt - Thực - Bản Hư Tiêu Thực' — xung đột Hư/Thực mức
        # triệu chứng bật sớm rồi khối seed ngoại cảm thêm lại 'Thực'; 'Hàn' rò từ metadata hội
        # chứng kèm cạnh 'Nhiệt' của cốt lõi). Ép 2 bất biến trước khi hiển thị:
        # (1) Trục HƯ/THỰC theo HỘI CHỨNG ĐÃ CHỌN — toàn bộ biện chứng Mục 3-4 bám theo chúng:
        #     không hội chứng hư nào được chọn -> không được nói 'Bản Hư' (khớp Mục 3 'thuần Thực').
        _chosen_names = [final_primary] + (
            [final_concurrent] if final_concurrent and final_concurrent.strip().lower()
            not in ("không có", "", "chưa rõ") else [])
        _any_hu_chosen = any(self._syndrome_is_hu(n) for n in _chosen_names)
        _any_thuc_evidence = has_thuc or any(self._syndrome_is_thuc_pure(n) for n in _chosen_names)
        if not _any_hu_chosen:
            if "Bản Hư Tiêu Thực" in all_bat_cuong or "Hư" in all_bat_cuong:
                all_bat_cuong.discard("Bản Hư Tiêu Thực")
                all_bat_cuong.discard("Hư")
                if _any_thuc_evidence:
                    all_bat_cuong.add("Thực")
        elif "Bản Hư Tiêu Thực" in all_bat_cuong:
            all_bat_cuong.discard("Hư")
            all_bat_cuong.discard("Thực")
        elif "Hư" in all_bat_cuong and "Thực" in all_bat_cuong and _any_thuc_evidence:
            all_bat_cuong.discard("Hư")
            all_bat_cuong.discard("Thực")
            all_bat_cuong.add("Bản Hư Tiêu Thực")
        # (2) Trục HÀN/NHIỆT không được đứng cạnh nhau rời rạc: hai phía đều có căn cứ THẬT
        #     (triệu chứng chủ quan/tên hội chứng) -> 'Hàn Nhiệt Thác Tạp'; một phía -> giữ phía đó;
        #     tag chỉ đến từ metadata không căn cứ -> bỏ cả hai.
        if "Hàn Nhiệt Thác Tạp" in all_bat_cuong:
            all_bat_cuong.discard("Hàn")
            all_bat_cuong.discard("Nhiệt")
        elif "Hàn" in all_bat_cuong and "Nhiệt" in all_bat_cuong:
            _names_l = " | ".join(_chosen_names).lower()
            _han_corr = bool(re.search(r'\bhàn\b', _names_l)) or self._kw_hit_clean(
                symptoms_lower_all, ["sợ lạnh", "úy hàn", "rét run", "tay chân lạnh", "chân tay lạnh", "lưng lạnh"])
            _nhiet_corr = bool(re.search(r'\b(nhiệt|hỏa|hoả)\b', _names_l)) or has_heat_pulse_indicator
            if _han_corr and _nhiet_corr:
                all_bat_cuong.discard("Hàn")
                all_bat_cuong.discard("Nhiệt")
                all_bat_cuong.add("Hàn Nhiệt Thác Tạp")
            elif _nhiet_corr:
                all_bat_cuong.discard("Hàn")
            elif _han_corr:
                all_bat_cuong.discard("Nhiệt")
            else:
                all_bat_cuong.discard("Hàn")
                all_bat_cuong.discard("Nhiệt")

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

        # [TỔNG CƯƠNG ÂM-DƯƠNG] Bát Cương = 2 tổng cương (Âm/Dương) + 6 cương mục. Đồ thị chỉ tag
        # 6 cương mục (Biểu/Lý, Hàn/Nhiệt, Hư/Thực) — đúng thiết kế, vì Âm/Dương là KẾT LUẬN suy
        # từ 6 cương (Lý/Hàn/Hư nghiêng Âm; Biểu/Nhiệt/Thực nghiêng Dương), không phải tag dữ liệu.
        # Bổ sung ở tầng hiển thị; nhãn hợp nhất tính cho CẢ hai phía; CHỈ nêu khi nghiêng rõ
        # (cân bằng thì bỏ — không dùng chữ 'thác tạp' ở đây để không kích nhầm bộ đồng bộ Mục 4).
        _am_score = sum(1 for x in ("Lý", "Hàn", "Hư") if x in all_bat_cuong)
        _duong_score = sum(1 for x in ("Biểu", "Nhiệt", "Thực") if x in all_bat_cuong)
        for _mixed in ("Hàn Nhiệt Thác Tạp", "Bản Hư Tiêu Thực", "Biểu - Lý đồng bệnh"):
            if _mixed in all_bat_cuong:
                _am_score += 1
                _duong_score += 1
        if bat_cuong_order and _am_score != _duong_score:
            bat_cuong_hint += f" (tổng cương: thiên {'Âm' if _am_score > _duong_score else 'Dương'})"

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

        # Cờ mâu thuẫn HÀN-NHIỆT: dấu nhiệt (tiểu vàng/đờm vàng/rêu vàng...) trên nền hội chứng
        # hư/hàn — luật phủ-100% từng ép LLM "nhét" dấu nhiệt vào cơ chế dương hư ('thận dương hư
        # ... khiến nước tiểu vàng đậm' — SAI: dương hư cho tiểu trong dài, tiểu vàng là nhiệt).
        temp_conflict_hint = ""
        if has_yinyang_conflict:
            _heat_found = [k for k in heat_kws if self._kw_hit_clean(symptoms_lower_all, [k])]
            temp_conflict_hint = (
                f"\n        [CỜ MÂU THUẪN HÀN-NHIỆT - HINT QUAN TRỌNG]: Bệnh cảnh có dấu NHIỆT ({', '.join(_heat_found)}) "
                "song song với nền Hư/Hàn (Hàn Nhiệt Thác Tạp). TUYỆT ĐỐI CẤM giải thích các dấu nhiệt này bằng cơ chế "
                "dương hư/hư hàn (CẤM kiểu: 'dương hư khiến nước tiểu vàng' — dương hư cho tiểu TRONG DÀI). "
                "Phải giải thích chúng là NHIỆT/THẤP NHIỆT cục bộ (vd thấp nhiệt hạ tiêu, đàm uất hóa nhiệt) thuộc phần TIÊU, "
                "và Mục 4 PHẢI mô tả yếu tố nhiệt này, KHÔNG được ghi 'Không có Tiêu Thực'."
            )

        # BƯỚC 2, 3, 4: GỌI LLM BIỆN CHỨNG THEO CHAIN-OF-THOUGHT
        rag_prompt = f"LÝ GIẢI Y LÝ CHUẨN (BẮT BUỘC BÁM SÁT): {rag_context_str}" if rag_context_str else ""

        # [CHỐNG BỊA] Gợi ý biện luận luật 7 sinh ĐỘNG theo input: chỉ nhắc tên triệu chứng
        # thật sự có trong danh sách (LLM 7B không tuân điều kiện "nếu có" viết tĩnh trong prompt
        # — nó chép nguyên 'chóng mặt, hoa mắt' vào ca không có).
        _hint_parts = []
        if any(k in symptoms_lower for k in ("chóng mặt", "hoa mắt")):
            _hint_parts.append(
                "Dương khí hư suy dẫn đến Thanh dương bất thăng (khí trong trẻo không đủ sức thăng lên "
                "thượng tiêu), não bộ mất đi sự nuôi dưỡng phát sinh chóng mặt/hoa mắt.")
        if any(k in symptoms_lower for k in ("đau đầu", "nhức đầu")):
            _hint_parts.append(
                "Âm hàn ngưng trệ kinh mạch vùng đầu cổ, khí huyết kém lưu thông sinh ra đau đầu.")
        # Hằn răng KHÔNG kèm lưỡi bệu: LLM hay chép cơ chế mẫu 'thủy thấp làm lưỡi căng bệu và có
        # hằn răng' như một cặp — bịa thêm trạng thái bệu cho bệnh nhân chỉ có dấu răng nhẹ.
        # [NHẤT QUÁN LUẬT 15] KHÔNG nêu cứng 'Tỳ khí hư': hint cũ ép Mục 3 gọi tên một hội chứng
        # ngoài chẩn đoán đã chốt ('...cho thấy Tỳ khí hư' trong khi cốt lõi là Phế khí hư + kèm
        # Huyết hư) — mâu thuẫn chính luật 15 của prompt này. Theo mẫu luật 13 (lưỡi bệu): chỉ gọi
        # đích danh hội chứng Tỳ khi nó NẰM TRONG chẩn đoán; còn lại diễn đạt bằng CƠ CHẾ sinh lý
        # (Tỳ chủ vận hóa) không gọi tên hội chứng mới.
        if "hằn răng" in symptoms_lower and "lưỡi bệu" not in symptoms_lower:
            # Khớp theo TOKEN (âm tiết) trên bộ hội chứng ĐÃ CHỐT, giống concept_groups (dòng 1614):
            _chosen_hc = [s for s in (final_primary, final_concurrent or "") if s]
            _chosen_toks = set()
            for _s in _chosen_hc:
                _chosen_toks |= set(re.findall(r'[^\W\d_]+', _s.lower()))
            _ty_chosen = next((s for s in _chosen_hc if 'tỳ' in
                               set(re.findall(r'[^\W\d_]+', s.lower()))), None)
            _khihu_chosen = next((s for s in _chosen_hc if self._syndrome_is_hu(s) and 'khí' in
                                  set(re.findall(r'[^\W\d_]+', s.lower()))), None)
            if _ty_chosen:
                # (a) Chẩn đoán CÓ hội chứng Tỳ -> được gọi đích danh (nhất quán chẩn đoán).
                _han_rang_hint = (
                    f"Rìa lưỡi có hằn răng do {_ty_chosen} khiến vận hóa thủy thấp kém, thủy thấp "
                    "lưu giữ nhẹ khiến rìa lưỡi bị răng ép thành ngấn.")
            elif _khihu_chosen:
                # (b) Chẩn đoán là một hội chứng KHÍ HƯ (vd Phế khí hư) -> quy về khí hư đã chốt,
                # NHƯNG định vị đúng TẠNG: vận hóa thủy thấp là chức năng của TỲ (không phải Phế) —
                # trước đây ghi "kiện vận thủy thấp" trống tạng nên LLM gắn nhầm "vận hóa thủy thấp
                # của Phế" (sai y lý). Nêu như CƠ CHẾ đi kèm khí hư, KHÔNG dựng 'Tỳ khí hư' làm
                # hội chứng mới (Luật 15 đã cấm câu kết luận 'cho thấy Tỳ khí hư').
                _han_rang_hint = (
                    "Rìa lưỡi có hằn răng: vận hóa thủy thấp là chức năng của TẠNG TỲ (Tỳ chủ vận "
                    "hóa — KHÔNG phải Phế); trên nền khí hư đã chốt "
                    f"({_khihu_chosen}), Tỳ kiện vận thủy thấp kém nên thủy thấp lưu giữ nhẹ, rìa "
                    "lưỡi bị răng ép thành ngấn. Đây chỉ là CƠ CHẾ đi kèm khí hư, TUYỆT ĐỐI KHÔNG "
                    "kết luận thêm hội chứng 'Tỳ khí hư'/'Tỳ hư' ngoài chẩn đoán đã chốt.")
            else:
                # (c) Chẩn đoán không có khí hư/Tỳ (vd Âm hư, Huyết ứ) -> chỉ nêu CƠ CHẾ trung tính,
                # tuyệt đối không gán cơ chế khí hư lẫn tên hội chứng ngoài chẩn đoán.
                _han_rang_hint = (
                    "Rìa lưỡi có hằn răng do thủy thấp lưu giữ nhẹ khiến rìa lưỡi bị răng ép thành "
                    "ngấn — chỉ mô tả riêng dấu hằn răng, TUYỆT ĐỐI KHÔNG gọi tên hội chứng nào "
                    f"ngoài chẩn đoán đã chốt ({final_primary}, {final_concurrent}).")
            _hint_parts.append(
                _han_rang_hint +
                " LƯU Ý: bệnh nhân KHÔNG có lưỡi bệu — TUYỆT ĐỐI không mô tả thân lưỡi "
                "căng bệu/phù đại, chỉ giải thích riêng dấu hằn răng.")
        _hint_bien_luan = (
            "Gợi ý biện luận từ chuyên gia Đông y (CHỈ dùng cho triệu chứng CÓ trong danh sách): "
            + " ".join(_hint_parts)
        ) if _hint_parts else "(Không có gợi ý thêm — chỉ biện luận trên đúng danh sách triệu chứng.)"
        
        prompt_reason = f"""
        Vai trò: Chuyên gia Y học Cổ truyền chuyên nghiệp.
        Bệnh nhân có các triệu chứng: {symptoms_str}.
        Chẩn đoán cốt lõi: {final_primary}.
        Hội chứng kèm theo: {final_concurrent}.
        {rag_prompt}
        {face_conflict_hint}
        {temp_conflict_hint}

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
        5. CHỐT CHẶN TÂN DỊCH: Khi giải thích Đàm, Thấp, Ẩm, TUYỆT ĐỐI KHÔNG gọi mầm bệnh là "Tân dịch". Phải dùng từ "Thủy thấp" hoặc "Đàm trọc". Và NGƯỢC LẠI: Thủy thấp/Đàm trọc là TÀ KHÍ ứ đọng, KHÔNG PHẢI chất nuôi dưỡng — TUYỆT ĐỐI CẤM viết kiểu "thủy thấp không đủ để nuôi dưỡng da/cơ thể". Da khô, khô miệng phải giải thích bằng TÂN DỊCH hoặc ÂM HUYẾT bất túc không nuôi dưỡng được.
        6. LUẬT CHẶN HƯ THỰC (BẢN - TIÊU) (Cập nhật):
           - Nếu trạng thái Bát Cương hoặc Hội chứng thuộc Thực chứng thuần túy (ví dụ: Ngoại cảm phong nhiệt, Phong hàn phạm biểu, Thấp nhiệt... chỉ chứa chữ "Thực" and không có chữ "Hư"), ở phần "### 3. Phân tích Cơ chế Gốc (Bản Hư)", BẮT BUỘC phải ghi: "Không có Bản Hư, đây là bệnh lý Thực chứng thuần túy." TUYỆT ĐỐI KHÔNG được tự suy diễn ra các hội chứng tạng phủ suy nhược (như Thận hư, Tỳ hư, Tâm huyết hư).
           - Ngược lại, nếu bệnh thuộc Hư chứng thuần túy (chỉ chứa chữ "Hư" và không có chữ "Thực"), ở phần "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)", BẮT BUỘC phải ghi duy nhất một câu: "Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy." TUYỆT ĐỐI KHÔNG được viết thêm bất kỳ dòng diễn giải hay danh sách liệt kê triệu chứng nào khác ở mục này. Tất cả triệu chứng phải được giải thích gói gọn trong phần Bản Hư (Mục 3).
        7. LUẬT PHỦ LẤP TRIỆU CHỨNG 100% VÀ HÀNH VĂN MƯỢT MÀ (Cập nhật):
           - Bạn BẮT BUỘC phải đưa từng triệu chứng trong danh sách [{symptoms_str}] vào phần giải thích y lý và giải thích cơ chế vì sao có triệu chứng đó dưới góc độ sinh lý tạng phủ. 
           - Danh sách triệu chứng bắt buộc giải thích: {symptoms_str}.
           - Tuy nhiên, TUYỆT ĐỐI KHÔNG ĐƯỢC liệt kê gạch đầu dòng từng triệu chứng một cách rời rạc hay máy móc. Hãy hành văn thành một hoặc hai đoạn văn trôi chảy, xâu chuỗi các cơ chế lại với nhau một cách logic, mạch lạc như một danh y thực thụ. Nếu bạn bỏ sót bất kỳ triệu chứng nào (ví dụ không giải thích đau đầu, chóng mặt hay hoa mắt), chẩn đoán này sẽ bị coi là Thất bại hoàn toàn.
           - {_hint_bien_luan}
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
           - Ợ HƠI / Ợ CHUA tương tự: cơ chế BẮT BUỘC là VỊ mất hòa giáng → VỊ khí thượng nghịch (nếu có Can khí uất thì diễn đạt Can khí phạm Vị làm Vị càng nghịch). Nguyên tắc thăng-giáng: Tỳ chủ THĂNG (đưa thanh khí lên), Vị chủ GIÁNG (đưa trọc khí xuống) — TUYỆT ĐỐI CẤM (PROHIBITED) viết "khí Tỳ không thể hành xuống" hay quy ợ hơi/ợ chua cho khí Tỳ nghịch lên; khí bốc ngược trong ợ hơi luôn là khí của tạng VỊ.
        13. CHỐT CHẶN VỌNG CHẨN LƯỠI (TONGUE DIAGNOSIS CONSTRAINT) (MỚI):
           - Lưỡi bệu (hoặc lưỡi sưng, lưỡi to): Trong Đông y, lưỡi bệu/sưng BẮT BUỘC phải giải thích là do vận hóa bất thường khiến Thủy thấp ứ đọng/thủy dịch bất hóa làm thân lưỡi căng bệu sưng to. Tạng phủ gây bệnh phải CHỌN NHẤT QUÁN với chẩn đoán đã chốt ở Bước 1: chỉ quy về Thận dương hư khi hội chứng cốt lõi/kèm theo ({final_primary}, {final_concurrent}) có Thận dương hư; các trường hợp còn lại quy về Tỳ (Tỳ hư mất kiện vận / khí hư không vận hóa được thủy thấp). TUYỆT ĐỐI CẤM nêu "Thận dương hư" khi nó không nằm trong chẩn đoán đã chốt, và CẤM giải thích sai lệch lưỡi bệu là do "thiếu ẩm", "thiếu tân dịch" hay "không đủ dịch để bôi trơn".
           - Chất lưỡi đạm đỏ (lưỡi đỏ nhạt/nhạt đỏ): Phải giải thích do khí huyết suy kém không vinh nhuận đầy đủ lên thân lưỡi.
           - Lưỡi có vết nứt (nứt lưỡi): Phải giải thích do âm huyết hư tổn/tân dịch khuy tổn làm mất sự nuôi dưỡng trên bề mặt lưỡi.
        14. TÔN TRỌNG VÀ BIỆN GIẢI ĐÚNG VỀ RÊU TRẮNG MỎNG (Cập nhật):
           - Rêu trắng mỏng là ranh giới giữa Sinh lý và Bệnh lý.
           - CHỈ KHI bệnh nhân có dấu ngoại cảm thực sự (sợ gió, sợ lạnh, phát sốt, hắt hơi, sổ mũi, ngạt mũi) VÀ Bát Cương có chữ "Biểu": rêu trắng mỏng mới được giải thích là Ngoại cảm phong hàn (Tà khí mới xâm nhập đang ở phần Biểu nông).
           - Nếu KHÔNG có dấu ngoại cảm hoặc Bát Cương đã chốt là "Lý": TUYỆT ĐỐI CẤM (PROHIBITED) nhắc đến "tà khí xâm nhập", "Biểu nông" hay "ngoại cảm" khi bàn về rêu trắng mỏng — mâu thuẫn trực tiếp với định vị Lý chứng. Khi đó chỉ được nhận định: rêu trắng mỏng cho thấy vị khí còn tốt, chính khí chưa suy kiệt, bệnh còn ở mức độ nhẹ.
           - TUYỆT ĐỐI CẤM (PROHIBITED) giải thích rêu trắng mỏng là do Đàm ẩm, Thủy thấp ứ đọng hay Huyết ứ (vì khi đã có Thủy thấp/Đàm trọc tích tụ thì rêu lưỡi bắt buộc phải dày, nhớt hoặc trơn).
           - CẤM tự nâng cấp mô tả rêu: nếu danh sách triệu chứng chỉ có "rêu trắng mỏng" và/hoặc "rêu trắng nhớt" thì TUYỆT ĐỐI KHÔNG được viết thành "rêu dày" hay "rêu trắng dày" (nhớt ≠ dày; rêu mỏng hơi nhớt chỉ là thấp trệ NHẸ mới hình thành).
           - Với các từ khóa như 'lưỡi hồng', 'mạch hoãn', 'mạch bình thường', hãy nhận định đây là dấu hiệu sinh lý bình thường (vị khí còn tốt, chính khí chưa suy).
        15. CẤM TỰ BIÊN TỰ DIỄN HỘI CHỨNG MỚI (MỚI):
           - Bạn TUYỆT ĐỐI KHÔNG ĐƯỢC tự ý lôi kéo các hội chứng tạng phủ suy nhược khác không được chốt ở Bước 1 vào lập luận. Ví dụ: Nếu chẩn đoán cốt lõi và hội chứng kèm theo ở Bước 1 ({final_primary}, {final_concurrent}) KHÔNG có 'Thận âm hư', 'Tỳ dương hư' hay 'Can âm hư', bạn TUYỆT ĐỐI CẤM (PROHIBITED) sử dụng các thuật ngữ đó làm nguyên nhân gây hư hỏa hay bốc hỏa ở Mục 3 và Mục 4. Hãy giải thích cơ chế bốc hỏa/đỏ mặt dựa trên chính khí huyết hư (Ví dụ: huyết hư bất năng nhiếp dương, khiến hư hỏa/hư dương nổi lên trên) để đảm bảo tính nhất quán tuyệt đối giữa các bước chẩn đoán.
           - CỤ THỂ VỚI DẤU HẰN RĂNG / LƯỠI NHẠT: nếu chẩn đoán đã chốt ({final_primary}, {final_concurrent}) KHÔNG chứa 'Tỳ khí hư' hay 'Tỳ hư', TUYỆT ĐỐI CẤM (PROHIBITED) viết câu kết luận chẩn đoán kiểu "... cho thấy Tỳ khí hư" / "chứng tỏ Tỳ hư". Chỉ được diễn đạt hằn răng như CƠ CHẾ (Tỳ chủ vận hóa thủy thấp kém) bám theo đúng hội chứng cốt lõi, không nâng nó thành một hội chứng chẩn đoán riêng.
           - CỤ THỂ VỚI KHÍ HƯ TỔNG QUÁT: nếu hội chứng cốt lõi là 'Khí hư' / 'Khí huyết lưỡng hư' (tên KHÔNG mang tạng phủ cụ thể), cơ chế sinh khí phải quy về Tỳ (nguồn hóa sinh khí huyết, hậu thiên chi bản) và Phế (chủ khí). TUYỆT ĐỐI CẤM (PROHIBITED) viết "khí của tạng Can suy yếu" hay "Can khí hư" — bệnh lý điển hình của Can là KHÍ TRỆ/KHÍ UẤT (Thực chứng, thuộc Mục 4), Can KHÔNG phải nguồn Bản Hư của khí. Nếu hội chứng kèm theo là Can khí uất kết/Can khí uất trệ, nó chỉ được biện luận ở Mục 4 (Tiêu Thực), TUYỆT ĐỐI CẤM lấy nó làm nguyên nhân gốc của Bản Hư ở Mục 3.
        16. CHỐT CHẶN MỒ HÔI (ĐẠO HÃN vs TỰ HÃN) (MỚI):
           - ĐẠO HÃN (mồ hôi trộm — ra mồ hôi lúc ngủ, tỉnh dậy thì hết): BẮT BUỘC là biểu hiện của ÂM HƯ. Cơ chế: âm hư sinh nội nhiệt (hư hỏa), nhiệt bức tân dịch tiết ra ngoài về đêm. Phải giải thích ở "### 3. Phân tích Cơ chế Gốc (Bản Hư)" theo cơ chế âm hư → hư nhiệt → bức mồ hôi. TUYỆT ĐỐI CẤM (PROHIBITED) giải thích đạo hãn bằng cơ chế "khí hư/khí huyết hư/dương hư không giữ được mồ hôi" (đó là cơ chế của TỰ HÃN), và CẤM xếp đạo hãn vào "### 4. Phân tích Cơ chế Ngọn (Tiêu Thực)".
           - TỰ HÃN (mồ hôi tự ra ban ngày, vận động càng ra nhiều): mới là biểu hiện KHÍ HƯ/DƯƠNG HƯ (vệ khí bất cố, tấu lý không kín) → giải thích theo cơ chế khí hư bất cố nhiếp ở phần Bản Hư.
        17. CHỐT CHẶN QUẦNG THÂM MẮT / QUẦNG ĐEN DƯỚI MẮT (MỚI):
           - Quầng thâm (quầng đen) dưới mắt là dấu hiệu NỀN MẠN TÍNH — trong Đông y thường liên quan Thận hư, huyết ứ hoặc mất ngủ/mệt mỏi kéo dài. Nó KHÔNG phải triệu chứng của ngoại cảm cấp.
           - TUYỆT ĐỐI CẤM (PROHIBITED) bịa cơ chế gán quầng thâm mắt cho ngoại tà cấp tính (kiểu "ngoại tà ức chế làm khí huyết không lưu thông vùng dưới mắt") — tà khí mới phạm Biểu vài ngày không kịp tạo quầng thâm.
           - Nếu chẩn đoán đã chốt ({final_primary}, {final_concurrent}) là ngoại cảm biểu chứng và KHÔNG có hội chứng nội thương kèm theo: chỉ được nhận định trung tính rằng quầng thâm mắt là dấu hiệu nền có từ trước (có thể do thiếu ngủ, mệt mỏi kéo dài hoặc huyết ứ nhẹ), KHÔNG thuộc bệnh cảnh ngoại cảm cấp lần này và nên theo dõi thêm; CẤM dùng nó làm bằng chứng cho tà khí ở Biểu.
           - CHỈ KHI chẩn đoán đã chốt có hội chứng Thận hư/huyết ứ/mất ngủ lâu ngày: mới được giải thích quầng thâm mắt theo đúng cơ chế của hội chứng đó, và tuân thủ luật 15 (không tự biên hội chứng mới ngoài Bước 1).
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
            # [SUY GIẢM MỀM KHI LLM LỖI] Không bỏ trắng Mục 2-4 nữa. Bát Cương (Mục 2) đã được suy
            # ĐỊNH TÍNH từ graph (bat_cuong_hint) nên vẫn hiển thị đúng; Mục 3/4 dựng khung xác định
            # theo trạng thái Hư/Thực đã chốt + nêu rõ phần biện luận chi tiết tạm khuyết do lỗi kết
            # nối AI (tránh vừa 'Không xác định' vừa vẫn kê đơn — người dùng phải biết vì sao thiếu).
            _bc_l = (bat_cuong_hint or "").lower()
            _has_hu = "hư" in _bc_l
            _has_thuc = "thực" in _bc_l or "thác tạp" in _bc_l
            _note = ("_(Phần biện luận cơ chế chi tiết tạm thời chưa tạo được do lỗi kết nối máy chủ AI. "
                     "Định vị Bát Cương và hội chứng dưới đây được suy trực tiếp từ cơ sở tri thức Neo4j "
                     "nên vẫn tin cậy; vui lòng thử lại để có phần phân tích y lý đầy đủ.)_")
            if _has_hu and not _has_thuc:
                _m3 = (f"- Hội chứng cốt lõi **{final_primary}** thuộc Hư chứng: chính khí (khí/huyết/"
                       f"âm/dương) suy yếu, tạng phủ mất sự nuôi dưỡng và ôn hóa, sinh ra các triệu chứng "
                       f"nền: {symptoms_str}.\n{_note}")
                _m4 = "- Không có Tiêu Thực, đây là bệnh lý Hư chứng thuần túy."
            elif _has_thuc and not _has_hu:
                _m3 = "- Không có Bản Hư, đây là bệnh lý Thực chứng thuần túy."
                _m4 = (f"- Hội chứng cốt lõi **{final_primary}** thuộc Thực chứng: tà khí (phong/hàn/"
                       f"nhiệt/thấp/đàm/ứ) ứ trệ gây ra các biểu hiện: {symptoms_str}.\n{_note}")
            else:
                _m3 = (f"- Bệnh mang tính chất Bản Hư Tiêu Thực (vừa có gốc hư vừa có tà thực). Gốc bệnh "
                       f"là sự suy yếu của chính khí ở hội chứng cốt lõi **{final_primary}**.\n{_note}")
                _m4 = ("- Phần tà khí/triệu chứng cấp (Tiêu Thực) tạm thời chưa biện luận chi tiết được "
                       "do lỗi kết nối máy chủ AI; vui lòng thử lại.")
            llm_explanation = (
                f"### 2. Định vị Bát Cương\n- **Thuộc chứng:** {bat_cuong_hint}\n\n"
                f"### 3. Phân tích Cơ chế Gốc (Bản Hư)\n{_m3}\n\n"
                f"### 4. Phân tích Cơ chế Ngọn (Tiêu Thực / Triệu chứng cấp)\n{_m4}"
            )

        # [FIX] Bổ sung hậu xử lý: phát hiện và chèn triệu chứng bị LLM bỏ sót
        llm_explanation = self._patch_missing_symptoms(llm_explanation, symptoms_str)
        # [ĐỒNG BỘ] Ép Mục 4 nói cùng chiều với Bát Cương đã chốt ở Mục 2 (chạy CUỐI,
        # sau patch triệu chứng sót — patch có thể vừa chèn nội dung Thực vào Mục 4)
        llm_explanation = self._sync_tieu_thuc_with_bat_cuong(llm_explanation, bat_cuong_hint, symptoms_str)
        # [NHẤT QUÁN HÀN] Gỡ cơ chế 'hàn ngưng' bịa khi ca không có căn cứ Hàn (chạy sau cùng)
        llm_explanation = self._strip_unfounded_cold_mechanism(
            llm_explanation, bat_cuong_hint, final_primary, final_concurrent, symptoms_str)
        # [NHẤT QUÁN NHIỆT] MIRROR: gỡ cơ chế 'âm hư (sinh) nội nhiệt / nhiệt bức tân dịch' bịa cho
        # mồ hôi khi ca THUẦN HƯ-HÀN không có căn cứ Nhiệt (mồ hôi = tự hãn do vệ khí bất cố).
        llm_explanation = self._strip_unfounded_heat_mechanism(
            llm_explanation, bat_cuong_hint, final_primary, final_concurrent, symptoms_str)
        # [NHẤT QUÁN RÊU MỎNG] Gỡ câu quy 'rêu trắng mỏng' cho thấp/đàm hoặc quy nhân trực tiếp
        # (vi phạm luật 14 — Mục 3 từng mâu thuẫn thẳng với Mục 4 'vị khí còn tốt'); truyền
        # final_primary để CHỪA ca ngoại cảm biểu (được phép 'tà mới xâm nhập -> rêu trắng mỏng')
        llm_explanation = self._strip_thin_coating_damp_claims(llm_explanation, final_primary)
        # [NHẤT QUÁN LƯỠI-KHÔNG-RÊU] Chặn LLM quy 'lưỡi không/ít rêu' (= âm hư/tân dịch khuy) cho
        # huyết hư/khí hư/thấp/huyết ứ khi cốt lõi không liên quan âm/tân (over-fit theo core).
        llm_explanation = self._strip_no_coating_yin_claims(llm_explanation, final_primary, bat_cuong_hint)

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
        _printed_pairs = set()  # (benh_ly, bai_thuoc) đã in — khử trùng giữa nguồn graph và CSV
        if disease_names and len(disease_names) <= 3:
            for data in detailed_kg_data:
                syn_name = data["syndrome"].lower().strip()
                if syn_name not in active_syndromes:
                    continue
                is_core = (syn_name == primary_key)
                role = "Bản – gốc bệnh" if is_core else "Tiêu – nhánh/kèm theo"
                for tb in data.get("treatments_by_disease", []):
                    # Khớp chính xác bệnh lý VÀ có bài thuốc thật (OPTIONAL MATCH trên graph có thể
                    # trả dòng p=null khi node HoiChung biến thể hoa/thường không có bài -> chặn in
                    # "Dùng bài **None**").
                    if tb.get("disease") in disease_names and tb.get("bai_thuoc"):
                        # Chuẩn hoá vị thuốc về chuỗi sạch (KG trả list, CSV trả string) -> tránh in ['...','...']
                        vi_raw = tb.get("vi_thuoc") or []
                        if isinstance(vi_raw, (list, tuple, set)):
                            vi_str = ", ".join(str(v).strip() for v in vi_raw if str(v).strip())
                        else:
                            vi_str = str(vi_raw).strip()
                        vi_str = self._dedupe_herbs(vi_str) or "(chưa cập nhật vị thuốc)"
                        line = (
                            f"- Trị Bệnh **{tb['disease']}** — *{role}* (Hội chứng {data['syndrome']}) → Dùng bài **{tb['bai_thuoc']}**\n"
                            f"  - *Vị thuốc:* {vi_str}\n"
                        )
                        (core_lines if is_core else branch_lines).append(line)
                        _printed_pairs.add((str(tb.get("disease", "")).strip().lower(),
                                            str(tb.get("bai_thuoc", "")).strip().lower()))

            # [FIX CSV-FALLBACK TRUY HỒI] Đồ thị có thể CHƯA đồng bộ các dòng CSV mới (bệnh danh khớp
            # từ CSV nhưng cặp Hội chứng × Bệnh chưa tồn tại trên graph -> Mục 5 rỗng oan). Bổ sung
            # truy hồi trực tiếp từ CSV cho đúng cặp (bệnh danh đã chốt × hội chứng cốt lõi/kèm theo),
            # so khớp KHÔNG phân biệt hoa thường ('Tỳ Thận Dương hư' CSV == 'Tỳ thận dương hư' graph).
            _diseases_lower = {d.strip().lower() for d in disease_names}
            # Ưu tiên xử lý dòng thuộc hội chứng CỐT LÕI (Bản) TRƯỚC dòng kèm-theo (Tiêu): nếu cùng
            # một (bệnh, bài thuốc) tồn tại ở cả 2 vai trò, phải gán 'Bản' (không để thứ tự dòng CSV
            # quyết định nhầm bài gốc thành 'Tiêu' rồi in "chưa có bài đặc trị gốc" sai).
            _fallback_rows = sorted(
                (getattr(self, "csv_rows", None) or []),
                key=lambda r: r.get("hoi_chung", "").strip().lower() != primary_key
            )
            for _row in _fallback_rows:
                _b = _row.get("benh_ly", "").strip()
                _hc = _row.get("hoi_chung", "").strip().lower()
                _bt = _row.get("bai_thuoc", "").strip()
                if not _b or not _bt or _b.lower() not in _diseases_lower or _hc not in active_syndromes:
                    continue
                if (_b.lower(), _bt.lower()) in _printed_pairs:
                    continue
                _printed_pairs.add((_b.lower(), _bt.lower()))
                _is_core = (_hc == primary_key)
                _role = "Bản – gốc bệnh" if _is_core else "Tiêu – nhánh/kèm theo"
                _vi = self._dedupe_herbs(_row.get("vi_thuoc", "").strip()) or "(chưa cập nhật vị thuốc)"
                _line = (
                    f"- Trị Bệnh **{_b}** — *{_role}* (Hội chứng {_row.get('hoi_chung', '').strip()}) → Dùng bài **{_bt}**\n"
                    f"  - *Vị thuốc:* {_vi}\n"
                )
                (core_lines if _is_core else branch_lines).append(_line)

            # [FALLBACK THỂ TỔNG QUÁT] Cốt lõi đặc hiệu theo tạng (vd 'Phế khí hư') nhưng KB gán
            # bài theo thể TỔNG QUÁT của chính nó tại đúng bệnh danh đã chốt (vd 'Viêm yết hầu ×
            # Khí hư' — CSV:1015 'Phương bổ khí thanh hỏa') -> khớp tên chính xác trượt oan và
            # Mục 5 báo "chưa có bài thuốc" sai thực chất. Chỉ nhận thể KB mà MỌI âm tiết đều nằm
            # trong tên cốt lõi (tập-con THẬT SỰ: 'Khí hư' ⊂ 'Phế khí hư') — KHÔNG dùng
            # _are_syndromes_related (nhóm khái niệm rộng, kéo được cả bài trái cực 'Khí trệ huyết
            # ứ' cho ca hư). Hai lưới chắn bổ sung (đã rà toàn bộ CSV, loại 777 cặp lệch cực/nhãn
            # rác kiểu 'Thể nhẹ'): thể KB phải chứa token bệnh lý thật, và cùng cực Hư/Thực với
            # cốt lõi. In rõ vai trò 'thể tổng quát' để không bị hiểu nhầm là bài đặc trị đích danh.
            if not core_lines and primary_key not in ("chưa rõ", "", "không có"):
                _patho_toks = {
                    "hư", "suy", "nhược", "tổn", "nhiệt", "hàn", "thấp", "đàm", "đờm", "trọc",
                    "ẩm", "hỏa", "hoả", "ứ", "trệ", "uất", "kết", "tích", "nghịch", "độc",
                    "táo", "thử", "phong", "khí", "huyết", "âm", "dương",
                }
                _core_toks = set(re.findall(r'[^\W\d_]+', primary_key))
                _gen_rows = []
                for _row in (getattr(self, "csv_rows", None) or []):
                    _b = _row.get("benh_ly", "").strip()
                    _hc = _row.get("hoi_chung", "").strip()
                    _bt = _row.get("bai_thuoc", "").strip()
                    if not _b or not _hc or not _bt or _b.lower() not in _diseases_lower:
                        continue
                    if (_b.lower(), _bt.lower()) in _printed_pairs:
                        continue
                    _hc_toks = set(re.findall(r'[^\W\d_]+', _hc.lower()))
                    _subset_ok = bool(_hc_toks) and (_hc_toks < _core_toks)
                    # [NGOẠI CẢM CÙNG CHỮ KÝ] Cốt lõi ngoại cảm biểu ('Phong hàn phạm biểu') vs thể
                    # KB ngoại cảm CÙNG CHỮ KÝ BỆNH LÝ ('Phong hàn tập phế' — patho-token {phong,hàn}
                    # bằng nhau, chỉ khác từ định vị phạm biểu/tập phế) -> không chiều nào là tập con
                    # của chiều kia nên phép thử subset trượt oan dù đây chính là thể đúng cực của
                    # bệnh danh (Khái thấu × Phong hàn tập phế — Tam ảo thang). Chữ ký bằng nhau tự
                    # loại thể trái cực ('Phong táo/nhiệt...' có patho-token khác) và thể nội thương
                    # (đòi cả hai đều là ngoại cảm biểu).
                    _ext_sig_ok = (
                        not _subset_ok and bool(_hc_toks)
                        and self._syndrome_is_exterior_wind(final_primary)
                        and self._syndrome_is_exterior_wind(_hc)
                        and (_hc_toks & _patho_toks) == (_core_toks & _patho_toks)
                    )
                    # [BẮC CẦU DƯƠNG HƯ → KHÍ HƯ] 阳虚 = 气虚 + hàn: cốt lõi 'X dương hư' MƯỢN ĐƯỢC bài
                    # 'X khí hư' (bổ/nạp khí) CÙNG TẠNG của bệnh khi KB thiếu thể dương hư — kèm ghi chú
                    # gia ôn dương. MỘT CHIỀU (chỉ dương->khí; khí hư KHÔNG mượn ngược vì không cần ôn
                    # dương). Cùng tạng = token định vị (bỏ dương/khí/hư) TRÙNG (Thận dương hư↔Thận khí hư,
                    # KHÔNG Thận dương hư↔Phế khí hư). Ca thật: COPD × Thận dương hư -> Bổ thận nạp khí.
                    _duong_khi_ok = (
                        not (_subset_ok or _ext_sig_ok) and bool(_hc_toks)
                        and "dương" in _core_toks and "khí" in _hc_toks
                        and "dương" not in _hc_toks and "âm" not in _hc_toks
                        and (_core_toks - {"dương", "khí", "hư"}) == (_hc_toks - {"dương", "khí", "hư"})
                    )
                    if not (_subset_ok or _ext_sig_ok or _duong_khi_ok):
                        continue
                    if not (_hc_toks & _patho_toks):
                        continue
                    # Token dư của cốt lõi chỉ được là từ ĐỊNH VỊ (tạng phủ: phế/tỳ/thận...),
                    # KHÔNG được là token bệnh lý — nếu không, thể KB chỉ là MỘT THÀNH PHẦN của
                    # chứng hỗn hợp chứ không phải thể tổng quát (vd cốt lõi 'Khí hư huyết trệ'
                    # mà nhận thể 'Khí hư' thì bỏ rơi tà thực 'huyết trệ'). BỎ QUA cho cầu dương->khí
                    # (token dư 'dương' là phần được thay bằng 'khí', không phải thành phần bị rơi).
                    if not _duong_khi_ok and (_core_toks - _hc_toks) & _patho_toks:
                        continue
                    if self._syndrome_is_hu(_hc) != self._syndrome_is_hu(final_primary):
                        continue
                    _gen_rows.append((_hc, _b, _bt, _row.get("vi_thuoc", "").strip(), _subset_ok, _duong_khi_ok))
                # Thể sát cốt lõi nhất trước (nhiều âm tiết hơn = ít khái quát hơn); tối đa 2 bài
                _gen_rows.sort(key=lambda t: -len(re.findall(r'[^\W\d_]+', t[0])))
                for _hc, _b, _bt, _vi, _was_subset, _dk in _gen_rows[:2]:
                    _printed_pairs.add((_b.lower(), _bt.lower()))
                    _rel_word = "bao quát" if _was_subset else ("dạng khí hư của" if _dk else "tương ứng")
                    _dk_note = (" — *thể DƯƠNG hư: gia thêm vị ôn dương (Phụ tử/Nhục quế/Can khương)*"
                                if _dk else "")
                    core_lines.append(
                        f"- Trị Bệnh **{_b}** — *Bản – thể tổng quát của hội chứng cốt lõi* "
                        f"(Hội chứng {_hc} — {_rel_word} {final_primary}) → Dùng bài **{_bt}**{_dk_note}\n"
                        f"  - *Vị thuốc:* {self._dedupe_herbs(_vi) or '(chưa cập nhật vị thuốc)'}\n"
                    )

            # [FALLBACK THỂ KB CÙNG CỰC KHỚP BỆNH CẢNH] Chiều NGƯỢC của thể-tổng-quát: cốt lõi
            # TỔNG QUÁT thuần bệnh lý ('Khí hư') còn KB gán bài dưới thể ĐẶC HIỆU TẠNG của chính
            # bệnh danh đã chốt ('Bĩ mãn × Tỳ vị hư nhược' — Hương sa lục quân) -> token thể
            # không là tập con của cốt lõi nên tầng trên trượt, Mục 5 trắng oan dù KB có bài.
            # Tương tự cho Tiêu ('Can khí uất kết' vs thể 'Can vị bất hòa (can khí uất trệ)').
            # Ràng buộc y lý nằm trong _kb_polarity_fallback_rows (cùng cực Hư/Thực, không xung
            # trục tứ hư, liên quan y lý cho Tiêu, tả >=1 triệu chứng của ca).
            if not core_lines and primary_key not in ("chưa rõ", "", "không có"):
                _case_syms_m5 = list(dict.fromkeys((search_terms or []) + (detected_symptoms or [])))
                for _is_core2, _row2 in self._kb_polarity_fallback_rows(
                        _diseases_lower, final_primary, final_concurrent, _case_syms_m5, _printed_pairs):
                    _b2 = _row2.get("benh_ly", "").strip()
                    _hc2 = _row2.get("hoi_chung", "").strip()
                    _bt2 = _row2.get("bai_thuoc", "").strip()
                    _printed_pairs.add((_b2.lower(), _bt2.lower()))
                    _role2 = ("Bản – thể KB cùng cực khớp bệnh cảnh" if _is_core2
                              else "Tiêu – thể KB khớp hội chứng kèm theo")
                    _target2 = final_primary if _is_core2 else final_concurrent
                    _line2 = (
                        f"- Trị Bệnh **{_b2}** — *{_role2}* (Hội chứng {_hc2} — tương ứng {_target2}) → Dùng bài **{_bt2}**\n"
                        f"  - *Vị thuốc:* {self._dedupe_herbs(_row2.get('vi_thuoc', '').strip()) or '(chưa cập nhật vị thuốc)'}\n"
                    )
                    (core_lines if _is_core2 else branch_lines).append(_line2)
                    logger.info(f"[FALLBACK THỂ KB CÙNG CỰC] {'Bản' if _is_core2 else 'Tiêu'}: "
                                f"{_b2} × {_hc2} -> {_bt2}")

            # [FALLBACK THỂ ĐỒNG NGHĨA] Core lệch TÊN với hội chứng của bệnh đã chốt nhưng ĐỒNG
            # NGHĨA cùng pháp trị theo bản đồ curated (data/syndrome_synonyms.json) — vd core 'Can
            # vị bất hòa' (grounded ở bệnh khác) vs 'Can khí phạm vị' của Nôn mửa/Ẩu thổ. Anh-em
            # đồng nghĩa KHÔNG tập-con nên mọi tầng token phía trên trượt -> Mục 5 trắng OAN dù KB
            # có bài. CHỈ bắc cầu cặp ĐÃ THẨM ĐỊNH (không dùng _are_syndromes_related lỏng — 46%
            # cặp related lệch trục Âm/Dương, sẽ kê trái cực). Belt-and-suspenders: vẫn gác cùng
            # cực Hư/Thực + không xung nhiệt (dù cụm đã curated cùng cực).
            if not core_lines and primary_key not in ("chưa rõ", "", "không có"):
                _syn_rows = []
                for _row in (getattr(self, "csv_rows", None) or []):
                    _b = _row.get("benh_ly", "").strip()
                    _hc = _row.get("hoi_chung", "").strip()
                    _bt = _row.get("bai_thuoc", "").strip()
                    if not _b or not _hc or not _bt or _b.lower() not in _diseases_lower:
                        continue
                    if (_b.lower(), _bt.lower()) in _printed_pairs:
                        continue
                    if not self._syndromes_are_synonyms(final_primary, _hc):
                        continue
                    if self._syndrome_is_hu(_hc) != self._syndrome_is_hu(final_primary):
                        continue
                    if self._syndromes_thermal_conflict(final_primary, _hc):
                        continue
                    _syn_rows.append((_hc, _b, _bt, _row.get("vi_thuoc", "").strip()))
                for _hc, _b, _bt, _vi in _syn_rows[:2]:
                    _printed_pairs.add((_b.lower(), _bt.lower()))
                    core_lines.append(
                        f"- Trị Bệnh **{_b}** — *Bản – thể tương đương của hội chứng cốt lõi* "
                        f"(Hội chứng {_hc} — đồng nghĩa {final_primary}) → Dùng bài **{_bt}**\n"
                        f"  - *Vị thuốc:* {self._dedupe_herbs(_vi) or '(chưa cập nhật vị thuốc)'}\n"
                    )
                    logger.info(f"[FALLBACK THỂ ĐỒNG NGHĨA] {_b} × {_hc} (đồng nghĩa {final_primary}) -> {_bt}")

        # [NHẤT QUÁN TEXT ↔ ĐỒ THỊ] Cốt lõi/kèm theo chưa có bài grounded nhưng một hội chứng
        # LIÊN QUAN trong danh sách ứng viên (vd Huyết hư khi cốt lõi là Khí huyết lưỡng hư) CÓ bài
        # gán đúng bệnh danh — đồ thị lập luận vẫn vẽ bài đó nên Mục 5 nói "chưa có bài thuốc" là
        # tự mâu thuẫn với hình. Bổ sung tầng THAM KHẢO: chỉ lấy hội chứng liên quan y lý với cốt
        # lõi, ghi rõ vai trò để không bị hiểu nhầm là bài trị gốc.
        related_lines = []
        if disease_names and len(disease_names) <= 3 and not core_lines and not branch_lines:
            for data in detailed_kg_data:
                syn_name = data["syndrome"].lower().strip()
                if syn_name in active_syndromes:
                    continue
                if not self._are_syndromes_related(final_primary, data["syndrome"]):
                    continue
                # [CHỐNG BÀI TRÁI CỰC] Cốt lõi Phong hàn (tân ôn) không được gợi ý bài của ứng viên
                # Phong táo/Phong nhiệt (tân lương nhuận táo/thanh nhiệt) — 'liên quan' qua chữ
                # 'phong' nhưng pháp trị NGƯỢC nhau. Thà để Mục 5 'chưa có bài, tham khảo thầy
                # thuốc' còn hơn kê bài trái cực (vd Tang hạnh thang cho ca cảm phong hàn).
                if self._syndromes_thermal_conflict(final_primary, data["syndrome"]):
                    continue
                for tb in data.get("treatments_by_disease", []):
                    if tb.get("disease") in disease_names and tb.get("bai_thuoc"):
                        _key = (str(tb.get("disease", "")).strip().lower(),
                                str(tb.get("bai_thuoc", "")).strip().lower())
                        if _key in _printed_pairs:
                            continue
                        _printed_pairs.add(_key)
                        vi_raw = tb.get("vi_thuoc") or []
                        vi_str = (", ".join(str(v).strip() for v in vi_raw if str(v).strip())
                                  if isinstance(vi_raw, (list, tuple, set)) else str(vi_raw).strip())
                        vi_str = self._dedupe_herbs(vi_str)
                        related_lines.append(
                            f"- Trị Bệnh **{tb['disease']}** — *Tham khảo – hội chứng liên quan* "
                            f"(Hội chứng {data['syndrome']}) → Dùng bài **{tb['bai_thuoc']}**\n"
                            f"  - *Vị thuốc:* {vi_str or '(chưa cập nhật vị thuốc)'}\n"
                        )

        has_treatment = bool(core_lines or branch_lines or related_lines)

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

        # In bài THAM KHẢO từ hội chứng liên quan (chỉ khi không có bài Bản/Tiêu nào)
        if related_lines:
            final_markdown += (
                f"- **Pháp trị gốc (Bản) — {final_primary}:** cơ sở tri thức hiện **chưa có bài thuốc "
                f"đặc trị** gán trực tiếp cho hội chứng cốt lõi ở bệnh danh đã chốt — cần thầy thuốc "
                f"kê bài tư bổ theo gốc.\n"
                f"- Bài thuốc dưới đây thuộc **hội chứng liên quan** trong danh sách ứng viên "
                f"(tham khảo thêm, KHÔNG phải bài trị hội chứng cốt lõi):\n"
            )
            final_markdown += "".join(related_lines)

        if not has_treatment:
            if not disease_names or len(disease_names) > 3:
                # Dùng lại symptoms_lower_all ỔN ĐỊNH đã dựng ở đầu khối Bát Cương — bản gán đè cũ
                # theo symptoms_str khiến cả check mạch/lưỡi lẫn cổng lời khuyên Mục 5 phía dưới
                # ăn theo từ đồng nghĩa LLM bung (vd 'tiết tả' bịa -> kê nhầm phương tễ tả lỵ).
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

        # Cập nhật lời khuyên đặc trị nếu có (Thêm trực tiếp vào Mục 5).
        # [AN TOÀN] Lời khuyên KHÔNG được cài cứng theo TÊN hội chứng bất chấp triệu chứng: bản cũ
        # cứ thấy 'Tỳ thận dương hư' là kê Tứ thần hoàn/Phụ tử lý trung thang "trị TIÊU CHẢY" cho cả
        # bệnh nhân không hề tiêu chảy (Phụ tử có độc). Chỉ nêu phương tễ tả lỵ khi THẬT SỰ có tiêu chảy.
        if "Tỳ thận dương hư" in final_primary:
            _has_tieu_chay = any(k in symptoms_lower_all for k in [
                "tiêu chảy", "đại tiện lỏng", "phân lỏng", "phân nát", "phân sống", "đi lỏng",
                "tiết tả", "ngũ canh tả", "ỉa chảy", "đi ngoài lỏng"])
            if _has_tieu_chay:
                final_markdown += "\n- **Lời khuyên bổ sung:** Ôn bổ Tỳ Thận, Sáp trường chỉ tả. Phương tễ kinh điển nhất để điều trị Tỳ thận dương hư tiêu chảy là Tứ thần hoàn (hoặc Phụ tử lý trung thang gia giảm).\n"
            else:
                final_markdown += "\n- **Lời khuyên bổ sung:** Ôn bổ Tỳ Thận, phù trợ dương khí.\n"
        elif "Khí huyết đều hư" in final_primary:
            final_markdown += "\n- **Lời khuyên bổ sung:** Ích khí kiện Tỳ, bổ huyết dưỡng Tâm để phục hồi từ gốc.\n"
            
        return final_markdown


    # Triệu chứng "cờ đỏ" cần đi cấp cứu ngay, không tự dùng thuốc Đông y
    # Triệu chứng "cờ đỏ" cần đi cấp cứu ngay, không tự dùng thuốc Đông y.
    # Khớp bằng REGEX (re.search trên text đã .lower()), KHÔNG khớp substring trần, để tránh false positive:
    #  - "khó thở" trần = đoản khí mạn tính của khí hư -> chỉ bật khi có bối cảnh CẤP;
    #  - "liệt" trần dính "liệt dương" (Thận dương hư);
    #  - "xuất huyết" trần dính "ban/nốt/chấm/điểm xuất huyết" (dấu hiệu da từ VLM, face_symptoms_list).
    # Gap [^,.;\n]{0,N} giới hạn trong MỘT mệnh đề, không cho ghép nhầm qua dấu phẩy của combined_query.
    _RED_FLAG_PATTERNS = [re.compile(p) for p in [
        # --- Tim mạch / hô hấp cấp ---
        r"đau (thắt )?ngực",
        r"khó thở[^,.;\n]{0,12}(dữ dội|đột ngột|dồn dập)",
        r"khó thở cấp",
        r"(đột ngột|đột nhiên|tự nhiên)[^,.;\n]{0,8}khó thở",
        r"không thở (được|nổi)",
        r"thở không (được|nổi)",
        r"nghẹt thở",
        r"ngạt thở",
        # --- Thần kinh / đột quỵ ---
        r"(?<!ngây )ngất",                                    # "ngất", "ngất xỉu"; chặn "ngây ngất"
        r"co giật",                                           # bao luôn "sốt cao co giật"
        r"liệt (nửa )?(người|mặt|mồm|miệng|tay|chân|tứ chi|toàn thân)",
        r"(tê|yếu) liệt",
        r"(đột ngột|đột nhiên)[^,.;\n]{0,10}liệt(?!\s*dương)",
        r"méo miệng",
        r"yếu nửa người",
        r"bán thân bất toại",
        r"nói khó",
        r"(?<!chuyện )khó nói",                               # chặn uyển ngữ "chuyện khó nói"
        r"lơ mơ",
        r"hôn mê",
        r"mất ý thức",
        r"mất ngôn ngữ",
        r"cứng gáy",
        r"đau đầu dữ dội",
        # --- Xuất huyết / chảy máu ---
        r"ho ra máu",
        r"khạc[^,.;\n]{0,8}máu",                              # khạc máu / khạc ra máu
        r"đờm[^,.;\n]{0,14}máu",                              # đờm có lẫn máu tươi / đờm dính máu
        r"máu[^,.;\n]{0,14}đờm",                              # máu trong đờm
        r"ộc máu",
        r"nôn ra máu",
        r"đi ngoài ra máu",
        r"(?<!ban )(?<!nốt )(?<!chấm )(?<!điểm )xuất huyết",  # chặn dấu hiệu da liễu từ VLM
        r"chảy máu không cầm",
        # --- Chấn thương ---
        r"chấn thương sọ não",
        r"tai nạn",
    ]]

    def _append_medical_disclaimer(self, markdown: str, raw_text: str = "") -> str:
        """Chèn cảnh báo cấp cứu (nếu phát hiện cờ đỏ) và disclaimer y tế vào cuối MỌI câu trả lời."""
        markdown = markdown or ""
        text_lower = (raw_text or "").lower()

        red_flag_block = ""
        if any(rx.search(text_lower) for rx in self._RED_FLAG_PATTERNS):
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

    def _detect_makeup(self, face_desc: str) -> bool:
        """Phát hiện khuôn mặt CÓ trang điểm từ mô tả của VLM, có xử lý phủ định + PHỦ ĐỊNH LIỆT KÊ.
        VLM hay chốt kiểu 'Không có trang điểm rõ (không thấy son môi, phấn má, kẻ mắt)' — nếu chỉ
        tách mệnh đề theo dấu phẩy thì 'phấn má'/'kẻ mắt' rơi vào mệnh đề không còn chữ 'không' và
        bật cờ sai. Quy tắc: trong MỘT CÂU, sau khi đã gặp từ phủ định, các mệnh đề sau kế thừa
        phủ định — trừ khi mệnh đề đó tự khẳng định lại ('..., có son môi đỏ')."""
        makeup_keywords = ["makeup", "lipstick", "eyeliner", "blush", "mascara", "foundation", "eyeshadow", "trang điểm", "son môi", "má hồng", "kẻ mắt", "phấn má", "phấn nền"]
        negation_markers = ["không", "chưa", "no ", "not ", "without", "n't"]
        # dấu hiệu ĐÃ trang điểm: khi mệnh đề sau phủ định vẫn nêu keyword kèm mô tả có/màu/độ đậm
        # ('son môi đỏ nhạt') -> là khẳng định thật, không phải mục liệt kê bị phủ định ('phấn má','kẻ mắt').
        descriptor_re = r"(đỏ|hồng|nâu|cam|tím|nhạt|đậm|dày|tô|thoa|đánh|nhẹ|rõ|\bcó\b|visible|wearing|has)"
        import re as _re
        for sentence in _re.split(r"[.;]", (face_desc or "").lower()):
            # Từ tương phản ('nhưng', 'tuy nhiên', 'but') RESET kế thừa phủ định: 'không son môi
            # NHƯNG phấn nền dày' -> vế sau là khẳng định, không kế thừa phủ định của vế trước.
            for segment in _re.split(r"\bnhưng\b|\btuy nhiên\b|\bbut\b|\bhowever\b", sentence):
                neg_seen = False
                for clause in _re.split(r"[,:()]", segment):
                    has_neg = any(neg in clause for neg in negation_markers)
                    has_kw = any(kw in clause for kw in makeup_keywords)
                    if has_kw:
                        if has_neg:
                            # mệnh đề phủ định VỀ trang điểm -> kế thừa cho các mục liệt kê phía sau
                            # ('không thấy son môi, phấn má, kẻ mắt')
                            neg_seen = True
                            continue
                        if neg_seen:
                            # đã ở trong ngữ cảnh phủ định trang điểm: chỉ tính là CÓ trang điểm khi
                            # mệnh đề tự mô tả sự hiện diện (màu/độ đậm) NGOÀI phần tên keyword
                            residual = clause
                            for kw in makeup_keywords:
                                residual = residual.replace(kw, " ")
                            if not _re.search(descriptor_re, residual):
                                continue
                        return True
                    # Mệnh đề phủ định về đặc điểm KHÁC ('không có mụn') KHÔNG kế thừa sang mệnh đề
                    # trang điểm dương tính sau nó -> chỉ makeup-negation mới set neg_seen (ở nhánh trên).
        return False

    def run_diagnosis(self, user_symptoms: str = "", face_img_path: str = None, tongue_img_path: str = None,
                      sex: str = None) -> dict:
        vision_analysis_text = ""
        raw_vision_data = None
        self._has_makeup = False
        # [CỔNG GIỚI TÍNH] Giới khai báo (cấu trúc) từ form -> _find_matching_diseases loại bệnh khác
        # giới. Đặt lại mỗi lần chạy (tránh dính giới ca trước); rỗng -> suy từ text làm dự phòng.
        self._patient_sex = self._norm_sex(sex)

        if face_img_path or tongue_img_path:
            logger.info("Bắt đầu phân tích hình ảnh qua mô hình vision...")
            try:
                raw_vision_data = self.vision_pipeline.run(tongue_image_path=tongue_img_path, face_image_path=face_img_path)
                if raw_vision_data and isinstance(raw_vision_data, dict):
                    tongue_desc = raw_vision_data.get("tongue_description", "")
                    face_desc = raw_vision_data.get("face_description", "")
                    
                    if face_desc:
                        self._has_makeup = self._detect_makeup(face_desc)

                    raw_vision_data["has_makeup"] = self._has_makeup
                    
                    detected_symptoms = []

                    # [VỌNG CHẨN CÓ CẤU TRÚC] Nếu VLM đã trả JSON (chế độ structured), dùng triệu chứng
                    # map DETERMINISTIC — bỏ bước LLM đọc prose. Chỉ modality nào KHÔNG có structured
                    # mới cần gọi _map_desc_to_symptoms (fallback prose). Tiết kiệm 1-2 lượt LLM/ca và
                    # loại nguồn ảo giác do LLM đọc mô tả tự do.
                    _struct = raw_vision_data.get("structured", {}) or {}
                    _struct_tongue = _struct.get("tongue", {}).get("symptoms") if _struct.get("tongue") else None
                    _struct_face = _struct.get("face", {}).get("symptoms") if _struct.get("face") else None
                    # Trang điểm: lấy trực tiếp từ trường JSON (chắc chắn hơn dò prose)
                    if _struct.get("face", {}).get("data"):
                        from src.vision_schema import is_makeup
                        if is_makeup(_struct["face"]["data"]):
                            self._has_makeup = True
                            raw_vision_data["has_makeup"] = True

                    # Khớp mô tả lưỡi + mặt SONG SONG (2 lượt gọi LLM độc lập nhau) — chỉ cho phần prose
                    from concurrent.futures import ThreadPoolExecutor
                    _match_futures = {}
                    with ThreadPoolExecutor(max_workers=2) as _ex:
                        if tongue_desc and _struct_tongue is None:
                            logger.info("Khớp mô tả lưỡi với danh sách triệu chứng chuẩn...")
                            _match_futures["tongue"] = _ex.submit(
                                self._map_desc_to_symptoms, tongue_desc, self.tongue_symptoms_list)
                        if face_desc and _struct_face is None:
                            logger.info("Khớp mô tả sắc mặt với danh sách triệu chứng chuẩn...")
                            _match_futures["face"] = _ex.submit(
                                self._map_desc_to_symptoms, face_desc, self.face_symptoms_list)

                    if _struct_tongue is not None:
                        logger.info(f"Triệu chứng lưỡi (structured/deterministic): {_struct_tongue}")
                        detected_symptoms.extend(_struct_tongue)
                    elif "tongue" in _match_futures:
                        mapped_tongue = _match_futures["tongue"].result()
                        logger.info(f"Triệu chứng lưỡi đã khớp: {mapped_tongue}")
                        detected_symptoms.extend(mapped_tongue)
                    if _struct_face is not None:
                        logger.info(f"Triệu chứng mặt (structured/deterministic): {_struct_face}")
                        detected_symptoms.extend(_struct_face)
                    elif "face" in _match_futures:
                        mapped_face = _match_futures["face"].result()
                        logger.info(f"Triệu chứng mặt đã khớp: {mapped_face}")
                        detected_symptoms.extend(mapped_face)
                        
                    detected_symptoms = list(set(detected_symptoms))
                    detected_symptoms = self._resolve_symptom_conflicts(detected_symptoms)
                            
                    raw_vision_data["detected_symptoms"] = detected_symptoms
                    vision_analysis_text = ", ".join(detected_symptoms)
                    raw_vision_data["analysis"] = vision_analysis_text
                    
                    # Structured -> prose đã TIẾNG VIỆT sẵn, gán thẳng (bỏ 1 lượt LLM dịch/modality).
                    if tongue_desc:
                        raw_vision_data["tongue_description_vi"] = (
                            tongue_desc if _struct_tongue is not None
                            else self._translate_english_description(tongue_desc))
                    if face_desc:
                        raw_vision_data["face_description_vi"] = (
                            face_desc if _struct_face is not None
                            else self._translate_english_description(face_desc))

                    # [GÁC CỔNG ẢNH] Nếu ảnh tải nhầm ô -> hiển thị CẢNH BÁO thay cho mô tả (đã bỏ qua,
                    # không map thành triệu chứng nên không làm sai chẩn đoán).
                    _vw = raw_vision_data.get("vision_warnings") or {}
                    if _vw.get("tongue"):
                        raw_vision_data["tongue_description_vi"] = _vw["tongue"]
                    if _vw.get("face"):
                        raw_vision_data["face_description_vi"] = _vw["face"]
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

        # [CỔNG ÂM-HƯ + THERMAL — CHẠY SỚM] Hạ core âm-hư/nhiệt SAI cực NGAY trên _grounded, TRƯỚC các
        # gác biểu/chủ-chứng (chúng loại ứng viên thay thế non-nhiệt khi core là ngoại cảm thuần ->
        # cổng ở _generate_explainable_answer không còn ứng viên mà đưa lên). Discriminator '0 dấu
        # nhiệt' (xem _demote_*). Giữ điểm gốc để ranking tiếp theo không lệch.
        if _grounded:
            _g_names = [s for s, _ in _grounded]
            _case_txt = (user_symptoms + " " + combined_query).lower()
            _g_names, _amhu_r = self._demote_amhu_without_heat(_g_names, _case_txt)
            if _amhu_r:
                logger.info(f"[CỔNG ÂM-HƯ KHÔNG NHIỆT] {_amhu_r}")
            _g_names, _nhiet_r = self._demote_nhiet_without_heat(_g_names, _case_txt)
            if _nhiet_r:
                logger.info(f"[CỔNG THERMAL-POLARITY] {_nhiet_r}")
            _score_map = dict(_grounded)
            _grounded = [(s, _score_map.get(s, 0.0)) for s in _g_names]

        # [FIX TRUY HỒI NHẤT QUÁN — CỔNG CHỦ CHỨNG] Hội chứng KÈM THEO không được chốt chỉ bằng vài
        # dấu lưỡi/mặt chung chung: node hội chứng hẹp/ít triệu chứng (vd 'Đờm Trọc Ngưng Kết' — bệnh
        # nam khoa, chỉ cần 'lưỡi nhạt + rìa lưỡi có vết răng' là khớp 2 hits) sẽ leo hạng nhì và lật
        # chẩn đoán giữa các lần chạy. Khi bệnh nhân CÓ lời khai chủ chứng, mọi hội chứng từ hạng 2
        # trở xuống phải khớp >= 1 triệu chứng chủ chứng; hạng 1 (cốt lõi) giữ nguyên theo điểm tổng.
        if _grounded and len(_grounded) > 1 and _clean_terms:
            _chief_syns = self._syndromes_matching_terms(_clean_terms)
            if _chief_syns:
                _dropped = [s for s, _sc in _grounded[1:] if s not in _chief_syns]
                if _dropped:
                    logger.info(f"[CỔNG CHỦ CHỨNG] Loại hội chứng kèm theo không khớp lời khai: {_dropped}")
                    _grounded = [_grounded[0]] + [(s, sc) for s, sc in _grounded[1:] if s in _chief_syns]

        # [CỔNG HÀN-NHIỆT KÈM THEO] Bệnh cảnh CHỈ có dấu hàn (sợ lạnh/rét run, không một dấu nhiệt
        # nào) -> loại hội chứng kèm theo NGOẠI TÀ NHIỆT thuần ('Phong nhiệt'... leo hạng nhờ triệu
        # chứng dùng chung sổ mũi/hắt hơi rồi rò tag 'Nhiệt' vào Bát Cương); đối xứng cho bệnh cảnh
        # chỉ-nhiệt với hội chứng hàn-thực. Chỉ đụng hạng 2+ và chỉ đụng hội chứng THỰC THUẦN —
        # không đụng hư nhiệt/hư hàn nội thương, không đụng ca hàn nhiệt lẫn lộn (đã có luật riêng).
        if _grounded and len(_grounded) > 1:
            _txt_hc = (user_symptoms + " " + combined_query).lower()
            _cold_only = self._kw_hit_clean(_txt_hc, ["sợ lạnh", "úy hàn", "rét run", "lạnh run"])
            # Sắc mặt TRẮNG NHỢT là dấu Hư/Hàn, nghịch hẳn với Nhiệt. Trên bệnh cảnh KHÔNG một dấu
            # nhiệt nào, nó đủ để loại companion NHIỆT THỰC THUẦN (Phong nhiệt...) leo hạng nhờ triệu
            # chứng dùng chung (chảy mũi/hắt hơi) rồi rò 'Nhiệt' vào Bát Cương cho ca mặt trắng nhợt.
            # CHỈ dùng cho nhánh loại-Nhiệt, KHÔNG đưa vào nhánh đối xứng loại-Hàn (mặt nhợt không
            # phải bằng chứng nhiệt).
            _pale_cold_face = self._kw_hit_clean(_txt_hc, ["mặt nhợt", "mặt nhợt nhạt", "sắc mặt trắng",
                                                           "mặt trắng", "trắng nhợt", "trắng bệch", "sắc mặt nhợt"])
            _heat_only = self._kw_hit_clean(_txt_hc, ["sốt", "phát nhiệt", "khát nước", "đỏ bừng",
                                                      "mạch sác", "tế sác", "rêu vàng", "tiểu vàng",
                                                      "họng đỏ", "đờm vàng", "mũi vàng", "vàng đục"])
            _drop_kw = None
            if (_cold_only or _pale_cold_face) and not _heat_only:
                _drop_kw = ("nhiệt", "hỏa", "hoả")
            elif _heat_only and not _cold_only:
                _drop_kw = ("hàn",)
            # (Nhánh theo-tên-cốt-lõi được chuyển XUỐNG SAU luật Bản Hư Tiêu Thực — xem
            # [CỔNG HÀN-NHIỆT THEO CỐT LÕI CUỐI]: ở vị trí này ranking chưa chốt, cốt lõi tạm thời
            # có thể là ứng viên THỰC sắp bị hạ, lấy cực của nó làm chuẩn sẽ loại nhầm/giữ nhầm.)
            if _drop_kw:
                _dropped_hn = [s for s, _sc in _grounded[1:]
                               if self._syndrome_is_thuc_pure(s) and any(k in s.lower() for k in _drop_kw)]
                if _dropped_hn:
                    logger.info(f"[CỔNG HÀN-NHIỆT] Loại hội chứng kèm theo trái cực hàn-nhiệt: {_dropped_hn}")
                    _grounded = [_grounded[0]] + [(s, sc) for s, sc in _grounded[1:] if s not in _dropped_hn]

        # [CỔNG DƯƠNG HƯ] Hội chứng DƯƠNG HƯ làm CỐT LÕI bắt buộc có >=1 dấu hư hàn thật (sợ lạnh,
        # tay chân lạnh, tiểu đêm, ngũ canh tả, phân sống, lưng lạnh, liệt dương...) — dương hư định
        # nghĩa bằng hư hàn. Sau khi gộp node trùng tên (dedupe-casefold), bằng chứng hợp nhất có thể
        # đẩy 'Tỳ thận dương hư' vượt 'Tỳ khí hư' trên ca tiêu hóa thuần không dấu hàn nào — khi đó
        # hoán vị xuống dưới ứng viên HƯ không-dương-hư đầu tiên (vẫn giữ trong danh sách kèm theo).
        if _grounded:
            _txt_dh = (user_symptoms + " " + combined_query).lower()
            _has_cold_sign = self._kw_hit_clean(_txt_dh, [
                "sợ lạnh", "úy hàn", "rét run", "lạnh run", "tay chân lạnh", "chân tay lạnh",
                "chi lạnh", "lưng lạnh", "lạnh bụng", "bụng lạnh", "tiểu đêm", "ngũ canh",
                "phân sống", "liệt dương", "lưng gối lạnh", "sợ gió"])
            # Nhận diện họ DƯƠNG HƯ bằng classifier chuẩn (bắt cả 'bất túc'/'nhược'), không chỉ
            # regex 'dương (hư|suy)' — bản cũ bị tên đồng nghĩa 'Thận dương bất túc' lách qua và
            # còn được chọn làm ứng viên THAY THẾ (đề bạt chính từ đồng nghĩa của hội chứng bị hạ).
            _is_duong_hu = lambda n: bool(re.search(r'\bdương\b', n.lower())) \
                and self._syndrome_is_hu(n) and 'âm' not in n.lower()
            if not _has_cold_sign and _is_duong_hu(_grounded[0][0]):
                _alt_idx = next((i for i, (s, _sc) in enumerate(_grounded)
                                 if not _is_duong_hu(s) and self._syndrome_is_hu(s)),
                                None)
                if _alt_idx is not None:
                    logger.info(f"[CỔNG DƯƠNG HƯ] '{_grounded[0][0]}' làm cốt lõi nhưng lời khai không có "
                                f"dấu hàn -> hoán vị với '{_grounded[_alt_idx][0]}'")
                    _grounded.insert(0, _grounded.pop(_alt_idx))
            # Không dấu hàn -> loại hội chứng họ DƯƠNG HƯ khỏi vị trí KÈM THEO (hạng 2+): dương hư
            # kèm theo trên ca không hàn cũng vô căn cứ (đã xảy ra: cốt lõi Huyết hư + kèm 'Tỳ thận
            # dương hư' cho ca không sợ lạnh/tiểu đêm). Giữ hạng 1 (cốt lõi) theo điểm.
            if not _has_cold_sign and len(_grounded) > 1:
                _kept = [_grounded[0]] + [(s, sc) for s, sc in _grounded[1:] if not _is_duong_hu(s)]
                if len(_kept) < len(_grounded):
                    logger.info("[CỔNG DƯƠNG HƯ] Loại hội chứng dương-hư khỏi kèm theo (không có dấu hàn).")
                    _grounded = _kept

        final_syndromes = [s for s, _sc in _grounded] if _grounded else llm_syndromes

        # [FIX] Độc lập cưỡng bức Âm Dương Lưỡng Hư lên đầu khi có mâu thuẫn Hàn - Nhiệt lâm sàng
        symptoms_lower_all = (user_symptoms + " " + combined_query).lower()
        # Cùng chuẩn khớp (\b + gỡ bạn-hữu-giả) với khối Bát Cương trong _generate_explainable_answer
        # — nếu không, 'sốt' dính 'sốt ruột' ở đây có thể cưỡng bức Âm Dương Lưỡng Hư sai.
        has_cold_indicator = self._kw_hit_clean(symptoms_lower_all, ["sợ lạnh", "úy hàn", "sợ gió", "rét run"])
        has_heat_pulse_indicator = self._kw_hit_clean(
            symptoms_lower_all, ["mạch sác", "tế sác", "sác", "mạch trầm sác", "khát nước", "sốt", "đỏ bừng", "khô miệng"])
        # [GÁC NGOẠI CẢM BIỂU] Ca ngoại cảm cấp (Phong hàn/Phong nhiệt phạm biểu): SỐT + SỢ GIÓ/SỢ
        # LẠNH đồng thời là biểu hiện BIỂU CHỨNG kinh điển (chính-tà giao tranh ở biểu), TUYỆT ĐỐI
        # không phải hàn-nhiệt thác tạp nội thương → không được cưỡng bức 'Âm Dương Lưỡng Hư' (một
        # chứng hư tổn nội thương sâu) lên cốt lõi. Chỉ chặn khi hội chứng dẫn đầu đang là ngoại cảm biểu.
        _top_is_exterior = bool(final_syndromes) and self._syndrome_is_exterior_wind(final_syndromes[0])
        if has_cold_indicator and has_heat_pulse_indicator and not _top_is_exterior:
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
        if has_dao_han and has_duong_hu_sign and not _top_is_exterior:
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
            # [ƯU TIÊN CỐT LÕI HỢP CỰC HÀN] Khi lời khai có NHIỀU dấu hư hàn nội thương rõ (>=2) và
            # KHÔNG dấu nhiệt, ưu tiên đưa ứng viên HƯ họ DƯƠNG HƯ (Tỳ/Thận dương hư — định nghĩa bằng
            # hư hàn) lên cốt lõi thay vì hội chứng Hư TỔNG QUÁT đứng đầu (Khí hư/Huyết hư). CHỈ hoán
            # khi ứng viên dương hư có điểm grounded SÁT (>=90%) ứng viên Hư đầu — tránh kéo một dương
            # hư điểm thấp vô căn cứ lên. Cốt lõi Hư chung chung làm Bát Cương rụng trục Hàn và Mục 5
            # tra bài thuốc trượt khỏi thể dương hư (ca đại tiện lỏng + sợ lạnh + tay chân lạnh).
            if _hu_idx is not None and not has_heat_pulse_indicator:
                _cold_signs_bc = ["sợ lạnh", "úy hàn", "rét run", "tay chân lạnh", "chân tay lạnh",
                                  "chi lạnh", "lưng lạnh", "bụng lạnh", "lạnh bụng", "tiểu đêm",
                                  "ngũ canh", "phân sống", "đại tiện lỏng", "phân lỏng"]
                _cold_n_bc = sum(1 for _k in _cold_signs_bc if self._kw_hit_clean(symptoms_lower_all, [_k]))
                if _cold_n_bc >= 2:
                    _gsc = {s.lower(): sc for s, sc in (_grounded or [])}
                    _base_hu = _gsc.get(final_syndromes[_hu_idx].lower(), 0.0)
                    _is_duong_hu_core = lambda n: bool(re.search(r'\bdương\b', n.lower())) \
                        and self._syndrome_is_hu(n) and 'âm' not in n.lower()
                    if _base_hu > 0:
                        for _i, _s in enumerate(final_syndromes):
                            if _is_duong_hu_core(_s) and _gsc.get(_s.lower(), 0.0) >= _base_hu * 0.9:
                                if _i != _hu_idx:
                                    logger.info(f"[ƯU TIÊN CỐT LÕI HỢP CỰC HÀN] {_cold_n_bc} dấu hư hàn -> "
                                                f"ưu tiên '{_s}' làm cốt lõi thay '{final_syndromes[_hu_idx]}'")
                                _hu_idx = _i
                                break
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

        # [CỔNG HUYẾT Ứ] Nhãn họ HUYẾT Ứ ('Huyết ứ', 'Khí trệ huyết ứ', 'Đàm ứ'...) đòi hỏi bằng
        # chứng ứ trệ THỰC THỂ (lưỡi tím/tía/tối, điểm ứ huyết, môi tím, sắc mặt xạm, đau cố định/như
        # kim châm/cự án, mạch sáp, máu cục...). 'Khí trệ huyết ứ' là mega-node gom triệu chứng ~20
        # bệnh nên hút khớp bừa các than phiền tiêu hóa/hàn chung chung (buồn nôn, ợ hơi, đau bụng,
        # tay chân lạnh) rồi rò nhãn 'huyết ứ' vô căn cứ. Không MỘT dấu ứ nào -> loại khỏi vị trí KÈM
        # THEO (đối xứng [CỔNG DƯƠNG HƯ]); giữ hạng 1 (cốt lõi) theo điểm. 'Quầng đen dưới mắt' KHÔNG
        # tính là dấu ứ (đa nghĩa: thận hư/huyết hư/tâm tỳ hư), tránh vòng tự chứng minh.
        if final_syndromes and len(final_syndromes) > 1:
            _stasis_signs = [
                "lưỡi tím", "lưỡi tía", "lưỡi tối", "lưỡi xanh tím", "lưỡi có điểm", "điểm ứ huyết",
                "ứ huyết", "ban ứ huyết", "môi tím", "môi thâm tím", "sắc mặt xạm", "mặt xám xịt",
                "da xạm", "đau cố định", "đau như kim châm", "đau như dùi đâm", "đau xuyên",
                "đau cự án", "cự án", "mạch sáp", "máu cục", "huyết khối", "cục máu đông",
                "kinh có cục", "ban xuất huyết", "ban tím"]
            _has_stasis = self._kw_hit_clean(symptoms_lower_all, _stasis_signs)
            if not _has_stasis:
                _is_huyet_u = lambda n: bool(re.search(r'\bứ\b', n.lower())) or 'huyết ứ' in n.lower()
                _dropped_hu = [s for s in final_syndromes[1:] if _is_huyet_u(s)]
                if _dropped_hu:
                    logger.info(f"[CỔNG HUYẾT Ứ] Loại hội chứng huyết ứ khỏi kèm theo (không dấu ứ trệ): {_dropped_hu}")
                    final_syndromes = [final_syndromes[0]] + [s for s in final_syndromes[1:] if not _is_huyet_u(s)]

        # [CỔNG HÀN-NHIỆT THEO CỐT LÕI CUỐI] Chạy SAU mọi luật chọn/đảo cốt lõi (Bản Hư Tiêu Thực,
        # ngoại cảm, dương hư...): lời khai KHÔNG có dấu hàn/nhiệt nào -> lấy CỰC THEO TÊN của cốt
        # lõi ĐÃ CHỐT làm chuẩn, loại kèm theo THỰC-THUẦN mang cực đối lập chỉ-có-tên ('Phế khí hư
        # hàn' + kèm 'Phong nhiệt'/'Phế kinh phục nhiệt' không một dấu nhiệt nào — tên nhiệt đó còn
        # kéo nhãn thành 'Hàn Nhiệt Thác Tạp' qua corroboration tên, vòng tự chứng minh).
        if final_syndromes and len(final_syndromes) > 1:
            _txt_hc2 = (user_symptoms + " " + combined_query).lower()
            _cold2 = self._kw_hit_clean(_txt_hc2, ["sợ lạnh", "úy hàn", "rét run", "lạnh run"])
            _heat2 = self._kw_hit_clean(_txt_hc2, ["sốt", "phát nhiệt", "khát nước", "đỏ bừng", "mạch sác",
                                                   "tế sác", "rêu vàng", "tiểu vàng", "họng đỏ",
                                                   "đờm vàng", "mũi vàng", "vàng đục"])
            if not _cold2 and not _heat2:
                _top_l2 = final_syndromes[0].lower()
                _drop_kw2 = None
                if re.search(r'\bhàn\b', _top_l2) and not re.search(r'\b(nhiệt|hỏa|hoả)\b', _top_l2):
                    _drop_kw2 = ("nhiệt", "hỏa", "hoả")
                elif re.search(r'\b(nhiệt|hỏa|hoả)\b', _top_l2) and not re.search(r'\bhàn\b', _top_l2):
                    _drop_kw2 = ("hàn",)
                if _drop_kw2:
                    _dropped_hn2 = [s for s in final_syndromes[1:]
                                    if self._syndrome_is_thuc_pure(s) and any(k in s.lower() for k in _drop_kw2)]
                    if _dropped_hn2:
                        logger.info(f"[CỔNG HÀN-NHIỆT CUỐI] Loại kèm theo trái cực với cốt lõi đã chốt: {_dropped_hn2}")
                        final_syndromes = [final_syndromes[0]] + [s for s in final_syndromes[1:] if s not in _dropped_hn2]

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
            # Soi TOÀN BỘ hội chứng của bệnh (hoi_chung_all — sau khử trùng mỗi bệnh 1 ứng viên)
            # [FIX CỬA SỔ-TRƯỚC] Cùng chuẩn với khối bệnh danh Mục 1 trong _generate_explainable_answer:
            # cửa sổ ratio theo match TỐT NHẤT TOÀN CỤC (chief complaint) áp TRƯỚC lọc chéo hội chứng,
            # để bệnh khớp-hội-chứng ratio thấp không soán ngôi match chief-complaint mạnh (tránh
            # 'Trưng hà/Long bế/Tĩnh mạch viêm tắc' cho ca hô hấp). Giữ Mục 5 nhất quán với Mục 1.
            _global_best = matched_diseases[0]["ratio"] if matched_diseases else 0.0
            _window = [
                m for m in matched_diseases
                if m["ratio"] >= _global_best - 0.15 and m["ratio"] >= 0.30
            ]
            core_matched = [
                m for m in _window
                if core_syn and any(self._are_syndromes_related(core_syn, hc)
                                    for hc in m.get("hoi_chung_all", [m["hoi_chung"]]))
            ]
            any_matched = [
                m for m in _window
                if any(self._are_syndromes_related(vs, hc)
                       for vs in valid_syndromes
                       for hc in m.get("hoi_chung_all", [m["hoi_chung"]]))
            ]
            syndrome_matched_diseases = core_matched if core_matched else any_matched
            target_matches = syndrome_matched_diseases if syndrome_matched_diseases else _window

            if target_matches:
                filtered_matches = target_matches[:3]
                disease_names = list(dict.fromkeys([m["benh_ly"].strip() for m in filtered_matches]))
            
        # Danh sách triệu chứng HIỂN THỊ (tính sớm để đồ thị dùng cùng nhãn với input_fusion)
        _display_terms0 = self.qa_pipeline._preprocess_question(user_symptoms) if user_symptoms else []
        all_symptoms_list = self._resolve_symptom_conflicts(
            list(dict.fromkeys(_display_terms0 + (raw_vision_data["detected_symptoms"] if raw_vision_data else [])))
        )
        # [ĐỒ THỊ NHẤT QUÁN] Triệu chứng khớp THẬT theo TỪNG hội chứng — trước đây gán nguyên
        # search_terms cho mọi hội chứng khiến frontend nối mọi triệu chứng vào mọi hội chứng
        # thành mạng nhện không đọc được và trông như mọi hội chứng đều khớp 100%.
        _graph_matched = self._matched_terms_by_syndrome(all_symptoms_list, final_syndromes)

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
                "matching_symptoms": _graph_matched.get(syn, []),
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
        # [NHÃN BÀI = PHÁP TRỊ] Tên bài là pháp-trị TRẦN (vd 'Dưỡng âm thanh nhiệt hoạt huyết') ->
        # 'Dùng bài <X>' đọc khó hiểu; đổi thành 'Dùng bài theo pháp <X>'. Phương danh thật giữ nguyên.
        final_markdown = self._relabel_bare_phaptri_bai(final_markdown)

        # [CẢNH BÁO MÂU THUẪN GIỚI] Khai giới nhưng lời khai có dấu đặc thù giới KHÁC (vd Nam + 'âm
        # hộ'): cổng giới đã LOẠI bệnh khác giới nên kết quả có thể lệch/generic (Mục 5 trắng) — báo
        # rõ để người dùng sửa nhập liệu thay vì tin chẩn đoán sai.
        _sx_conflict = self._sex_symptom_conflict(user_symptoms)
        if _sx_conflict:
            _sex_vn = "Nam" if getattr(self, "_patient_sex", None) == "nam" else "Nữ"
            final_markdown = (
                f"> ⚠️ **Mâu thuẫn giới tính ↔ triệu chứng:** bạn khai giới **{_sex_vn}** nhưng lời "
                f"khai có triệu chứng đặc thù giới KHÁC: **{', '.join(_sx_conflict)}**. Hệ đã LOẠI "
                f"bệnh không phù hợp giới khai báo — kết quả bên dưới có thể chưa sát. Vui lòng kiểm "
                f"tra lại **giới tính** hoặc **triệu chứng**.\n\n"
            ) + final_markdown

        # [CẢNH BÁO INPUT MỎNG] Đông y chẩn bằng TỨ CHẨN (nhiều dấu). Chỉ 1-2 triệu chứng — nhất là
        # dấu CHUNG ('tiểu nhiều','mệt mỏi'...) — KHÔNG đủ chốt bệnh danh+hội chứng tin cậy (cùng 1
        # dấu gặp ở nhiều bệnh). Không hỏi lại (cơ chế hỏi động đã gỡ) -> CẢNH BÁO input mỏng, nêu rõ
        # kết quả chỉ ĐỊNH HƯỚNG SƠ BỘ + gợi ý bổ sung. Đếm dấu THẬT (lời khai + vọng chẩn ảnh).
        _n_signs = len(all_symptoms_list)
        if _n_signs <= 2:
            _sl = ", ".join(all_symptoms_list) if all_symptoms_list else "(trống)"
            final_markdown = (
                f"> ⚠️ **Lời khai quá ít triệu chứng ({_n_signs} dấu: {_sl}).** Đông y chẩn đoán bằng "
                f"TỨ CHẨN (vọng–văn–vấn–thiết); một–hai dấu đơn lẻ (nhất là dấu CHUNG) KHÔNG đủ để chốt "
                f"bệnh danh + hội chứng tin cậy — cùng một triệu chứng có thể gặp ở NHIỀU bệnh khác nhau. "
                f"Kết quả bên dưới chỉ là **ĐỊNH HƯỚNG SƠ BỘ**, độ tin cậy thấp. Vui lòng bổ sung: hàn/nhiệt "
                f"(sợ lạnh/sợ nóng), khát nước, mồ hôi, ăn–ngủ, đại–tiểu tiện (màu/lượng), vị trí đau, thời "
                f"gian mắc, và ảnh lưỡi/sắc mặt… để chẩn đoán chính xác hơn.\n\n"
            ) + final_markdown

        # [AN TOÀN Y TẾ] Luôn chèn cảnh báo cấp cứu (nếu có cờ đỏ) + miễn trừ trách nhiệm
        final_markdown = self._append_medical_disclaimer(
            final_markdown, raw_text=f"{user_symptoms} {combined_query}"
        )

        # Chuỗi triệu chứng HIỂN THỊ (all_symptoms_list đã tính ở trên, trước khối detailed_kg_data
        # — chỉ triệu chứng người dùng thật nhập + vọng chẩn, KHÔNG bung synonym nội bộ).
        input_fusion_str = ", ".join(all_symptoms_list) if all_symptoms_list else (combined_query or "Không xác định")

        # [VẤN CHẨN CHUYÊN SÂU] KHÔNG chặn kết quả (khác hẳn 'hỏi bệnh động' đã gỡ): tính SAU khi đã
        # có chẩn đoán trực tiếp, chỉ GỢI Ý yếu tố đáng hỏi (mang thai/giới + triệu chứng đặc hiệu
        # bệnh nghi) để người dùng chọn -> frontend gộp vào lời khai + tự phân tích lại tinh chỉnh.
        deep_inquiry = self._compute_deep_inquiry(user_symptoms, all_symptoms_list, all_symptoms_list)

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
            "deep_inquiry": deep_inquiry,
            "status": "completed",
            "questions": []
        }

    def close(self):
        if hasattr(self, 'qa_pipeline'): self.qa_pipeline.close()
        if hasattr(self, 'vision_pipeline') and hasattr(self.vision_pipeline, 'close'): self.vision_pipeline.close()
        logger.info("Đã giải phóng tài nguyên.")
