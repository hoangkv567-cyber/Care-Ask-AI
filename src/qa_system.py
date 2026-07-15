# src/qa_system.py
import os
import json
import logging
import re
from neo4j import GraphDatabase
import ollama
from src.config_loader import load_config
from src.utils import normalize_symptoms_text

logger = logging.getLogger(__name__)

class TCMQA:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.llm_model = self.config.get("qa", {}).get("llm_model", "qwen2.5:7b")
        self.temperature = self.config.get("qa", {}).get("temperature", 0.0)
        self.seed = self.config.get("qa", {}).get("seed", 42)
        self.top_p = self.config.get("qa", {}).get("top_p", 0.9)

        # Cấu hình sử dụng SiliconFlow, OpenRouter, Hugging Face Cloud hoặc Ollama
        siliconflow_cfg = self.config.get("siliconflow", {})
        openrouter_cfg = self.config.get("openrouter", {})
        hf_cfg = self.config.get("huggingface", {})
        import os
        
        if siliconflow_cfg.get("use_cloud", False):
            token = os.environ.get("SILICONFLOW_API_KEY") or siliconflow_cfg.get("api_key")
            model_id = siliconflow_cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")
            self.llm_model = model_id
            
            class SiliconFlowChatClient:
                def __init__(self, token_val: str, model_val: str, proxy: str = None,
                             fallback_model: str = None, fallback_host: str = None):
                    self.token = token_val
                    self.model_id = model_val
                    self.url = "https://api.siliconflow.com/v1/chat/completions"
                    # [FALLBACK LOCAL] Model ollama dùng khi cloud lỗi (403 hết số dư/mạng) để TEXT LLM
                    # (Mục 3/4 + trích xuất hội chứng) vẫn chạy — đối xứng fallback LLaVA của phần ảnh.
                    self.fallback_model = fallback_model
                    self.fallback_host = fallback_host
                    self._ollama = None
                    import httpx
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    }
                    if proxy:
                        self.http_client = httpx.Client(proxies=proxy, headers=headers, timeout=120.0)
                        logger.info(f"Khởi tạo SiliconFlow Client cho model {model_val} qua proxy: {proxy}")
                    else:
                        self.http_client = httpx.Client(headers=headers, timeout=120.0)
                        logger.info(f"Khởi tạo SiliconFlow Client cho model: {model_val}")

                def chat(self, model: str, messages: list, options: dict = None) -> dict:
                    temperature = 0.0
                    if options:
                        if "temperature" in options:
                            temperature = options["temperature"]
                        if temperature == 0.0:
                            temperature = 0.01
                    payload = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False
                    }
                    if options and "max_tokens" in options:
                        payload["max_tokens"] = options["max_tokens"]
                    try:
                        response = self.http_client.post(self.url, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "message": {
                                "role": "assistant",
                                "content": content
                            }
                        }
                    except Exception as e:
                        logger.error(f"Lỗi gọi SiliconFlow API: {e}")
                        if 'response' in locals() and response is not None:
                            logger.error(f"Chi tiết phản hồi lỗi: {response.text}")
                        # [FALLBACK OLLAMA LOCAL] Cloud lỗi (403 hết số dư / mạng) -> chuyển TEXT LLM
                        # sang ollama local để Mục 3/4 + trích xuất hội chứng KHÔNG chết (miễn phí,
                        # độc lập số dư). Chỉ raise nếu ollama cũng lỗi.
                        if self.fallback_model:
                            try:
                                if self._ollama is None:
                                    if self.fallback_host:
                                        from ollama import Client as _OllamaClient
                                        self._ollama = _OllamaClient(host=self.fallback_host)
                                    else:
                                        import ollama as _ollama_mod
                                        self._ollama = _ollama_mod
                                _opts = {"temperature": temperature}
                                if options and "seed" in options:
                                    _opts["seed"] = options["seed"]
                                if options and "max_tokens" in options:
                                    _opts["num_predict"] = options["max_tokens"]
                                logger.warning(f"[FALLBACK LLM] SiliconFlow lỗi -> chuyển ollama local "
                                               f"'{self.fallback_model}' cho TEXT LLM.")
                                _resp = self._ollama.chat(model=self.fallback_model, messages=messages,
                                                          options=_opts)
                                _content = (_resp["message"]["content"] if isinstance(_resp, dict)
                                            else _resp.message.content)
                                # Model local 7B hay thêm lời rào ("Tôi hiểu... Dưới đây là ví dụ...
                                # ---") trước nội dung thật -> cắt preamble để không lọt vào Mục 3/4.
                                import re as _re_fb
                                _content = _re_fb.sub(
                                    r'^\s*(?:(?:tôi hiểu|dưới đây|chắc chắn|vâng|được|tất nhiên|sure|'
                                    r'certainly|here (?:is|are)|okay|ok)[^\n]*\n)+\s*(?:[-–—]{3,}\s*\n)?',
                                    '', _content, flags=_re_fb.IGNORECASE).strip()
                                return {"message": {"role": "assistant", "content": _content}}
                            except Exception as e2:
                                logger.error(f"[FALLBACK LLM] ollama local cũng lỗi: {e2}")
                        raise e

            proxy = siliconflow_cfg.get("proxy")
            # Model ollama local làm fallback khi cloud lỗi (mặc định theo llm_model của config).
            _fb_model = (siliconflow_cfg.get("fallback_ollama_model")
                         or self.config.get("llm_model") or "qwen2.5:7b")
            _fb_host = self.config.get("host") or self.config.get("ollama", {}).get("host")
            self.client = SiliconFlowChatClient(token, model_id, proxy, _fb_model, _fb_host)
            logger.info(f"TCMQA kết nối SiliconFlow thành công! (fallback ollama: {_fb_model})")
            
        elif openrouter_cfg.get("use_cloud", False):
            token = os.environ.get("OPENROUTER_API_KEY") or openrouter_cfg.get("api_key")
            model_id = openrouter_cfg.get("model", "qwen/qwen-2.5-vl-72b-instruct:free")
            self.llm_model = model_id
            
            class OpenRouterChatClient:
                def __init__(self, token_val: str, model_val: str, proxy: str = None):
                    self.token = token_val
                    self.model_id = model_val
                    self.url = "https://openrouter.ai/api/v1/chat/completions"
                    import httpx
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "TCM-Qwen-QA"
                    }
                    if proxy:
                        self.http_client = httpx.Client(proxies=proxy, headers=headers, timeout=60.0)
                        logger.info(f"Khởi tạo OpenRouter Client cho model {model_val} qua proxy: {proxy}")
                    else:
                        self.http_client = httpx.Client(headers=headers, timeout=60.0)
                        logger.info(f"Khởi tạo OpenRouter Client cho model: {model_val}")

                def chat(self, model: str, messages: list, options: dict = None) -> dict:
                    temperature = 0.0
                    if options:
                        if "temperature" in options:
                            temperature = options["temperature"]
                        if temperature == 0.0:
                            temperature = 0.01
                    payload = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False
                    }
                    if options and "max_tokens" in options:
                        payload["max_tokens"] = options["max_tokens"]
                    try:
                        response = self.http_client.post(self.url, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "message": {
                                "role": "assistant",
                                "content": content
                            }
                        }
                    except Exception as e:
                        logger.error(f"Lỗi gọi OpenRouter API: {e}")
                        if 'response' in locals() and response is not None:
                            logger.error(f"Chi tiết phản hồi lỗi: {response.text}")
                        raise e

            proxy = openrouter_cfg.get("proxy")
            self.client = OpenRouterChatClient(token, model_id, proxy)
            logger.info("TCMQA kết nối OpenRouter thành công!")
            
        elif hf_cfg.get("use_cloud", False):
            # Khởi tạo client Hugging Face Serverless
            token = os.environ.get("HF_TOKEN") or hf_cfg.get("token")
            model_id = hf_cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")
            self.llm_model = model_id
            
            class HuggingFaceChatClient:
                def __init__(self, token_val: str, model_val: str, proxy: str = None,
                             fallback_model: str = None, fallback_host: str = None):
                    self.token = token_val
                    self.model_id = model_val
                    self.url = "https://router.huggingface.co/v1/chat/completions"
                    # [FALLBACK LOCAL] ollama khi HF lỗi (403 thiếu quyền / hết credit / mạng).
                    self.fallback_model = fallback_model
                    self.fallback_host = fallback_host
                    self._ollama = None
                    import httpx
                    if proxy:
                        self.http_client = httpx.Client(proxies=proxy, timeout=60.0)
                        logger.info(f"Khởi tạo HuggingFace Serverless Client cho model {model_val} qua proxy: {proxy}")
                    else:
                        self.http_client = httpx.Client(timeout=60.0)
                        logger.info(f"Khởi tạo HuggingFace Serverless Client cho model: {model_val}")

                def chat(self, model: str, messages: list, options: dict = None) -> dict:
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json"
                    }
                    temperature = 0.0
                    if options:
                        if "temperature" in options:
                            temperature = options["temperature"]
                        if temperature == 0.0:
                            temperature = 0.01
                    payload = {
                        "model": self.model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False
                    }
                    if options and "max_tokens" in options:
                        payload["max_tokens"] = options["max_tokens"]
                    try:
                        response = self.http_client.post(self.url, headers=headers, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "message": {
                                "role": "assistant",
                                "content": content
                            }
                        }
                    except Exception as e:
                        logger.error(f"Lỗi gọi Hugging Face Serverless API: {e}")
                        if 'response' in locals() and response is not None:
                            logger.error(f"Chi tiết phản hồi lỗi: {response.text}")
                        # [FALLBACK OLLAMA LOCAL] HF lỗi (403/credit/mạng) -> ollama local để TEXT LLM
                        # không chết. Chỉ raise nếu ollama cũng lỗi.
                        if self.fallback_model:
                            try:
                                if self._ollama is None:
                                    if self.fallback_host:
                                        from ollama import Client as _OllamaClient
                                        self._ollama = _OllamaClient(host=self.fallback_host)
                                    else:
                                        import ollama as _ollama_mod
                                        self._ollama = _ollama_mod
                                _opts = {"temperature": temperature}
                                if options and "seed" in options:
                                    _opts["seed"] = options["seed"]
                                if options and "max_tokens" in options:
                                    _opts["num_predict"] = options["max_tokens"]
                                logger.warning(f"[FALLBACK LLM] HF lỗi -> chuyển ollama local "
                                               f"'{self.fallback_model}' cho TEXT LLM.")
                                _resp = self._ollama.chat(model=self.fallback_model, messages=messages,
                                                          options=_opts)
                                _content = (_resp["message"]["content"] if isinstance(_resp, dict)
                                            else _resp.message.content)
                                import re as _re_fb
                                _content = _re_fb.sub(
                                    r'^\s*(?:(?:tôi hiểu|dưới đây|chắc chắn|vâng|được|tất nhiên|sure|'
                                    r'certainly|here (?:is|are)|okay|ok)[^\n]*\n)+\s*(?:[-–—]{3,}\s*\n)?',
                                    '', _content, flags=_re_fb.IGNORECASE).strip()
                                return {"message": {"role": "assistant", "content": _content}}
                            except Exception as e2:
                                logger.error(f"[FALLBACK LLM] ollama local cũng lỗi: {e2}")
                        raise e

            proxy = hf_cfg.get("proxy")
            _fb_model = (hf_cfg.get("fallback_ollama_model")
                         or self.config.get("llm_model") or "qwen2.5:7b")
            _fb_host = self.config.get("host") or self.config.get("ollama", {}).get("host")
            self.client = HuggingFaceChatClient(token, model_id, proxy, _fb_model, _fb_host)
            logger.info(f"TCMQA kết nối Hugging Face thành công! (fallback ollama: {_fb_model})")
        else:
            # Hỗ trợ host remote
            self.ollama_host = self.config.get("host") or self.config.get("ollama", {}).get("host")
            if self.ollama_host:
                from ollama import Client
                self.client = Client(host=self.ollama_host)
                logger.info(f"TCMQA kết nối Ollama remote host: {self.ollama_host}")
            else:
                import ollama
                self.client = ollama
                logger.info("TCMQA kết nối Ollama local host")

        # Kết nối Neo4j (secrets đã được config_loader bơm từ .env)
        neo4j_cfg = self.config.get("neo4j", {})
        neo4j_uri = neo4j_cfg.get("uri") or os.getenv("NEO4J_URI")
        neo4j_user = neo4j_cfg.get("user") or os.getenv("NEO4J_USER")
        neo4j_password = neo4j_cfg.get("password") or os.getenv("NEO4J_PASSWORD")
        if not (neo4j_uri and neo4j_user and neo4j_password):
            raise ValueError(
                "Thiếu thông tin kết nối Neo4j. Hãy đặt NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD "
                "trong file .env (xem .env.example)."
            )
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        logger.info(f"Đã kết nối Neo4j: {neo4j_uri}")

        # Đọc schema
        with open("data/graph_schema.txt", "r", encoding="utf-8") as f:
            self.schema = f.read()
        self.db_schema = self.schema
        logger.info(f"Đã tải schema từ data/graph_schema.txt")

        # Danh sách nhãn và quan hệ để LLM biết
        self.node_labels = ["BenhLy", "HoiChung", "TrieuChung", "BaiThuoc", "ViThuoc"]
        self.rel_types = ["CHIA_THÀNH", "CÓ_BIỂU_HIỆN", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG", "BAO_GỒM"]

    def close(self):
        self.driver.close()

    def _extract_symptoms_by_matching(self, text: str) -> list:
        if not text:
            return []
        text_lower = text.lower()
        
        # Lấy tất cả triệu chứng từ database
        try:
            db_symptoms = self.neo4j_client.get_all_symptoms()
        except Exception as e:
            try:
                # Nếu neo4j_client.get_all_symptoms() không có/chưa khởi tạo, query trực tiếp từ self.driver
                with self.driver.session() as session:
                    result = session.run("MATCH (t:TrieuChung) RETURN t.name AS name")
                    db_symptoms = [record["name"] for record in result]
            except Exception as ex:
                logger.error(f"Lỗi truy vấn tất cả triệu chứng: {ex}")
                db_symptoms = []
            
        # Sắp xếp theo chiều dài giảm dần để ưu tiên triệu chứng dài trước
        db_symptoms_sorted = sorted(db_symptoms, key=len, reverse=True)
        
        extracted = []
        temp_text = text_lower
        
        # Danh sách các từ đơn lẻ đặc trưng trong y học Đông y được phép khớp trực tiếp
        whitelist_1word = {
            "sốt", "ho", "nôn", "mệt", "khát", "trướng", "chướng", "ngứa", 
            "phù", "táo", "bón", "tê", "loét", "run", "nấc", "đau", "lỵ"
        }
        
        # [CHẶN MẢNH VỠ ĐỊNH TÍNH] Node TrieuChung rác sinh ra do mô tả CSV bị tách theo dấu phẩy
        # ('...xuất hiện nhiều, liên tục, lâu ngày...' -> node 'liên tục'). Chúng là TRẠNG TỪ định
        # tính, không phải triệu chứng — nếu để khớp, chúng nuốt mất cụm thật ('ho liên tục' bị xé
        # thành 'liên tục' + 'ho') và chuỗi gộp hiển thị vô nghĩa. Chỉ chặn khi TOÀN BỘ tên node
        # bằng đúng mảnh vỡ; cụm dài chứa chúng ('ho kéo dài') vẫn khớp bình thường.
        qualifier_fragments = {
            "liên tục", "kéo dài", "thường xuyên", "lâu ngày", "từng cơn", "nhiều lần",
            "tái phát", "dữ dội", "âm ỉ", "đột ngột", "dai dẳng", "về đêm", "ban đêm",
        }

        spans = []   # (start, end, symptom) — vị trí mỗi triệu chứng đã trích trên text_lower
        for symptom in db_symptoms_sorted:
            sym_l = symptom.lower()
            if sym_l.strip() in qualifier_fragments:
                continue
            if len(sym_l.split()) < 2:
                if sym_l not in whitelist_1word:
                    continue
            # Dùng regex \b để khớp từ độc lập tránh substring trượt
            pattern = rf'\b{re.escape(sym_l)}\b'
            m = re.search(pattern, temp_text)
            if m:
                extracted.append(symptom)
                spans.append((m.start(), m.end(), symptom))
                # Thay bằng khoảng trắng CÙNG CHIỀU DÀI (giữ nguyên tọa độ) để cụm khác không đè trùng
                temp_text = temp_text[:m.start()] + (" " * (m.end() - m.start())) + temp_text[m.end():]

        # [CHỐNG PHỦ ĐỊNH VĂN BẢN] Loại triệu chứng bị lời khai phủ định ('không sốt, ho khan' -> bỏ
        # 'sốt'). Chạy SAU longest-match: các tên triệu chứng vốn chứa 'không' ('miệng nhạt không
        # khát', 'tay chân không ấm'...) đã được gom nguyên cụm và xóa khỏi temp_text, nên chữ
        # 'không' còn sót trong temp_text mới đúng là phủ định TỰ DO của người bệnh.
        return self._drop_negated_text_symptoms(extracted, spans, temp_text)

    # 'không những'/'không chỉ' = 'không riêng' (nhấn mạnh CÓ), KHÔNG phải phủ định triệu chứng.
    _NEG_TEXT_CUE = re.compile(r'\b(?:không|chẳng|chả|chưa|ko)\b')
    # Dấu ngắt mệnh đề / liên từ đối lập — phủ định KHÔNG vươn qua các ranh giới này.
    _NEG_TEXT_STOP = re.compile(r'[,.;:!?]|\b(?:nhưng|mà|còn|song|tuy)\b')

    def _drop_negated_text_symptoms(self, extracted: list, spans: list, blanked_text: str) -> list:
        """Bỏ khỏi 'extracted' các triệu chứng đứng trong tầm phủ định của một chữ 'không/chưa/
        chẳng' TỰ DO (không thuộc tên triệu chứng nào). Tầm phủ định = từ chữ phủ định tới dấu ngắt
        mệnh đề gần nhất (hoặc tối đa 40 ký tự). 'sốt, không ho' -> bỏ 'ho', GIỮ 'sốt'."""
        if not spans:
            return extracted
        cues = []
        for m in self._NEG_TEXT_CUE.finditer(blanked_text):
            tail = blanked_text[m.end():m.end() + 8].lstrip()
            if tail.startswith("những") or tail.startswith("chỉ"):   # 'không những/chỉ' -> bỏ qua
                continue
            cues.append(m.start())
        if not cues:
            return extracted
        negated = set()
        for (s, _e, name) in spans:
            for c in cues:
                if c >= s:                                   # phủ định phải đứng TRƯỚC triệu chứng
                    continue
                gap = blanked_text[c:s]
                if len(gap) > 40 or self._NEG_TEXT_STOP.search(gap):
                    continue                                 # quá xa / có ranh giới mệnh đề -> không tới
                negated.add(name)
                break
        if negated:
            logger.info(f"[CHỐNG PHỦ ĐỊNH VĂN BẢN] Loại triệu chứng bị phủ định: {sorted(negated)}")
        return [x for x in extracted if x not in negated]

    def _preprocess_question(self, question: str) -> list:
        # Chuẩn hóa văn bản trước khi khớp
        normalized_q = normalize_symptoms_text(question)
        # Sử dụng thuật toán khớp triệu chứng trực tiếp từ database (Longest Match First)
        return self._extract_symptoms_by_matching(normalized_q)

    def text_to_cypher(self, user_question: str, terms: list = None) -> str:
        if terms is None:
            terms = self._preprocess_question(user_question)
        if not terms:
            return ""
            
        symptoms_str = ", ".join([f"'{t}'" for t in terms])
        
        prompt = f"""
        You are an expert Neo4j Cypher developer for a Traditional Chinese Medicine database.
        Schema:
        {self.db_schema}
        
        CRITICAL RULES FOR CYPHER GENERATION:
        1. USE THE PROVIDED SYMPTOMS ONLY: Generate the Cypher query using exactly the following medical symptoms extracted from the user's question: {symptoms_str}.
           Do NOT extract, change, or add any other symptoms.
        
        2. EXACT WORD MATCHING (CRITICAL): NEVER use CONTAINS to prevent substring bugs. You MUST use Unicode-aware word boundaries `(^|[^\\p{{L}}])keyword($|[^\\p{{L}}])`.
            Syntax: WHERE toLower(t.name) =~ '.*(^|[^\\p{{L}}])keyword($|[^\\p{{L}}]).*' (keyword must be in lowercase, e.g., 'ho' instead of 'Ho')
            Example: For symptom 'ho liên tục', use: WHERE toLower(t.name) =~ '.*(^|[^\\p{{L}}])ho liên tục($|[^\\p{{L}}]).*'
         
                  3. MANDATORY RETURN STRUCTURE (NO EXCEPTIONS):
            MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
            WHERE toLower(t.name) =~ '.*(^|[^\\p{{L}}])keyword1($|[^\\p{{L}}]).*'
              AND r.benh_ly = b.name
            OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
            WHERE p.benh_ly = b.name AND p.hoi_chung = h.name
            RETURN DISTINCT b.name, h.name, p.name
         
                  4. FOR MULTIPLE SYMPTOMS: Use multiple MATCH clauses.
            Example for symptoms 'ho khan' and 'đau đầu':
            MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)
            MATCH (h)-[r1:CÓ_BIỂU_HIỆN]->(t1:TrieuChung) WHERE toLower(t1.name) =~ '.*(^|[^\\p{{L}}])ho khan($|[^\\p{{L}}]).*' AND r1.benh_ly = b.name
            MATCH (h)-[r2:CÓ_BIỂU_HIỆN]->(t2:TrieuChung) WHERE toLower(t2.name) =~ '.*(^|[^\\p{{L}}])đau đầu($|[^\\p{{L}}]).*' AND r2.benh_ly = b.name
            OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
            WHERE p.benh_ly = b.name AND p.hoi_chung = h.name
            RETURN DISTINCT b.name, h.name, p.name
        

        5. Output ONLY a SINGLE raw Cypher query. Do NOT use markdown code blocks (like ```cypher). No explanations.
        
        User Question: "{user_question}"
        Cypher:
        """
        try:
            response = self.client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precise Cypher query generator. Output only the raw query without any markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.0,  # Đưa temperature về 0.0 để loại bỏ hoàn toàn tính ngẫu nhiên
                    "seed": self.seed,
                    "top_p": self.top_p
                }
            )
            cypher = response['message']['content'].strip()
            # Dọn dẹp ký tự markdown nếu LLM lỡ sinh ra
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            
            # Sửa các lỗi chính tả/dấu cách trong tên mối quan hệ do LLM sinh ra
            cypher = cypher.replace("CÓ BIỂU HIỆN", "CÓ_BIỂU_HIỆN")
            cypher = cypher.replace("CÓ_BIỂU HIỆN", "CÓ_BIỂU_HIỆN")
            cypher = cypher.replace("CÓ BIỂU_HIỆN", "CÓ_BIỂU_HIỆN")
            cypher = cypher.replace("CHIA THÀNH", "CHIA_THÀNH")
            cypher = cypher.replace("ĐƯỢC ĐIỀU TRỊ BẰNG", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG")
            cypher = cypher.replace("ĐƯỢC_ĐIỀU_TRỊ BẰNG", "ĐƯỢC_ĐIỀU_TRỊ_BẰNG")
            cypher = cypher.replace("BAO GỒM", "BAO_GỒM")
            cypher = cypher.replace("BAO GÔM", "BAO_GỒM")
            cypher = cypher.replace("BAO_GÔM", "BAO_GỒM")
            
            return cypher
        except Exception as e:
            logger.error(f"Lỗi sinh Cypher: {e}")
            return ""

    # Từ khóa GHI/thủ tục — cấm tuyệt đối ở chế độ web read-only (chống Cypher injection
    # khi LLM tự sinh truy vấn). Các truy vấn tra cứu hợp lệ không bao giờ dùng những từ này.
    _CYPHER_WRITE_KEYWORDS = (
        "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
        "FOREACH", "CALL", "LOAD", "PERIODIC", "IMPORT",
        "GRANT", "DENY", "REVOKE", "ALTER", "RENAME", "TERMINATE",
    )

    @staticmethod
    def _strip_cypher_literals(cypher: str) -> str:
        """Bỏ chuỗi trong nháy, backtick và comment để quét từ khóa không bị nhầm với
        nội dung dữ liệu (vd triệu chứng chứa chữ 'set' nằm trong '...' không được tính)."""
        s = re.sub(r"/\*.*?\*/", " ", cypher, flags=re.S)   # block comment
        s = re.sub(r"//[^\n]*", " ", s)                       # line comment
        s = re.sub(r"'(?:\\.|[^'\\])*'", " ", s)              # chuỗi '...'
        s = re.sub(r'"(?:\\.|[^"\\])*"', " ", s)              # chuỗi "..."
        s = re.sub(r"`(?:[^`])*`", " ", s)                    # định danh `...`
        return s

    @classmethod
    def _is_read_only_cypher(cls, cypher: str) -> bool:
        """True nếu Cypher CHỈ đọc: không chứa từ khóa ghi/thủ tục và chỉ có 1 câu lệnh."""
        if not cypher or not cypher.strip():
            return False
        stripped = cls._strip_cypher_literals(cypher)
        # Chặn nhiều câu lệnh (chỉ cho phép dấu ; ở cuối) -> tránh "MATCH ...; DROP ..."
        if ";" in stripped.rstrip().rstrip(";"):
            return False
        upper = stripped.upper()
        for kw in cls._CYPHER_WRITE_KEYWORDS:
            if re.search(r"(?<![A-Z0-9_])" + kw + r"(?![A-Z0-9_])", upper):
                return False
        return True

    def run_cypher(self, cypher: str, params: dict = None, read_only: bool = False) -> list:
        """Thực thi Cypher trên Neo4j (hỗ trợ tham số $param để tránh Cypher injection).

        read_only=True (dùng cho luồng web hỏi tự do): chặn TĨNH mọi từ khóa ghi/thủ tục,
        VÀ chạy trong giao dịch READ để Neo4j server tự từ chối mọi thao tác ghi (2 lớp).
        Ném ValueError nếu câu Cypher không phải read-only để caller xử lý an toàn."""
        if not cypher:
            return []
        if read_only and not self._is_read_only_cypher(cypher):
            logger.warning(f"Từ chối Cypher không phải read-only (web): {cypher}")
            raise ValueError("Chỉ cho phép truy vấn đọc (read-only).")
        try:
            with self.driver.session() as session:
                if read_only:
                    return session.execute_read(
                        lambda tx: [dict(record) for record in tx.run(cypher, **(params or {}))]
                    )
                result = session.run(cypher, **(params or {}))
                return [dict(record) for record in result]
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Lỗi thực thi Cypher: {e}\nCypher: {cypher}")
            return []

    # ====================================================================
    # [FIX] TRUY HỒI NHẤT QUÁN THEO ĐỒ THỊ (giữ đúng ngữ cảnh Bệnh–Hội chứng–Bài thuốc)
    # Khắc phục 2 lỗi của pipeline web:
    #   - "Sai bài thuốc": trước đây lấy LIMIT 1 bài thuốc bất kỳ của hội chứng dùng chung.
    #   - "Thiếu kết quả": trước đây khớp '=' chính xác nên bỏ sót triệu chứng nằm trong node ghép.
    # ====================================================================
    @staticmethod
    def _word_boundary_pattern(term: str) -> str:
        """Tạo Java-regex khớp 'term' như một từ độc lập (tránh dính chuỗi con),
        bắt được cả khi term nằm trong node triệu chứng ghép."""
        # \Q...\E: trích dẫn nguyên văn, an toàn với ký tự đặc biệt như dấu ngoặc.
        return r'.*(^|[^\p{L}])\Q' + term.lower() + r'\E($|[^\p{L}]).*'

    def get_symptom_disease_map(self, terms: list) -> dict:
        """Trả về {hội chứng: [danh sách bệnh]} mà triệu chứng người bệnh THỰC SỰ thuộc về.
        Đi đúng đường đồ thị và ràng buộc r.benh_ly = b.name để loại các bệnh không liên quan."""
        mapping = {}
        for term in (terms or []):
            if not term:
                continue
            pattern = self._word_boundary_pattern(term)
            # CHẶT: bắt buộc r.benh_ly = b.name để cột chặt triệu chứng vào đúng bệnh của nó.
            # (DB có ~49% cạnh benh_ly=NULL trùng lặp; nếu nới 'IS NULL OR' sẽ khớp tràn sang mọi
            # bệnh dùng chung hội chứng. Đã kiểm chứng: 0 cặp (HC,triệu chứng) chỉ có bản NULL,
            # nên ràng buộc chặt KHÔNG bỏ sót kết quả nào.)
            cypher = """
            MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
            WHERE toLower(t.name) =~ $pattern AND r.benh_ly = b.name
            RETURN DISTINCT h.name AS syndrome, b.name AS disease
            """
            try:
                with self.driver.session() as session:
                    for rec in session.run(cypher, pattern=pattern):
                        if rec["syndrome"] and rec["disease"]:
                            mapping.setdefault(rec["syndrome"], set()).add(rec["disease"])
            except Exception as e:
                logger.error(f"Lỗi get_symptom_disease_map (term='{term}'): {e}")
        return {k: sorted(v) for k, v in mapping.items()}

    def get_treatments_for_syndrome(self, syndrome: str, diseases: list = None) -> list:
        """Trả về [{disease, bai_thuoc, vi_thuoc}] NHẤT QUÁN: mỗi bệnh đi kèm ĐÚNG bài thuốc của nó
        (ràng buộc p.benh_ly = b.name AND p.hoi_chung = h.name). Nếu diseases=None thì lấy mọi bệnh."""
        # [FIX CASE-INSENSITIVE] KG có ~24 nhóm HoiChung trùng tên chỉ khác hoa/thường
        # ('Tỳ Thận Dương hư' vs 'Tỳ thận dương hư') làm exact-match trượt bài thuốc oan.
        # Khớp toLower để gom đủ mọi biến thể; ràng buộc p.hoi_chung = h.name vẫn giữ per-node
        # nên mỗi biến thể chỉ trả đúng bài thuốc của chính nó (không khớp tràn).
        cypher = """
        MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)
        WHERE toLower(h.name) = toLower($syn)
          AND ($diseases IS NULL OR b.name IN $diseases)
        OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
          WHERE p.benh_ly = b.name AND p.hoi_chung = h.name
        OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
        RETURN b.name AS disease, p.name AS bai_thuoc, collect(DISTINCT v.name) AS vi_thuoc
        ORDER BY disease
        """
        out = []
        try:
            with self.driver.session() as session:
                for rec in session.run(cypher, syn=syndrome, diseases=diseases):
                    out.append({
                        "disease": rec["disease"],
                        "bai_thuoc": rec["bai_thuoc"],
                        "vi_thuoc": [v for v in (rec["vi_thuoc"] or []) if v],
                    })
        except Exception as e:
            logger.error(f"Lỗi get_treatments_for_syndrome ('{syndrome}'): {e}")
        return out

    def _filter_results(self, records: list, terms: list) -> list:
        if not terms:
            return records
        
        import re # Đảm bảo đã import re ở đầu file
        
        # 1. Ưu tiên AND logic
        filtered_and = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            # Dùng regex \b để bắt buộc từ khóa phải đứng độc lập (vd: \bho\b không match choáng)
            if all(re.search(rf'\b{re.escape(term.lower())}\b', record_text) for term in terms):
                filtered_and.append(record)
        
        if filtered_and:
            return filtered_and
        
        # 2. Fallback sang OR
        logger.warning("Không tìm thấy AND, fallback sang OR")
        filtered_or = []
        for record in records:
            record_text = " ".join([str(val) for val in record.values()]).lower()
            if any(re.search(rf'\b{re.escape(term.lower())}\b', record_text) for term in terms):
                filtered_or.append(record)
        
        return filtered_or

    # ====================================================================
    # [TÍNH NĂNG] HỎI TỔNG QUAN THEO BỆNH LÝ (BenhLy)
    # Câu hỏi kiểu "các hội chứng / bài thuốc của bệnh X", "bệnh X là gì" chứa TÊN BỆNH
    # (không phải triệu chứng) nên bộ trích triệu chứng cũ trả rỗng -> báo lỗi.
    # Ở đây phát hiện tên BenhLy trong câu hỏi rồi truy vấn TẤT ĐỊNH (không dùng LLM sinh Cypher,
    # tham số hoá $disease -> an toàn, read-only) trả về: Hội chứng -> Triệu chứng + Bài thuốc -> Vị thuốc.
    # ====================================================================
    # Cụm từ cho thấy người dùng muốn hỏi TỔNG QUAN về bệnh (dùng để phân biệt với câu hỏi triệu chứng)
    _DISEASE_CUES = (
        "hội chứng", "bài thuốc", "phương thuốc", "vị thuốc", "bài thuốc",
        "tổng qu", "thông tin", "gồm những", "có những", "thể bệnh",
        "thuốc chữa", "thuốc trị", "chữa bệnh", "trị bệnh", "điều trị bệnh",
    )

    def _get_all_disease_names(self) -> list:
        """Lấy toàn bộ tên BenhLy (bỏ node bẩn có số/ngoặc) để dò trong câu hỏi."""
        try:
            with self.driver.session() as session:
                return [rec["name"] for rec in session.run(
                    "MATCH (b:BenhLy) WHERE b.name IS NOT NULL AND NOT b.name =~ '.*[0-9(].*' "
                    "RETURN b.name AS name")]
        except Exception as e:
            logger.error(f"Lỗi lấy danh sách bệnh lý: {e}")
            return []

    def _match_disease_in_question(self, text: str):
        """Trả về TÊN BỆNH (đúng như lưu trong DB) nếu câu hỏi hướng về 1 bệnh lý cụ thể, ngược lại None.

        Chống nhầm với câu hỏi triệu chứng: bệnh nhiều chữ (vd 'Ách nghịch') luôn được nhận;
        bệnh 1 chữ (vd 'Ho', 'Lỵ') chỉ nhận khi câu hỏi có cụm gợi ý hỏi-bệnh (_DISEASE_CUES)."""
        if not text:
            return None
        text_lower = text.lower()
        names = self._get_all_disease_names()
        if not names:
            return None
        names_sorted = sorted(set(n for n in names if n), key=len, reverse=True)  # dài trước
        matched = []
        temp = text_lower
        for name in names_sorted:
            nl = name.lower().strip()
            if len(nl) < 2:
                continue
            # Khớp như từ độc lập (\w của Python đã bao gồm chữ có dấu tiếng Việt)
            pattern = rf'(?<!\w){re.escape(nl)}(?!\w)'
            if re.search(pattern, temp):
                matched.append(name)
                temp = re.sub(pattern, " " * len(nl), temp, count=1)
        if not matched:
            return None
        multiword = [m for m in matched if len(m.split()) >= 2]
        if multiword:
            return max(multiword, key=len)          # ưu tiên tên bệnh cụ thể/dài nhất
        if any(cue in text_lower for cue in self._DISEASE_CUES):
            return max(matched, key=len)             # bệnh 1 chữ: cần có ngữ cảnh hỏi-bệnh
        return None

    @staticmethod
    def _norm_ws(s: str) -> str:
        """Chuẩn hoá khoảng trắng để HIỂN THỊ: bỏ tab/space thừa đầu-cuối và gộp khoảng trắng lặp
        (một số tên node trong graph còn dính tab/space bẩn từ lúc import, vd '\\tĐầu thống')."""
        return re.sub(r"\s+", " ", (s or "")).strip()

    # ====================================================================
    # [TÍNH NĂNG] HỎI "HỘI CHỨNG X GỒM NHỮNG BỆNH LÝ GÌ" (HoiChung -> danh sách BenhLy + triệu chứng)
    # ====================================================================
    def _get_all_syndrome_names(self) -> list:
        """Lấy toàn bộ tên HoiChung (bỏ node bẩn có số/ngoặc/gắn cờ) để dò trong câu hỏi."""
        try:
            with self.driver.session() as session:
                return [rec["name"] for rec in session.run(
                    "MATCH (h:HoiChung) WHERE h.name IS NOT NULL AND NOT h.name =~ '.*[0-9(].*' "
                    "AND NOT coalesce(h._flagged_dirty, false) RETURN h.name AS name")]
        except Exception as e:
            logger.error(f"Lỗi lấy danh sách hội chứng: {e}")
            return []

    @staticmethod
    def _is_syndrome_to_diseases_intent(text: str) -> bool:
        """True nếu câu hỏi kiểu 'hội chứng X gồm/có những bệnh (lý) nào/gì' (hội chứng -> liệt kê bệnh)."""
        t = (text or "").lower()
        has_syn_ref = ("hội chứng" in t) or ("thể bệnh" in t)
        wants_diseases = any(c in t for c in (
            "bệnh gì", "bệnh nào", "bệnh lý gì", "bệnh lý nào", "những bệnh", "các bệnh",
            "gồm bệnh", "có ở bệnh", "ở bệnh nào", "thuộc bệnh", "bệnh lý gồm", "bệnh lý liên quan",
        ))
        return has_syn_ref and wants_diseases

    def _match_syndrome_in_question(self, text: str):
        """Trả về TÊN HỘI CHỨNG (đúng như DB) xuất hiện trong câu hỏi (longest-match), ngược lại None."""
        if not text:
            return None
        names = self._get_all_syndrome_names()
        if not names:
            return None
        text_lower = text.lower()
        names_sorted = sorted(set(n for n in names if n), key=len, reverse=True)  # dài trước
        for name in names_sorted:
            nl = name.lower().strip()
            if len(nl) < 2:
                continue
            pattern = rf'(?<!\w){re.escape(nl)}(?!\w)'
            if re.search(pattern, text_lower):
                return name
        return None

    def get_diseases_by_syndrome_overview(self, syndrome_name: str) -> dict:
        """Truy vấn TẤT ĐỊNH: hội chứng -> các BỆNH LÝ mang hội chứng đó, kèm TRIỆU CHỨNG của từng bệnh
        (ràng buộc r.benh_ly=b.name để triệu chứng đúng ngữ cảnh của từng bệnh). Read-only, tham số hoá."""
        # [FIX CASE-INSENSITIVE] gom mọi biến thể hoa/thường của cùng hội chứng
        cypher = """
        MATCH (b:BenhLy)-[:CHIA_THÀNH]->(h:HoiChung)
        WHERE toLower(h.name) = toLower($syn)
        OPTIONAL MATCH (h)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung) WHERE r.benh_ly = b.name
        WITH b, collect(DISTINCT t.name) AS symptoms
        RETURN b.name AS disease, symptoms
        ORDER BY disease
        """
        rows = []
        try:
            with self.driver.session() as session:
                rows = session.execute_read(
                    lambda tx: [dict(rec) for rec in tx.run(cypher, syn=syndrome_name)])
        except Exception as e:
            logger.error(f"Lỗi get_diseases_by_syndrome_overview ('{syndrome_name}'): {e}")

        if not rows:
            answer = (f"### 🩺 Hội chứng: **{syndrome_name}**\n\n"
                      f"Hệ tri thức chưa ghi nhận bệnh lý nào mang hội chứng này.")
            return {"question": "", "answer": answer, "data": []}

        syn_disp = self._norm_ws(syndrome_name)
        parts = [f"### 🩺 Hội chứng: **{syn_disp}**",
                 f"Trong hệ tri thức, hội chứng **{syn_disp}** xuất hiện ở **{len(rows)} bệnh lý**:\n"]
        for i, r in enumerate(rows, 1):
            parts.append(f"**{i}. {self._norm_ws(r['disease'])}**")
            syms = [self._norm_ws(s) for s in (r.get("symptoms") or []) if s and s.strip()]
            if syms:
                parts.append(f"- **Triệu chứng:** {', '.join(syms)}")
            else:
                parts.append("- **Triệu chứng:** _(chưa có trong dữ liệu)_")
            parts.append("")  # dòng trống ngăn cách
        answer = "\n".join(parts)
        return {"question": "", "answer": answer, "data": rows}

    def get_disease_overview(self, disease_name: str) -> dict:
        """Truy vấn TẤT ĐỊNH tổng quan 1 bệnh: mỗi Hội chứng kèm Triệu chứng đặc trưng,
        Bài thuốc điều trị và Vị thuốc trong bài. Ràng buộc r.benh_ly=b.name & p.benh_ly=b.name
        & p.hoi_chung=h.name để giữ ĐÚNG ngữ cảnh của bệnh (không lẫn bài thuốc của bệnh khác)."""
        cypher = """
        MATCH (b:BenhLy {name:$disease})-[:CHIA_THÀNH]->(h:HoiChung)
        OPTIONAL MATCH (h)-[r:CÓ_BIỂU_HIỆN]->(t:TrieuChung) WHERE r.benh_ly = b.name
        WITH b, h, collect(DISTINCT t.name) AS symptoms
        OPTIONAL MATCH (h)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
          WHERE p.benh_ly = b.name AND p.hoi_chung = h.name
        OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
        WITH b, h, symptoms, p, collect(DISTINCT v.name) AS herbs
        WITH h, symptoms,
             collect(DISTINCT CASE WHEN p IS NULL THEN NULL ELSE {bai_thuoc: p.name, vi_thuoc: herbs} END) AS formulas0
        RETURN h.name AS syndrome, symptoms, [f IN formulas0 WHERE f IS NOT NULL] AS formulas
        ORDER BY syndrome
        """
        rows = []
        try:
            with self.driver.session() as session:
                rows = session.execute_read(
                    lambda tx: [dict(rec) for rec in tx.run(cypher, disease=disease_name)])
        except Exception as e:
            logger.error(f"Lỗi get_disease_overview ('{disease_name}'): {e}")

        # ------- Dựng câu trả lời Markdown -------
        disease_disp = self._norm_ws(disease_name)
        if not rows:
            answer = (f"### 🩺 Bệnh lý: **{disease_disp}**\n\n"
                      f"Hệ tri thức chưa có dữ liệu hội chứng/bài thuốc cho bệnh này.")
            return {"question": "", "answer": answer, "data": []}

        n_syn = len(rows)
        n_formula = sum(len(r.get("formulas") or []) for r in rows)
        parts = [f"### 🩺 Tổng quan bệnh lý: **{disease_disp}**",
                 f"Trong hệ tri thức, bệnh **{disease_disp}** được chia thành **{n_syn} hội chứng** "
                 f"(thể bệnh){f', với tổng cộng {n_formula} bài thuốc điều trị' if n_formula else ''}:\n"]
        for i, r in enumerate(rows, 1):
            parts.append(f"**{i}. {self._norm_ws(r['syndrome'])}**")
            syms = [self._norm_ws(s) for s in (r.get("symptoms") or []) if s and s.strip()]
            if syms:
                parts.append(f"- **Triệu chứng đặc trưng:** {', '.join(syms)}")
            forms = [f for f in (r.get("formulas") or []) if f and f.get("bai_thuoc")]
            if forms:
                parts.append("- **Bài thuốc điều trị:**")
                for f in forms:
                    herbs = [self._norm_ws(h) for h in (f.get("vi_thuoc") or []) if h and h.strip()]
                    if herbs:
                        parts.append(f"    - *{self._norm_ws(f['bai_thuoc'])}* — gồm: {', '.join(herbs)}")
                    else:
                        parts.append(f"    - *{self._norm_ws(f['bai_thuoc'])}*")
            else:
                parts.append("- **Bài thuốc:** _(chưa có trong dữ liệu)_")
            parts.append("")  # dòng trống ngăn cách hội chứng
        answer = "\n".join(parts)
        return {"question": "", "answer": answer, "data": rows}

    def execute_and_answer(self, user_question: str, read_only: bool = False) -> dict:
        # read_only=True: luồng web — chặn Cypher ghi + KHÔNG lộ Cypher/stacktrace ra client.
        # Bước 0-A: Câu hỏi kiểu "hội chứng X gồm những bệnh lý gì" -> liệt kê BỆNH LÝ + triệu chứng
        # từng bệnh. CHẠY TRƯỚC dò bệnh vì có tên (vd "Khí hư") vừa là BenhLy vừa là HoiChung.
        try:
            if self._is_syndrome_to_diseases_intent(user_question):
                syndrome = self._match_syndrome_in_question(user_question)
                if syndrome:
                    syn_overview = self.get_diseases_by_syndrome_overview(syndrome)
                    if syn_overview.get("data"):
                        syn_overview["question"] = user_question
                        return syn_overview
        except Exception as e:
            logger.error(f"Lỗi xử lý intent hội chứng->bệnh lý: {e}")

        # Bước 0: Nếu câu hỏi hướng về 1 BỆNH LÝ -> trả TỔNG QUAN bệnh (tất định, không cần LLM).
        try:
            disease = self._match_disease_in_question(user_question)
        except Exception as e:
            logger.error(f"Lỗi dò bệnh lý trong câu hỏi: {e}")
            disease = None
        if disease:
            overview = self.get_disease_overview(disease)
            if overview.get("data"):          # chỉ dùng khi bệnh thực sự có dữ liệu
                overview["question"] = user_question
                return overview
            # bệnh khớp nhưng rỗng -> rơi xuống luồng triệu chứng bên dưới

        # Bước 1: Tiền xử lý câu hỏi
        terms = self._preprocess_question(user_question)
        if not terms:
            return {
                "question": user_question,
                "answer": "Không thể trích xuất từ khóa từ câu hỏi của bạn.",
                "data": []
            }
        
        # Bước 2: Sinh Cypher
        cypher_query = self.text_to_cypher(user_question, terms=terms)
        if not cypher_query:
            return {
                "question": user_question,
                "cypher_used": "",
                "answer": "Không thể dịch câu hỏi sang câu truy vấn database.",
                "data": []
            }
        
        # Bước 3: Thực thi trong Neo4j
        try:
            try:
                records = self.run_cypher(cypher_query, read_only=read_only)
            except ValueError:
                # Guard read-only từ chối câu Cypher sinh ra — KHÔNG lộ nội dung Cypher ra ngoài
                logger.warning("Cypher sinh ra bị từ chối vì không phải read-only.")
                return {
                    "question": user_question,
                    "answer": "Xin lỗi, hệ thống chỉ hỗ trợ truy vấn tra cứu (chỉ đọc) nên không thể xử lý yêu cầu này.",
                    "data": []
                }
            if not records:
                return {
                    "question": user_question,
                    "cypher_used": cypher_query,
                    "answer": "Không tìm thấy kết quả phù hợp với truy vấn này.",
                    "data": []
                }
            
            # Bước 4: Lọc kết quả bằng logic ưu tiên
            # Nhóm các bệnh theo tên
            disease_map = {}
            for record in records:
                disease = record.get("b.name", "")
                if not disease:
                    continue
                if disease not in disease_map:
                    disease_map[disease] = []
                disease_map[disease].append(record)
            
            # Tính điểm ưu tiên cho mỗi bệnh dựa trên số lượng hội chứng và bài thuốc
            scored_diseases = []
            for disease, records_list in disease_map.items():
                # Đếm số lượng hội chứng và bài thuốc duy nhất
                syndromes = set()
                prescriptions = set()
                for record in records_list:
                    if record.get("h.name"):
                        syndromes.add(record.get("h.name"))
                    if record.get("p.name"):
                        prescriptions.add(record.get("p.name"))
                
                score = len(syndromes) + len(prescriptions)
                scored_diseases.append({
                    "disease": disease,
                    "score": score,
                    "records": records_list,
                    "syndromes": list(syndromes),
                    "prescriptions": list(prescriptions)
                })
            
            # Sắp xếp theo điểm giảm dần
            scored_diseases.sort(key=lambda x: x["score"], reverse=True)
            
            # Chỉ lấy top 10 kết quả tốt nhất
            top_diseases = scored_diseases[:10]
            
            # Bước 5: Đóng gói câu trả lời tự nhiên (NLG)
            cau_tra_loi_parts = []
            
            if top_diseases:
                danh_sach_benh = [d["disease"] for d in top_diseases]
                gioi_han = danh_sach_benh[:5]
                text = f"Bệnh lý liên quan (ưu tiên theo độ khớp): {', '.join(gioi_han)}"
                if len(danh_sach_benh) > 5:
                    text += f" (và {len(danh_sach_benh) - 5} bệnh khác)"
                cau_tra_loi_parts.append(text)
                
                # Lấy bài thuốc cho bệnh có điểm cao nhất
                best_disease = top_diseases[0]
                if best_disease.get("prescriptions"):
                    text = f"Bài thuốc ưu tiên cho bệnh '{best_disease['disease']}': {', '.join(best_disease['prescriptions'])}"
                    cau_tra_loi_parts.append(text)
            
            if cau_tra_loi_parts:
                cau_tra_loi = "Dựa trên thông tin của bạn. " + " | ".join(cau_tra_loi_parts) + "."
            else:
                cau_tra_loi = "Hệ thống chưa tìm thấy thông tin phù hợp với truy vấn này."
            
            return {
                "question": user_question,
                "cypher_used": cypher_query,
                "answer": cau_tra_loi,
                "data": records
            }
        except Exception as e:
            logger.error(f"Lỗi khi chạy Cypher trên Neo4j: {e}")
            if read_only:
                # Web: KHÔNG lộ chi tiết lỗi / câu Cypher ra client
                return {
                    "question": user_question,
                    "answer": "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau.",
                    "data": []
                }
            return {"error": str(e), "failed_cypher": cypher_query}

    def get_syndrome_metadata(self, syndrome: str) -> dict:
        """Lấy thông tin Tạng Phủ và Bát Cương liên quan đến Hội chứng"""
        # [FIX CASE-INSENSITIVE] khớp không phân biệt hoa/thường. Nhưng nếu tồn tại node khớp ĐÚNG
        # hoa/thường thì CHỈ lấy node đó (tránh trộn tag Bát Cương của các biến thể trùng tên có thể
        # gắn nhãn lệch nhau -> reintroduce 'Thực' lạ kích 'Bản Hư Tiêu Thực'). Chỉ khi không có
        # node khớp đúng casing mới gom mọi biến thể (đảm bảo vẫn tra được khi tên gọi lệch casing).
        cypher = """
        MATCH (h:HoiChung)
        WHERE toLower(h.name) = toLower($syn)
        WITH collect(h) AS hs
        WITH hs, [x IN hs WHERE x.name = $syn] AS exact
        WITH CASE WHEN size(exact) > 0 THEN exact ELSE hs END AS chosen
        UNWIND chosen AS h
        OPTIONAL MATCH (h)-[:THUỘC_TẠNG]->(tp:TangPhu)
        OPTIONAL MATCH (h)-[:CÓ_TÍNH_CHẤT|TRẠNG_THÁI|VỊ_TRÍ]->(bc:BatCuong)
        RETURN collect(DISTINCT tp.name) AS organs, collect(DISTINCT bc.name) AS bat_cuong
        """
        try:
            with self.driver.session() as session:
                rec = session.run(cypher, syn=syndrome).single()
                if rec:
                    return {
                        "organs": rec["organs"] or [],
                        "bat_cuong": rec["bat_cuong"] or []
                    }
        except Exception as e:
            logger.error(f"Lỗi get_syndrome_metadata ('{syndrome}'): {e}")
        return {"organs": [], "bat_cuong": []}

    def ask(self, question: str, read_only: bool = False) -> dict:
        """Hỏi và nhận câu trả lời. read_only=True dùng cho luồng web (chặn Cypher ghi,
        không lộ Cypher/lỗi). CLI giữ read_only=False như cũ."""
        return self.execute_and_answer(question, read_only=read_only)

    def _format_answer(self, data: list) -> str:
        """Chuyển kết quả thành văn bản tự nhiên"""
        if not data:
            return "Không tìm thấy kết quả."
        return json.dumps(data, ensure_ascii=False, indent=2)
