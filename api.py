# api.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import re
import logging

from src.fusion_pipeline import TCMFusionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HealthWatch AI Clinic")

# Cấu hình CORS để Web Frontend có thể gọi được API.
# LƯU Ý: allow_credentials BẮT BUỘC là False khi allow_origins=["*"] (theo chuẩn CORS).
# Khi triển khai thật, hãy thay ["*"] bằng danh sách domain cụ thể của bạn.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo bộ não AI
logger.info("Đang khởi động AI Backend...")
fusion_engine = TCMFusionPipeline()
try:
    os.makedirs("temp_uploads", exist_ok=True)  # Thư mục lưu ảnh tạm (chỉ dùng khi chạy local)
except OSError:
    pass  # Serverless (Vercel) filesystem read-only — ảnh upload đã lưu vào tempfile.gettempdir()
logger.info("Sẵn sàng!")

@app.post("/api/diagnose")
async def diagnose(
    symptoms: str = Form(""), 
    face_img: UploadFile = File(None), 
    tongue_img: UploadFile = File(None)
):
    # Làm sạch dữ liệu rác truyền từ frontend (nếu JS truyền biến undefined/null ở dạng chuỗi)
    if symptoms:
        s_val = symptoms.strip().lower()
        if s_val in ["undefined", "null", "none"]:
            symptoms = ""

    # Kiểm tra bắt buộc: Phải cung cấp ít nhất triệu chứng bằng văn bản HOẶC tải lên ít nhất một hình ảnh
    if not symptoms.strip() and not face_img and not tongue_img:
        raise HTTPException(
            status_code=400, 
            detail="Bắt buộc phải cung cấp triệu chứng lâm sàng bằng văn bản hoặc tải lên ít nhất một hình ảnh (ảnh sắc mặt hoặc ảnh lưỡi)."
        )

    import uuid
    import tempfile
    face_path, tongue_path = None, None
    try:
        # Lưu file tạm với tên độc nhất trong thư mục temp của hệ điều hành để tránh lỗi quyền ghi
        temp_dir = tempfile.gettempdir()
        if face_img:
            ext = os.path.splitext(face_img.filename)[1] or ".jpg"
            face_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
            with open(face_path, "wb") as buffer:
                shutil.copyfileobj(face_img.file, buffer)
        if tongue_img:
            ext = os.path.splitext(tongue_img.filename)[1] or ".jpg"
            tongue_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
            with open(tongue_path, "wb") as buffer:
                shutil.copyfileobj(tongue_img.file, buffer)

        # Gọi hệ thống hợp nhất chẩn đoán
        result = fusion_engine.run_diagnosis(
            user_symptoms=symptoms, 
            face_img_path=face_path, 
            tongue_img_path=tongue_path
        )
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception:
        # Ghi log đầy đủ ở server, nhưng KHÔNG trả chi tiết lỗi/stacktrace ra client
        logger.exception("Lỗi khi xử lý chẩn đoán")
        raise HTTPException(status_code=500, detail="Đã xảy ra lỗi nội bộ khi xử lý yêu cầu. Vui lòng thử lại sau.")
    finally:
        # Xóa các file tạm sau khi đã xử lý xong
        for path in [face_path, tongue_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Đã dọn dẹp file tạm: {path}")
                except Exception as e:
                    logger.warning(f"Không thể xóa file tạm {path}: {e}")

from pydantic import BaseModel

class SymptomsRequest(BaseModel):
    symptoms: str

class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_endpoint(req: AskRequest):
    """Hỏi-đáp tự do trên Knowledge Graph (LLM sinh Cypher ở chế độ READ-ONLY, có disclaimer y tế)."""
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 1000 ký tự).")
    try:
        result = fusion_engine.answer_question(question)
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception:
        # Ghi log đầy đủ ở server, KHÔNG trả stacktrace/Cypher ra client
        logger.exception("Lỗi khi xử lý câu hỏi hỏi-đáp")
        raise HTTPException(status_code=500, detail="Đã xảy ra lỗi nội bộ khi xử lý câu hỏi. Vui lòng thử lại sau.")

@app.post("/api/related-symptoms")
async def get_related_symptoms_endpoint(req: SymptomsRequest):
    symptoms_text = req.symptoms
    if not symptoms_text.strip():
        return {"status": "success", "data": []}
        
    try:
        # 1. Trích xuất từ khóa triệu chứng
        terms = fusion_engine.qa_pipeline._preprocess_question(symptoms_text)
        if not terms:
            return {"status": "success", "data": []}
            
        # 2. Tìm tên triệu chứng chính xác trong DB
        matched_db_names = []
        with fusion_engine.qa_pipeline.driver.session() as session:
            for term in terms:
                pattern = fusion_engine.qa_pipeline._word_boundary_pattern(term)
                q_match = """
                MATCH (t:TrieuChung)
                WHERE toLower(t.name) =~ $pattern
                RETURN t.name AS name
                """
                for rec in session.run(q_match, pattern=pattern):
                    matched_db_names.append(rec["name"].lower())
                    
        if not matched_db_names:
            return {"status": "success", "data": []}
            
        # 3. Tìm các triệu chứng liên quan đồng xuất hiện trong cùng Hội chứng (HoiChung)
        cypher = """
        MATCH (h:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
        WHERE toLower(t.name) IN $matched_names
        MATCH (h)-[:CÓ_BIỂU_HIỆN]->(t_other:TrieuChung)
        WHERE NOT toLower(t_other.name) IN $matched_names
        RETURN t_other.name AS symptom, count(distinct h) AS frequency
        ORDER BY frequency DESC
        LIMIT 100
        """

        # [LỌC GỢI Ý] Loại các mục KHÔNG phải triệu chứng người bệnh tự mô tả bằng lời:
        #   - Vọng chẩn LƯỠI & SẮC MẶT/DA (dấu hiệu quan sát từ ảnh)
        #   - Thiết chẩn MẠCH (người bệnh không tự bắt mạch)
        #   - TÊN BỆNH / chẩn đoán lọt vào nhãn TrieuChung (ung thư, u não, "bệnh ...", hội chứng)
        EXCLUDE_KEYWORDS = (
            # Vọng chẩn lưỡi / sắc mặt / da
            "lưỡi", "rêu", "sắc mặt", "sắc da", "gò má", "má đỏ", "ửng đỏ", "ửng hồng",
            "mặt đỏ", "mặt nhợt", "mặt nhạt", "mặt vàng", "mặt trắng", "mặt xanh",
            "mặt sạm", "mặt xạm", "mặt xám", "mặt tái",
            # Thiết chẩn mạch
            "mạch",
            # Tên bệnh / chẩn đoán (không phải triệu chứng)
            "ung thư", "u não", "khối u", "hội chứng", "bệnh ",
        )
        # Node BẨN (không phải triệu chứng sạch): chứa nháy/ngoặc/chữ số, hoặc là mảnh câu
        # bắt đầu bằng liên từ "hoặc" (bị cắt rời khi nhập liệu).
        _DIRTY_CHARS = re.compile(r'''["'()\[\]0-9]''')

        related_symptoms = []
        kept_keys = []
        seen = set()

        def _is_redundant(key: str) -> bool:
            # Bỏ cụm DÀI nếu đã có gợi ý NGẮN HƠN (>=2 từ) là bộ phận của nó
            # (vd 'đầy bụng ăn kém' ⊇ 'ăn kém'; 'phiền táo dễ tức giận' ⊇ 'phiền táo').
            # Chỉ xét cụm >=2 từ để KHÔNG gộp nhầm từ đơn (vd 'nôn' vs 'buồn nôn').
            return any(len(k.split()) >= 2 and k in key and k != key for k in kept_keys)

        with fusion_engine.qa_pipeline.driver.session() as session:
            for rec in session.run(cypher, matched_names=matched_db_names):
                name = (rec["symptom"] or "").strip()
                key = name.lower()
                if not key or key in seen:                       # bỏ trùng (không phân biệt hoa/thường)
                    continue
                seen.add(key)
                if _DIRTY_CHARS.search(name) or key.startswith("hoặc "):  # bỏ node bẩn / mảnh câu
                    continue
                if any(kw in key for kw in EXCLUDE_KEYWORDS):     # bỏ lưỡi/sắc mặt/mạch/tên bệnh
                    continue
                if _is_redundant(key):                            # bỏ cụm-con trùng nghĩa
                    continue
                related_symptoms.append(name)
                kept_keys.append(key)
                if len(related_symptoms) >= 30:
                    break

        return {"status": "success", "data": related_symptoms}
    except Exception:
        logger.exception("Lỗi gợi ý triệu chứng")
        raise HTTPException(status_code=500, detail="Đã xảy ra lỗi nội bộ khi gợi ý triệu chứng.")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
