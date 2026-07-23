# api.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import re
import logging

from src.fusion_pipeline import TCMFusionPipeline
from src.interview import compose_interview_text, merge_symptoms
from src import auth as _auth
from src.auth import (
    ROLE_DOCTOR, ROLE_USER, get_current_user, require_doctor, get_store,
    create_token, validate_credentials, verify_password,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TCM AI Clinic")

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
# [XÁC THỰC] Tái dùng driver Neo4j của pipeline làm kho tài khoản (node :User) — không thêm hạ tầng
# và chạy được trên Vercel serverless (filesystem chỉ đọc nên không dùng được SQLite/file).
try:
    _auth.init_auth(fusion_engine.qa_pipeline.driver)
    logger.info("Tầng xác thực đã sẵn sàng (kho tài khoản: Neo4j :User).")
except Exception:
    logger.exception("Không khởi tạo được tầng xác thực — các API sẽ trả 503 khi đăng nhập.")
try:
    os.makedirs("temp_uploads", exist_ok=True)  # Thư mục lưu ảnh tạm (chỉ dùng khi chạy local)
except OSError:
    pass  # Serverless (Vercel) filesystem read-only — ảnh upload đã lưu vào tempfile.gettempdir()
logger.info("Sẵn sàng!")

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(default="", max_length=64)
    password: str = Field(default="", max_length=256)


class RegisterRequest(BaseModel):
    username: str = Field(default="", max_length=64)
    password: str = Field(default="", max_length=256)
    full_name: str = Field(default="", max_length=120)


@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """Tự đăng ký — LUÔN tạo vai 'user'. Tài khoản bác sĩ phải do quản trị tạo bằng
    scripts/seed_users.py; nếu cho tự chọn vai thì ai cũng tự nhận là bác sĩ để mở khoá
    tính năng chuyên môn."""
    uname = (req.username or "").strip().lower()
    err = validate_credentials(uname, req.password or "")
    if err:
        raise HTTPException(status_code=400, detail=err)
    try:
        u = get_store().create(uname, req.password, role=ROLE_USER,
                               full_name=(req.full_name or "").strip())
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Lỗi khi tạo tài khoản")
        raise HTTPException(status_code=500, detail="Không tạo được tài khoản. Vui lòng thử lại sau.")
    token = create_token(u["username"], u["role"], u["full_name"])
    return {"status": "success", "data": {"token": token, "user": u}}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    uname = (req.username or "").strip().lower()
    rec = None
    try:
        rec = get_store().get(uname)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Lỗi khi tra cứu tài khoản")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi đăng nhập.")
    # verify_password vẫn chạy bcrypt kể cả khi rec=None -> không lộ tài khoản nào tồn tại
    if not verify_password(req.password or "", rec.get("password_hash") if rec else None):
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không đúng.")
    if rec.get("active") is False:
        raise HTTPException(status_code=403, detail="Tài khoản đã bị khoá.")
    u = {"username": rec["username"], "role": rec.get("role") or ROLE_USER,
         "full_name": rec.get("full_name") or rec["username"]}
    return {"status": "success", "data": {"token": create_token(**{
        "username": u["username"], "role": u["role"], "full_name": u["full_name"]}), "user": u}}
@app.post("/api/auth/forgot-password")
async def forgot_password(req: LoginRequest):
    uname = (req.username or "").strip().lower()
    if not uname:
        raise HTTPException(status_code=400, detail="Vui lòng nhập tên đăng nhập.")
    try:
        rec = get_store().get(uname)
    except HTTPException:
        rec = None
    except Exception:
        logger.exception("Lỗi khi tra cứu tài khoản")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi tìm tài khoản.")
    if rec:
        logger.info(f"\n=================================================="
                    f"\n[KHÔI PHỤC MẬT KHẨU] Nhận yêu cầu cho tài khoản: {uname}"
                    f"\nLink khôi phục giả lập: http://localhost:8000/reset-password?username={uname}"
                    f"\n==================================================")
    return {
        "status": "success",
        "detail": "Yêu cầu khôi phục mật khẩu đã được ghi nhận. Vì hệ thống demo chưa kết nối cổng gửi email (SMTP), vui lòng kiểm tra logs của backend hoặc liên hệ Admin."
    }


class FirebaseLoginRequest(BaseModel):
    id_token: str = Field(default="")
    email: str = Field(default="", max_length=120)
    full_name: str = Field(default="", max_length=120)


@app.post("/api/auth/firebase-login")
async def firebase_login(req: FirebaseLoginRequest):
    email = (req.email or "").strip().lower()
    full_name = (req.full_name or "").strip()
    
    if req.id_token:
        try:
            payload = _auth.verify_firebase_token(req.id_token)
            email = payload.get("email", email).strip().lower()
            full_name = payload.get("name", full_name).strip()
        except Exception as e:
            logger.exception("Lỗi xác minh token Firebase")
            raise HTTPException(status_code=401, detail=f"Xác minh Firebase không thành công: {e}")
            
    if not email:
        raise HTTPException(status_code=400, detail="Thiếu địa chỉ email đăng nhập.")
        
    # Phân quyền: hoangkv567@gmail.com -> Bác sĩ, tài khoản khác -> Người dùng
    role = ROLE_DOCTOR if email == "hoangkv567@gmail.com" else ROLE_USER
    
    try:
        u = get_store().create_firebase_user(email, role, full_name)
    except Exception as e:
        logger.exception("Lỗi khi đồng bộ tài khoản Firebase vào Neo4j")
        raise HTTPException(status_code=500, detail="Lỗi đồng bộ tài khoản người dùng.")
        
    token = create_token(u["username"], u["role"], u["full_name"])
    return {
        "status": "success",
        "data": {
            "token": token,
            "user": u
        }
    }


@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return {"status": "success", "data": user}


def _strip_clinical_internals(result: dict) -> dict:
    """[PHÂN QUYỀN] Người dùng thường chỉ nhận PHẦN KẾT LUẬN đọc được. Các trường kỹ thuật
    (ứng viên bệnh + ratio, phổ hội chứng, dữ liệu KG thô, mô tả vọng chẩn nội bộ) chỉ dành cho
    bác sĩ: đọc sai chúng dễ dẫn tới tự chẩn đoán, và chúng phơi bày nội bộ hệ thống.
    Cắt ở BACKEND — nếu chỉ ẩn trên giao diện thì mở DevTools là thấy hết.

    NGOẠI LỆ CÓ CHỦ ĐÍCH — hai câu mô tả vọng chẩn ĐƯỢC giữ cho người dùng thường:
    'tongue_description_vi'/'face_description_vi' là tiếng Việt thường mô tả chính TẤM ẢNH NGƯỜI
    ĐÓ VỪA TẢI LÊN, không phải nội bộ hệ thống. Giữ lại vì hai lẽ:
      1) Cắt hết thì giao diện rơi vào nhánh else và in "AI chưa phát hiện được triệu chứng bất
         thường qua ảnh" — một câu SAI, vì AI có thấy, chỉ là bị cắt theo quyền. Trấn an sai chiều.
      2) Đó là đường DUY NHẤT để bệnh nhân phát hiện mô hình đọc nhầm ảnh của mình (đã có tiền lệ
         chế độ JSON vọng chẩn bịa dấu âm-hư). Mất nó thì lỗi vọng chẩn chỉ bác sĩ mới bắt được.
    Vẫn cắt 'analysis'/'detected_symptoms' (chuỗi triệu chứng đã chuẩn hóa — nội bộ),
    'structured' (JSON thô của VLM), 'input_fusion' và 'has_makeup'."""
    if not isinstance(result, dict):
        return result
    safe = {k: v for k, v in result.items()
            if k not in ("vision_details", "input_fusion", "has_makeup")}
    vd = result.get("vision_details")
    if isinstance(vd, dict):
        # Danh sách CHO PHÉP, không phải danh sách chặn: khóa mới thêm vào vision_details sau này
        # mặc định KHÔNG lọt ra người dùng thường.
        keep = {k: vd[k] for k in ("tongue_description_vi", "face_description_vi")
                if vd.get(k)}
        if keep:
            safe["vision_details"] = keep
    dr = safe.get("diagnosis_result")
    if isinstance(dr, dict):
        safe["diagnosis_result"] = {"answer": dr.get("answer", "")}
    return safe


@app.post("/api/diagnose")
async def diagnose(
    user: dict = Depends(get_current_user),
    symptoms: str = Form(""),
    face_img: UploadFile = File(None),
    tongue_img: UploadFile = File(None),
    # Vấn chẩn có cấu trúc (thập vấn + nhân khẩu) — tất cả tùy chọn, canonicalize ở backend
    age: str = Form(""),
    sex: str = Form(""),
    onset: str = Form(""),
    han_nhiet: str = Form(""),
    mo_hoi: str = Form(""),
    dai_tien: str = Form(""),
    tieu_tien: str = Form(""),
    khat: str = Form(""),
    an_uong: str = Form(""),
    ngu: str = Form(""),
):
    # Làm sạch dữ liệu rác truyền từ frontend (nếu JS truyền biến undefined/null ở dạng chuỗi)
    if symptoms:
        s_val = symptoms.strip().lower()
        if s_val in ["undefined", "null", "none"]:
            symptoms = ""

    # Ghép phần vấn chẩn có cấu trúc vào lời khai tự do (quy về từ khóa chuẩn của hệ chẩn đoán)
    interview_text = compose_interview_text(
        age=age, sex=sex, onset=onset,
        answers={
            "han_nhiet": han_nhiet, "mo_hoi": mo_hoi, "dai_tien": dai_tien,
            "tieu_tien": tieu_tien, "khat": khat, "an_uong": an_uong, "ngu": ngu,
        },
    )
    symptoms = merge_symptoms(symptoms, interview_text)

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
            tongue_img_path=tongue_path,
            sex=sex,   # [CỔNG GIỚI TÍNH] khai báo giới -> loại bệnh khác giới ở tầng khớp
        )
        
        # Lưu vào lịch sử chẩn đoán Neo4j
        import json
        history_id = str(uuid.uuid4())
        try:
            with get_store().driver.session() as session:
                session.run(
                    """
                    MATCH (u:User {username: $username})
                    CREATE (d:DiagnosisHistory {
                        id: $id,
                        timestamp: datetime(),
                        symptoms: $symptoms,
                        age: $age,
                        sex: $sex,
                        onset: $onset,
                        han_nhiet: $han_nhiet,
                        mo_hoi: $mo_hoi,
                        dai_tien: $dai_tien,
                        tieu_tien: $tieu_tien,
                        khat: $khat,
                        an_uong: $an_uong,
                        ngu: $ngu,
                        result_json: $result_json
                    })
                    CREATE (u)-[:HAS_DIAGNOSIS]->(d)
                    """,
                    username=user["username"],
                    id=history_id,
                    symptoms=symptoms,
                    age=age,
                    sex=sex,
                    onset=onset,
                    han_nhiet=han_nhiet,
                    mo_hoi=mo_hoi,
                    dai_tien=dai_tien,
                    tieu_tien=tieu_tien,
                    khat=khat,
                    an_uong=an_uong,
                    ngu=ngu,
                    result_json=json.dumps(result)
                )
        except Exception as he:
            logger.warning(f"Không thể lưu lịch sử chẩn đoán vào Neo4j: {he}")

        is_doctor = user.get("role") == ROLE_DOCTOR
        payload = result if is_doctor else _strip_clinical_internals(result)
        return {"status": "success", "data": payload, "role": user.get("role")}
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


@app.get("/api/history")
async def get_history(user: dict = Depends(get_current_user)):
    is_doctor = user.get("role") == ROLE_DOCTOR
    import json
    
    with get_store().driver.session() as session:
        if is_doctor:
            query = """
            MATCH (u:User)-[:HAS_DIAGNOSIS]->(d:DiagnosisHistory)
            RETURN d.id AS id, 
                   toString(d.timestamp) AS timestamp, 
                   d.symptoms AS symptoms, 
                   d.age AS age, 
                   d.sex AS sex, 
                   d.onset AS onset, 
                   d.result_json AS result_json,
                   u.username AS username,
                   u.full_name AS full_name
            ORDER BY d.timestamp DESC
            """
            records = session.run(query)
        else:
            query = """
            MATCH (u:User {username: $username})-[:HAS_DIAGNOSIS]->(d:DiagnosisHistory)
            RETURN d.id AS id, 
                   toString(d.timestamp) AS timestamp, 
                   d.symptoms AS symptoms, 
                   d.age AS age, 
                   d.sex AS sex, 
                   d.onset AS onset, 
                   d.result_json AS result_json,
                   u.username AS username,
                   u.full_name AS full_name
            ORDER BY d.timestamp DESC
            """
            records = session.run(query, username=user["username"])
        
        history_list = []
        for r in records:
            rec_dict = dict(r)
            try:
                raw_result = json.loads(rec_dict["result_json"])
                rec_dict["result"] = raw_result if is_doctor else _strip_clinical_internals(raw_result)
            except Exception:
                rec_dict["result"] = None
            
            if "result_json" in rec_dict:
                del rec_dict["result_json"]
            history_list.append(rec_dict)
            
    return {"status": "success", "data": history_list}


@app.delete("/api/history/{history_id}")
async def delete_history(history_id: str, user: dict = Depends(get_current_user)):
    is_doctor = user.get("role") == ROLE_DOCTOR
    
    with get_store().driver.session() as session:
        if is_doctor:
            query = """
            MATCH (d:DiagnosisHistory {id: $id})
            DETACH DELETE d
            RETURN count(d) AS deleted_count
            """
            result = session.run(query, id=history_id).single()
        else:
            query = """
            MATCH (u:User {username: $username})-[:HAS_DIAGNOSIS]->(d:DiagnosisHistory {id: $id})
            DETACH DELETE d
            RETURN count(d) AS deleted_count
            """
            result = session.run(query, username=user["username"], id=history_id).single()
            
        deleted = result["deleted_count"] > 0 if result else False
        if not deleted:
            raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi chẩn đoán hoặc bạn không có quyền xóa.")
            
    return {"status": "success", "message": "Đã xóa bản ghi chẩn đoán thành công."}

class SymptomsRequest(BaseModel):
    symptoms: str

class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_endpoint(req: AskRequest, user: dict = Depends(require_doctor)):
    """Hỏi-đáp tự do trên Knowledge Graph (LLM sinh Cypher ở chế độ READ-ONLY, có disclaimer y tế).

    [PHÂN QUYỀN] CHỈ BÁC SĨ. Đây là công cụ tra cứu chuyên môn sinh truy vấn thẳng trên KG —
    người bệnh dùng dễ hiểu sai thành lời khuyên điều trị cho bản thân."""
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
async def get_related_symptoms_endpoint(req: SymptomsRequest,
                                        user: dict = Depends(get_current_user)):
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
            
        # 3. Tìm triệu chứng liên quan ĐỒNG XUẤT HIỆN trong CÙNG (hội chứng × BỆNH) — scope theo
        #    benh_ly trên cạnh CÓ_BIỂU_HIỆN. Node HoiChung CHUNG ('Thận âm hư'/'Thận dương hư') ôm
        #    hợp triệu chứng của MỌI bệnh dùng nhãn đó, nên co-occur không-scope RÒ triệu chứng của
        #    BỆNH KHÁC: 'tiểu nhiều' (Đái tháo nhạt) từng kéo 'mù màu / sắc manh / dị thường sắc giác'
        #    (bệnh Sắc manh 色盲, cũng gán nhãn Thận âm/dương hư) làm gợi ý. Scope theo benh_ly (phủ
        #    99.2% cạnh) diệt tận gốc rò chéo-bệnh + xếp hạng theo SỐ BỆNH cùng biểu hiện.
        cypher_scoped = """
        MATCH (h:HoiChung)-[r1:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
        WHERE toLower(t.name) IN $matched_names AND r1.benh_ly IS NOT NULL
        MATCH (h)-[r2:CÓ_BIỂU_HIỆN]->(t_other:TrieuChung)
        WHERE NOT toLower(t_other.name) IN $matched_names AND r2.benh_ly = r1.benh_ly
        RETURN t_other.name AS symptom, count(DISTINCT r1.benh_ly) AS frequency
        ORDER BY frequency DESC
        LIMIT 100
        """
        # Fallback KHÔNG scope (dùng khi cạnh của triệu chứng khớp thiếu benh_ly -> scoped rỗng).
        cypher_unscoped = """
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
            # Vọng chẩn lưỡi / sắc mặt / da ('vết răng'/'hằn răng' = rìa lưỡi in răng, cũng là dấu
            # quan sát lưỡi nhưng không chứa chữ 'lưỡi' nên phải liệt riêng)
            "lưỡi", "rêu", "vết răng", "hằn răng", "sắc mặt", "sắc da", "gò má", "má đỏ", "ửng đỏ", "ửng hồng",
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
            rows = list(session.run(cypher_scoped, matched_names=matched_db_names))
            if not rows:                                      # cạnh thiếu benh_ly -> bay mù, không-scope
                rows = list(session.run(cypher_unscoped, matched_names=matched_db_names))
            for rec in rows:
                name = (rec["symptom"] or "").strip()
                key = name.lower()
                if not key or key in seen:                       # bỏ trùng (không phân biệt hoa/thường)
                    continue
                seen.add(key)
                if _DIRTY_CHARS.search(name) or key.startswith("hoặc "):  # bỏ node bẩn / mảnh câu
                    continue
                if any(kw in key for kw in EXCLUDE_KEYWORDS):     # bỏ lưỡi/sắc mặt/mạch/tên bệnh
                    continue
                if fusion_engine._is_treatment_principle(name):   # bỏ PHÁP TRỊ (dưỡng âm/bổ thận...) lọt nhãn triệu chứng
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
