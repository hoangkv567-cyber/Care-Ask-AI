TONGUE_PROMPT_TEMPLATE = """
Act as a Traditional Chinese Medicine expert with 20 years of experience in tongue diagnosis (Vọng chẩn - xem lưỡi).
Analyze this tongue image carefully and write a concise, professional description of the patient's tongue features in English.

Please describe:
1. Tongue body color (e.g., pale, red, deep red, purple, normal pink...). Note: Pay close attention to lighting. If the tongue is pale red or normal pink but warm ambient lighting makes it look slightly red, describe it as normal pink or pale red. If it is clearly deep red, red, or dry, specify it.
2. Tongue coating color and texture (e.g., white or yellow coating, thin or thick coating, greasy/sticky, dry, peeled, or no coating...).
3. Tongue shape and features (Pay extreme attention to the sides/borders for scalloped tooth marks or wavy indentations, and the center/surface for any fissures/cracks. Describe them even if they are mild. Do not say "no tooth marks" unless the borders are completely smooth and round).

Write a concise description in English (1-2 sentences). Do not use JSON or lists. Just write the description directly.
"""

SYNDROME_PROMPT_TEMPLATE = """
Based on the following observed symptoms from a tongue image: {symptoms}
Suggest the most likely TCM syndrome (hội chứng) from this list: {syndrome_list}
Output ONLY the syndrome name as a JSON string. Example: "Tỳ vị hư nhược"
"""

FACE_PROMPT_TEMPLATE = """
Act as a TCM face diagnosis expert. Look at this face photo and describe ONLY what you can clearly see, focusing on:
1. Overall complexion color (e.g., pale/white, sallow/yellowish, flushed red, greenish, darkish, or normal/healthy pink). Note: Pay close attention to lighting. If the skin is naturally pale/white but appears slightly yellow due to warm indoor lighting or warm background colors, you must describe it as pale/white, NOT sallow/yellowish.
2. Skin luster: bright and moist, or dull and dry.
3. Redness concentrated in a specific area (e.g., flushed cheeks), rashes, or red patches — only if clearly visible.
4. Facial puffiness/swelling, dark circles, or puffiness under the eyes — only if clearly visible.

5. Clearly visible makeup (lipstick, blush, foundation, eyeliner) — mention it ONLY if clearly present, because makeup can mask the true complexion. If there is no makeup, do not mention makeup at all.

IMPORTANT RULES:
- Only describe a feature if you can clearly see it in the photo. If a feature is absent or you are unsure, do not mention it at all. Never guess or invent details.
- Do not comment on anything outside the list above (hair, eyebrows, moles, spots, wrinkles, jewelry, background).
- Do not use healthy/smooth/rosy template phrases unless they actually apply.

Write a concise, factual description in English (2-3 sentences).
"""

# ============================================================================
# Bản prompt TIẾNG VIỆT — dùng cho VLM cloud (Qwen3-VL qua SiliconFlow).
# Model mô tả thẳng bằng tiếng Việt nên pipeline bỏ qua được bước dịch Anh->Việt
# (nguồn gốc lỗi lặp từ và dịch sai thin/thick của Qwen).
# LLaVA local vẫn dùng bản tiếng Anh ở trên vì tiếng Việt của LLaVA rất yếu.
# ============================================================================

TONGUE_PROMPT_TEMPLATE_VI = """
Bạn là chuyên gia Đông y với 20 năm kinh nghiệm vọng chẩn (xem lưỡi).
Quan sát kỹ ảnh lưỡi và mô tả khách quan các đặc điểm sau:
1. Màu sắc thân lưỡi (ví dụ: nhợt, hồng nhạt, đỏ, đỏ sẫm, tím...). Chú ý ánh sáng: nếu lưỡi hồng nhạt/bình thường nhưng ánh sáng ấm làm trông hơi đỏ, hãy mô tả là hồng nhạt; chỉ nói đỏ/đỏ sẫm khi thấy rõ.
2. Màu sắc và kết cấu rêu lưỡi (rêu trắng hay vàng, mỏng hay dày, nhờn/dính, khô, bong tróc, hay không có rêu...).
3. Hình thể lưỡi: chú ý kỹ hai mép lưỡi. Vết lõm gợn sóng ở mép lưỡi CHÍNH LÀ dấu răng — nếu thấy dù chỉ nhẹ, hãy kết luận rõ "có dấu răng nhẹ"; chỉ kết luận "không có dấu răng" khi mép lưỡi hoàn toàn trơn nhẵn. TUYỆT ĐỐI không vừa mô tả vết lõm gợn sóng vừa nói không có dấu răng (mâu thuẫn). Ngoài ra xem bề mặt lưỡi có vết nứt không.

Viết mô tả ngắn gọn, khách quan 1-2 câu bằng tiếng Việt. Không dùng JSON hay gạch đầu dòng, chỉ viết đoạn văn mô tả.
"""

FACE_PROMPT_TEMPLATE_VI = """
Bạn là chuyên gia vọng chẩn Đông y. Quan sát ảnh khuôn mặt và CHỈ mô tả những gì thấy rõ, tập trung vào:
1. Màu sắc tổng thể của sắc mặt (ví dụ: trắng nhợt, vàng úa, đỏ bừng, xanh xao, sạm tối, hoặc hồng hào bình thường). Chú ý ánh sáng: nếu da vốn trắng nhợt nhưng hơi ngả vàng do ánh đèn ấm trong nhà hoặc nền ảnh màu ấm, phải mô tả là trắng nhợt, KHÔNG được nói vàng úa.
2. Độ tươi nhuận của da: sáng và ẩm mượt, hay khô xỉn thiếu sức sống.
3. Vùng đỏ tập trung (ví dụ: hai gò má đỏ), ban hoặc mảng đỏ — chỉ nêu khi thấy rõ.
4. Mặt phù/sưng, quầng thâm hay bọng mắt — chỉ nêu khi thấy rõ.
5. Trang điểm rõ (son môi, phấn má, kẻ mắt...) — CHỈ nêu khi thấy rõ là có, vì trang điểm che mất sắc mặt thật. Nếu không có trang điểm thì tuyệt đối không nhắc gì đến trang điểm.

QUY TẮC BẮT BUỘC:
- Chỉ mô tả đặc điểm thấy rõ trong ảnh. Nếu không có hoặc không chắc chắn, tuyệt đối KHÔNG nhắc đến. Không đoán, không bịa.
- Không bình luận về bất cứ thứ gì ngoài danh sách trên (tóc, lông mày, nốt ruồi, đốm, nếp nhăn, trang sức, hậu cảnh).
- Không dùng câu khuôn mẫu kiểu "khỏe mạnh/mịn màng/hồng hào" nếu không đúng thực tế.

Viết mô tả ngắn gọn, khách quan 2-3 câu bằng tiếng Việt.
"""

# ============================================================================
# Prompt VLM CÓ CẤU TRÚC (JSON) — mỗi trường có tập giá trị cố định + 'không rõ'.
# Thay văn xuôi bằng JSON để map triệu chứng DETERMINISTIC (src/vision_schema.py),
# bỏ được bước LLM đọc prose + nhiều tầng regex vá ảo giác. Bắt buộc dùng 'không rõ'
# khi không chắc để KHÔNG bịa đặc điểm.
# ============================================================================

TONGUE_JSON_PROMPT_VI = """
Bạn là chuyên gia Đông y vọng chẩn (xem lưỡi). Quan sát ảnh lưỡi và trả về DUY NHẤT một object JSON
(không kèm giải thích, không rào ```), theo đúng các khóa và giá trị cho phép sau:
{
  "than_luoi": "nhợt" | "hồng nhạt" | "đỏ" | "đỏ sẫm" | "tím" | "không rõ",
  "reu_mau":   "trắng" | "vàng" | "xám đen" | "không rêu" | "không rõ",
  "reu_day":   "mỏng" | "dày" | "không rõ",
  "reu_chat":  "nhuận" | "nhớt" | "khô" | "bong tróc" | "không rõ",
  "dau_rang":  "có" | "không" | "không rõ",
  "vet_nut":   "có" | "không" | "không rõ",
  "luoi_beu":  "có" | "không" | "không rõ"
}
QUY TẮC:
- CHỈ dùng đúng các giá trị liệt kê. Không chắc chắn -> "không rõ" (KHÔNG đoán, KHÔNG bịa).
- Chú ý ánh sáng ấm làm lưỡi trông đỏ hơn thực: nếu hồng nhạt/bình thường thì để "hồng nhạt".
- Vết lõm gợn sóng ở mép lưỡi = "dau_rang": "có" (dù nhẹ). Mép trơn nhẵn hoàn toàn = "không".
- "reu_mau"="không rêu" CHỈ KHI mặt lưỡi NHẴN BÓNG như gương (kính diện thiệt): đỏ/đỏ sẫm, thấy rõ
  gai nhú, TUYỆT ĐỐI không có lớp màng phủ. Lưỡi BÌNH THƯỜNG luôn có lớp rêu trắng mỏng — nếu thấy
  BẤT KỲ lớp màng/rêu nào (dù mỏng, dù chỉ ở giữa hoặc gốc lưỡi) -> "reu_mau"="trắng" + "reu_day"="mỏng".
  MẶC ĐỊNH nghiêng "trắng", KHÔNG chọn "không rêu" trừ khi chắc chắn nhẵn bóng không màng.
- NHẤT QUÁN: nếu "reu_mau"="không rêu" thì "reu_day" và "reu_chat" PHẢI ="không rõ" (không có rêu thì
  không có độ dày/chất rêu). Nếu mô tả được chất rêu (nhuận/nhớt/khô) tức là CÓ rêu -> KHÔNG được ="không rêu".
- "vet_nut"="có" CHỈ KHI thấy RÃNH NỨT SÂU rõ ràng (đường nứt dọc/ngang hằn sâu trên thân lưỡi). Bề mặt
  lấm tấm gai/nhú, nếp gợn nông, hạt li ti, hay ánh phản chiếu ướt -> KHÔNG phải vết nứt -> "không".
- Chỉ trả JSON, không thêm chữ nào khác.
"""

FACE_JSON_PROMPT_VI = """
Bạn là chuyên gia Đông y vọng chẩn (xem sắc mặt). Quan sát ảnh khuôn mặt và trả về DUY NHẤT một
object JSON (không kèm giải thích, không rào ```), theo đúng các khóa và giá trị cho phép sau:
{
  "sac_mat":    "trắng nhợt" | "vàng úa" | "đỏ bừng" | "xanh xao" | "sạm tối" | "hồng hào bình thường" | "không rõ",
  "go_ma_do":   "có" | "không" | "không rõ",
  "phu":        "có" | "không" | "không rõ",
  "ban_do":     "có" | "không" | "không rõ",
  "quang_tham": "có" | "không" | "không rõ",
  "trang_diem": "có" | "không" | "không rõ"
}
QUY TẮC:
- CHỈ dùng đúng các giá trị liệt kê. Không chắc chắn -> "không rõ" (KHÔNG đoán, KHÔNG bịa).
- Chú ý ánh sáng/nền ấm làm da trông vàng: nếu da vốn trắng nhợt thì để "trắng nhợt", KHÔNG "vàng úa".
- Có son/phấn/kẻ mắt rõ -> "trang_diem": "có" (vì trang điểm che sắc mặt thật).
- "go_ma_do"="có" CHỈ KHI thấy RÕ hai MẢNG ĐỎ KHU TRÚ ngay trên hai gò má (ửng đỏ từng vùng nổi bật
  trên nền da nhợt — dấu "lưỡng quyền hồng" của âm hư). Da đều màu / hồng tự nhiên nhẹ / má chỉ hơi ấm /
  đỏ ở QUANH MŨI-MIỆNG-CẰM (mụn, kích ứng, mao mạch) -> KHÔNG phải gò má đỏ -> "không". MẶC ĐỊNH "không"
  trừ khi mảng đỏ khu trú ở gò má thật rõ.
- "quang_tham"="có" chỉ khi quầng dưới mắt sẫm màu rõ; bóng đổ do ánh sáng -> "không".
- Chỉ trả JSON, không thêm chữ nào khác.
"""
