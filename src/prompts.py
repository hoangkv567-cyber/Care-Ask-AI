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
- "luoi_beu"="có" CHỈ KHI thấy RÕ ÍT NHẤT HAI trong ba dấu sau trên chính THÂN LƯỠI:
  (1) thân lưỡi BÈ NGANG bất thường — bề ngang lớn so với chiều dài, lưỡi trông vuông/mập chứ không
      thon dài; (2) thân lưỡi DÀY VỒNG lên, mặt cắt tròn đầy, không phẳng-mỏng; (3) ĐẦU LƯỠI TÙ TRÒN,
      không thon nhọn.
  KHÔNG phải bệu -> "không": lưỡi thon dài, lưỡi dày trung bình, lưỡi thè gồng hết sức (bè TẠM THỜI),
  lưỡi nghiêng lệch hoặc bị cắt mất rìa trong khung hình.
  Chỉ thấy MỘT dấu, hoặc ảnh không cho đánh giá được hình thể thân lưỡi -> "không rõ".
  MẶC ĐỊNH "không rõ"; TUYỆT ĐỐI không chọn "có" khi chưa đủ hai dấu.
  Ba mốc trên đều NỘI TẠI thân lưỡi, KHÔNG cần thấy cung răng/khoang miệng — ảnh vọng chẩn là lưỡi
  THÈ nên mốc "rìa áp sát hàm răng" thường không có trong khung hình.
  ⚠ Đánh giá ĐỘC LẬP với "dau_rang": KHÔNG được suy từ vết hằn răng ra lưỡi bệu. Ngưỡng của
  "dau_rang" cố ý đặt RẤT THẤP ("dù nhẹ"); bắc cầu sẽ biến mọi vết lõm nhẹ thành áp lực báo bệu.
  (Gọi thừa "lưỡi bệu" kéo chẩn đoán sang Tỳ hư / thủy thấp OAN và LOẠI OAN thể âm hư — một token
  đủ để hạ bậc toàn bộ họ âm-hư khi lời khai không có dấu nhiệt.)
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
- Ánh sáng/nền ấm dễ làm da trông VÀNG: chỉ chọn "vàng úa" khi vàng thật rõ (thường kèm củng mạc mắt
  vàng); nếu chỉ do ánh sáng thì đánh giá theo sắc da thật, KHÔNG chọn "vàng úa".
- "sac_mat":
  * Chọn "trắng nhợt" khi sắc da TÁI NHẠT, NHỢT BỆCH, KHÔNG CÓ SẮC MÁU HOẶC KHUÔN MẶT GẦY HỐC HÁC TRŨNG SÂU (như khuôn mặt nữ da trắng/Châu Âu gầy hốc hác, da nhạt màu thiếu tưới nhuận).
  * Chọn "hồng hào bình thường" khi da mặt tươi nhuận, sắc da vàng hồng hoặc hồng nhạt khỏe mạnh tự nhiên, da đầy đặn tươi sáng có sức sống.
- "trang_diem"="có" CHỈ KHI thấy RÕ LỚP PHẤN NỀN (foundation) che phủ da, SON MÔI ĐẬM HOẶC KẺ MẮT/MASCARA TRANG ĐIỂM RÕ RÀNG. Da mặt mộc tự nhiên (kể cả có mụn trứng cá, vết thâm, tàn nhang, nốt ruồi, da khô, da nhờn, môi tự nhiên) -> MẶC ĐỊNH "trang_diem": "không". TUYỆT ĐỐI CẤM (PROHIBITED) đánh giá nhầm mặt mộc có mụn/vết thâm thành "có trang điểm".
- "go_ma_do"="có" CHỈ KHI thấy RÕ hai MẢNG ĐỎ KHU TRÚ ngay trên hai gò má (ửng đỏ từng vùng nổi bật
  trên nền da nhợt — dấu "lưỡng quyền hồng" của âm hư). Da đều màu / hồng tự nhiên nhẹ / má chỉ hơi ấm /
  đỏ ở QUANH MŨI-MIỆNG-CẰM (mụn, kích ứng, mao mạch) -> KHÔNG phải gò má đỏ -> "không". MẶC ĐỊNH "không"
  trừ khi mảng đỏ khu trú ở gò má thật rõ.
  PHÂN BIỆT THEO HÌNH THÁI, KHÔNG THEO VỊ TRÍ: "lưỡng quyền hồng" là MẢNG ĐỎ LIỀN, ĐỀU MÀU, không nổi
  gờ. Nếu vùng đỏ gồm các NỐT/SẨN/MỤN MỦ RỜI RẠC có ranh giới (mụn trứng cá) thì -> "không", KỂ CẢ khi
  chúng nằm ĐÚNG trên hai gò má. Đây là điểm bổ khuyết: danh sách loại trừ ở trên chỉ nêu quanh
  mũi-miệng-cằm nên mụn Ở MÁ lọt vào, bị gọi thành "gò má đỏ" và kéo chẩn đoán sang âm hư hỏa vượng OAN
  (đã xảy ra thật). Cổng "ban_do" ngay dưới vốn đã loại mụn ở má — hai cổng phải nhất quán.
- "ban_do"="có" CHỈ KHI có BAN/PHÁT BAN THẬT: vùng đỏ LAN TỎA thành mảng hoặc nốt ban dày (ban sởi,
  mề đay, phát ban dị ứng, đơn độc) phủ rộng trên da. MỤN TRỨNG CÁ / mụn cám (nốt sẩn đỏ rải rác ở
  má-cằm-trán-quanh mũi), tàn nhang, nốt ruồi, sẹo, mao mạch, kích ứng nhỏ -> KHÔNG phải ban -> "không".
  MẶC ĐỊNH "không" trừ khi có ban/phát ban LAN TỎA thật rõ. (Gọi nhầm mụn thành 'ban' sẽ kéo biện luận
  sang huyết nhiệt/tinh huyết hư OAN.)
- "quang_tham":
  * "quang_tham"="có": KHI DƯỚI MẮT CÓ MẢNG DA THÂM NÂU / SẠM TỐI / BÓNG THÂM RÕ RỆT (mi mắt dưới sẫm màu hơn hẳn vùng da gò má kế bên, như ở khuôn mặt nam Châu Á có quầng thâm).
  * "quang_tham"="không": KHI VÙNG DA DƯỚI MẮT PHẲNG MỊN, SÁNG ĐỀU MÀU CÙNG TÔNG DA MẶT (không có mảng thâm sẫm nào, như ở khuôn mặt nữ da trắng có mi dưới phẳng mịn sáng màu).

MẪU THỰC TẾ ĐỂ ĐỐI CHIẾU (FEW-SHOT ANCHORS):
1) Ảnh nữ da trắng gầy hốc hác, da nhạt bệch, mi mắt dưới phẳng sáng màu -> {"sac_mat": "trắng nhợt", "go_ma_do": "không", "phu": "không", "ban_do": "không", "quang_tham": "không", "trang_diem": "không"}
2) Ảnh nam Châu Á da vàng hồng tươi nhuận đầy đặn, mi mắt dưới sạm nâu thâm sẫm -> {"sac_mat": "hồng hào bình thường", "go_ma_do": "không", "phu": "không", "ban_do": "không", "quang_tham": "có", "trang_diem": "không"}
3) Ảnh trẻ em/thiếu niên da hồng tươi nhuận, mi mắt dưới nhẵn mịn -> {"sac_mat": "hồng hào bình thường", "go_ma_do": "không", "phu": "không", "ban_do": "không", "quang_tham": "không", "trang_diem": "không"}
- Chỉ trả JSON, không thêm chữ nào khác.
"""
