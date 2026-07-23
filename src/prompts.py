# Prompt templates cho mô hình Vision (VLM) và Text LLM

# Prompt VLM Tiếng Việt - Lưỡi (Mô tả tự do / Prose)
TONGUE_PROMPT_TEMPLATE_VI = """
Hãy quan sát kỹ hình ảnh lưỡi được cung cấp và mô tả ngắn gọn các đặc điểm Đông y (chỉ nêu những gì thấy RÕ NÉT):
1. Thân lưỡi (sắc chất lưỡi): hồng nhạt, đỏ, nhợt, tím, hay có điểm ứ huyết/vết tím? Thân lưỡi thon gọn hay bệu to? Rìa lưỡi có dấu răng không? Có vết nứt không?
2. Rêu lưỡi: màu sắc (trắng, vàng, xám/đen), độ dày (mỏng hay dày), chất rêu (nhuận, khô, dính/nhờn, hay bong tróc)?
3. Thái độ vọng chẩn: Chỉ mô tả những đặc điểm nhìn thấy RÕ BẰNG MẮT THƯỜNG. Nếu nhòe hoặc không rõ, nêu "không rõ". KHÔNG đoán mò, KHÔNG suy diễn hội chứng. Trả lời ngắn gọn 2-4 câu bằng tiếng Việt.
"""

# Prompt VLM Tiếng Việt - Sắc mặt (Mô tả tự do / Prose)
FACE_PROMPT_TEMPLATE_VI = """
Hãy quan sát kỹ hình ảnh khuôn mặt được cung cấp và mô tả ngắn gọn các đặc điểm Đông y:
1. Sắc mặt (thần sắc & màu da): hồng hào tươi nhuận, trắng nhợt, vàng úa/nhiễm vàng, xanh xao, sạm tối, hay đỏ bừng?
2. Biểu hiện vùng mặt: hai gò má có mảng đỏ nổi bật không? Có phù nề ở mặt/mi mắt không? Có phát ban dị ứng mảng lớn phủ rộng trên da mặt không? Có quầng thâm sẫm dưới mắt không?
3. QUY TẮC NGHIÊM NGẶT:
   - "Ban đỏ" CHỈ KHI có BAN/PHÁT BAN THẬT (mảng đỏ lan tỏa, ban dị ứng/sởi). Mụn trứng cá/mụn cám rải rác -> KHÔNG PHẢI BAN.
   - "Gò má đỏ" CHỈ KHI có mảng đỏ khu trú trên hai gò má (lưỡng quyền hồng).
   - "Vàng úa": khi da mặt hoặc củng mạc mắt nhuốm màu vàng rõ rệt (vàng da sơ sinh, hoàng đản).
   - "Trang điểm": chỉ ghi có khi thấy rõ lớp phấn nền dày, son môi đậm. Mặt mộc có mụn -> KHÔNG TRANG ĐIỂM.
4. Trả lời ngắn gọn 2-4 câu bằng tiếng Việt.
"""

# Prompt VLM Tiếng Việt - Lưỡi (JSON CÓ CẤU TRÚC)
TONGUE_JSON_PROMPT_VI = """
Bạn là chuyên gia Đông y vọng chẩn (xem lưỡi). Quan sát ảnh lưỡi và trả về DUY NHẤT một object JSON (không kèm giải thích, không rào ```), theo đúng các khóa và giá trị cho phép sau:
{
  "than_luoi": "hồng nhạt" | "đỏ" | "đỏ sẫm" | "nhợt" | "tím" | "điểm ứ huyết" | "không rõ",
  "reu_mau":   "trắng" | "vàng" | "xám đen" | "ít/không rêu" | "không rõ",
  "reu_day":   "mỏng" | "dày" | "không rõ",
  "reu_chat":  "nhuận" | "khô" | "nhờn dính" | "bong tróc" | "không rõ",
  "dau_rang":  "có" | "không" | "không rõ",
  "vet_nut":   "có" | "không" | "không rõ",
  "luoi_beu":  "có" | "không" | "không rõ"
}
QUY TẮC:
- CHỈ dùng đúng các giá trị liệt kê ở trên. Không chắc chắn -> "không rõ" (KHÔNG đoán, KHÔNG bịa).
- "dau_rang"="có" KHI THẤY RÕ CÁC DẤU HẰN RĂNG (VẾT RĂNG LÕM) Ở RÌA LƯỠI (dù nhẹ hay rõ).
- "luoi_beu"="có" KHI THẤY THÂN LƯỜI DÀY PHÌ ĐẠI, TO BÈ LẤY ĐẦY KHOANG MIỆNG. Đánh giá ĐỘC LẬP với dau_rang.
- Chỉ trả JSON, không thêm chữ nào khác.
"""

# Prompt VLM Tiếng Việt - Sắc mặt (JSON CÓ CẤU TRÚC)
FACE_JSON_PROMPT_VI = """Bạn là chuyên gia vọng chẩn Đông y. Hãy quan sát ảnh khuôn mặt và trả về DUY NHẤT một object JSON (không rào codeblock, không thêm bất kỳ văn bản nào khác):
{
  "sac_mat":    "trắng nhợt" | "vàng úa" | "đỏ bừng" | "xanh xao" | "sạm tối" | "hồng hào bình thường" | "không rõ",
  "go_ma_do":   "có" | "không" | "không rõ",
  "phu":        "có" | "không" | "không rõ",
  "ban_do":     "có" | "không" | "không rõ",
  "quang_tham": "có" | "không" | "không rõ",
  "trang_diem": "có" | "không" | "không rõ"
}

HƯỚNG DẪN SOI ẢNH LÂM SÀNG CỤ THỂ:
1. "sac_mat":
   - "trắng nhợt": BẮT BUỘC CHỌN KHI NGƯỜI TRONG ẢNH CÓ DA MẶT TÁI TRẮNG/TÁI BỆCH, KHUÔN MẶT GẦY HỐC HÁC, DA MẶT VÀ MÔI NHẠT MÀU THIẾU TƯƠI NHUẬN / THIẾU MÁU / SUY NHƯỢC (ĐẶC BIỆT KHI THẤY NGƯỜI PHỤ NỮ DA TRẮNG GẦY HỐC HÁC TÓC BÚI HOẶC DA TÁI NHỢT).
   - "vàng úa": BẮT BUỘC CHỌN KHI DA MẶT CÓ TÔNG MÀU VÀNG HOẶC VÀNG CAM RÕ RỆT (trẻ em/trẻ sơ sinh bị vàng da sơ sinh, bệnh nhân hoàng đản/gan).
   - "hồng hào bình thường": CHỈ CHỌN KHI DA MẶT ĐẦY ĐẶN, NỞ NANG, TƯƠI TẮN HỒNG THẮM KHỎE MẠNH (mặt đầy đặn tươi nhuận, không bị gầy hốc hác hay da tái).
   - "đỏ bừng": Chọn khi da mặt đỏ rực như bốc hỏa, sốt cao.
   - "xanh xao": Chọn khi da ngả màu xanh tím.
   - "sạm tối": Chọn khi da u tối, sạm đen, xám xịt.

2. "quang_tham":
   - HÃY SOI KỸ VÙNG MI MẮT DƯỚI (DƯỚI BỌNG MẮT):
   - "có": BẮT BUỘC CHỌN KHI VÙNG DA MI MẮT DƯỚI / BỌNG MẮT CÓ BÓNG THÂM, NẾP NHĂN SẠM NÂU, HOẶC THÂM QUỒNG RÕ RỆT.
   - "không": CHỈ CHỌN KHI VÙNG DƯỚI MẮT ĐỀU MÀU PHẲNG MỊN CÙNG TÔNG DA MẶT.

3. "go_ma_do":
   - "có": CHỈ CHỌN KHI THẤY RÕ 2 MẢNG MÀU ĐỎ HỒNG KHU TRÚ NGAY TRÊN HẠT GÒ MÁ (lưỡng quyền đỏ/hồng của âm hư hỏa vượng, sốt cao). Da mặt bình thường -> MẶC ĐỊNH "không".
   - "không": Khi hai gò má cùng màu với da trán và da mặt.

Chỉ xuất JSON chuẩn.
"""

TONGUE_PROMPT_TEMPLATE = TONGUE_PROMPT_TEMPLATE_VI
FACE_PROMPT_TEMPLATE = FACE_PROMPT_TEMPLATE_VI
