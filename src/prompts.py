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
FACE_JSON_PROMPT_VI = """
Bạn là chuyên gia Đông y vọng chẩn (xem sắc mặt). Quan sát ảnh khuôn mặt và trả về DUY NHẤT một object JSON (không kèm giải thích, không rào ```), theo đúng các khóa và giá trị cho phép sau:
{
  "sac_mat":    "trắng nhợt" | "vàng úa" | "đỏ bừng" | "xanh xao" | "sạm tối" | "hồng hào bình thường" | "không rõ",
  "go_ma_do":   "có" | "không" | "không rõ",
  "phu":        "có" | "không" | "không rõ",
  "ban_do":     "có" | "không" | "không rõ",
  "quang_tham": "có" | "không" | "không rõ",
  "trang_diem": "có" | "không" | "không rõ"
}
QUY TẮC PHÂN BIỆT SẮC MẶT:
- KHI ĐÁNH GIÁ "sac_mat", HÃY SOI KỸ MÀU SẮC DA TRÊN TRÁN, MÁ, MŨI VÀ CỦNG MẠC MẮT (LÒNG TRẮNG MẮT):
  * "vàng úa": CHỌN KHI DA MẶT CÓ TÔNG MÀU VÀNG NỔI BẬT, HOẶC CỦNG MẠC MẮT/VÙNG DA MẶT NHUỐM VÀNG (NHƯ Ở TRẺ VÀNG DA SƠ SINH - JAUNDICE, HOÀNG ĐẢN). ĐẶC BIỆT: NẾU KHUÔN MẶT LÀ TRẺ EM/EM BÉ HOẶC NGUỜI CÓ DA MÀU VÀNG RÕ RỆT THÌ BẮT BUỘC CHỌN "vàng úa" (TUYỆT ĐỐI KHÔNG CHỌN "hồng hào bình thường").
  * "trắng nhợt": khi da tái bệch, nhợt nhạt thiếu tươi nhuận.
  * "hồng hào bình thường": CHỈ CHỌN KHI DA MẶT HỒNG TƯƠI KHỎE MẠNH VÀ KHÔNG CÓ TÔNG MÀU VÀNG RÕ RỆT.
- "trang_diem"="có" CHỈ KHI thấy RÕ LỚP PHẤN NỀN (foundation) che phủ da, SON MÔI ĐẬM HOẶC KẺ MẮT/MASCARA TRANG ĐIỂM RÕ RÀNG. Da mặt mộc tự nhiên -> MẶC ĐỊNH "trang_diem": "không".
- "go_ma_do"="có" CHỈ KHI thấy RÕ hai MẢNG ĐỎ KHU TRÚ ngay trên hai gò má (lưỡng quyền hồng của âm hư).
- "ban_do"="có" CHỈ KHI có BAN/PHÁT BAN THẬT (ban sởi, mề đay, dị ứng lan tỏa). Mụn trứng cá -> "không".
- "quang_tham":
  * "quang_tham"="có": KHI DƯỚI MẮT CÓ MẢNG DA THÂM NÂU / SẠM TỐI / BÓNG THÂM RÕ RỆT.
  * "quang_tham"="không": KHI VÙNG DA DƯỚI MẮT PHẲNG MỊN, SÁNG ĐỀU MÀU CÙNG TÔNG DA MẶT.

MẪU ĐỐI CHIẾU THỰC TẾ:
1) Ảnh em bé/trẻ em hoặc người có da mặt màu vàng rõ rệt (vàng da sơ sinh/hoàng đản) -> {"sac_mat": "vàng úa", "go_ma_do": "không", "phu": "không", "ban_do": "không", "quang_tham": "không", "trang_diem": "không"}
2) Ảnh nữ da trắng gầy hốc hác, da nhạt bệch, mi mắt dưới phẳng sáng màu -> {"sac_mat": "trắng nhợt", "go_ma_do": "không", "phu": "không", "ban_do": "không", "quang_tham": "không", "trang_diem": "không"}
3) Ảnh nam Châu Á da vàng hồng tươi nhuận đầy đặn, mi mắt dưới sạm nâu thâm sẫm -> {"sac_mat": "hồng hào bình thường", "go_ma_do": "không", "phu": "không", "ban_do": "không", "quang_tham": "có", "trang_diem": "không"}
- Chỉ trả JSON, không thêm chữ nào khác.
"""

TONGUE_PROMPT_TEMPLATE = TONGUE_PROMPT_TEMPLATE_VI
FACE_PROMPT_TEMPLATE = FACE_PROMPT_TEMPLATE_VI
