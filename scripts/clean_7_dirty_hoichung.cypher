// ============================================================================
// (TUỲ CHỌN) TRIM WHITESPACE TÊN NODE — vd '\tĐầu thống' -> 'Đầu thống', 'Thấp nhiệt ' -> 'Thấp nhiệt'
// ----------------------------------------------------------------------------
// Frontend ĐÃ tự chuẩn hoá khi hiển thị (qa_system._norm_ws) nên bug NHÌN THẤY đã hết.
// Trim trong graph là dọn tận gốc (tuỳ chọn). CHẠY SAU BACKUP. Aura có APOC.
// ⚠️ Kiểm TRÙNG TÊN trước: nếu trim làm 2 node cùng tên -> cần dedupe (dùng dedupe-hoichung cho HoiChung).
//   (1) Xem trước bệnh/hội chứng dính whitespace:
// MATCH (n) WHERE (n:BenhLy OR n:HoiChung) AND n.name <> apoc.text.regreplace(n.name,'\\s+',' ')
// RETURN labels(n)[0] AS loai, n.name AS cu, apoc.text.regreplace(trim(n.name),'\\s+',' ') AS moi;
//   (2) Trim (sau khi chắc không tạo trùng tên hoặc đã tính dedupe):
// MATCH (b:BenhLy)  WHERE b.name <> apoc.text.regreplace(trim(b.name),'\\s+',' ') SET b.name = apoc.text.regreplace(trim(b.name),'\\s+',' ');
// MATCH (h:HoiChung) WHERE h.name <> apoc.text.regreplace(trim(h.name),'\\s+',' ') SET h.name = apoc.text.regreplace(trim(h.name),'\\s+',' ');
//   (3) Đồng bộ thuộc tính tham chiếu tên trên quan hệ/bài thuốc:
// MATCH ()-[r:CÓ_BIỂU_HIỆN]->() WHERE r.benh_ly IS NOT NULL AND r.benh_ly <> apoc.text.regreplace(trim(r.benh_ly),'\\s+',' ') SET r.benh_ly = apoc.text.regreplace(trim(r.benh_ly),'\\s+',' ');
// MATCH (p:BaiThuoc) WHERE p.benh_ly  IS NOT NULL AND p.benh_ly  <> apoc.text.regreplace(trim(p.benh_ly),'\\s+',' ')  SET p.benh_ly  = apoc.text.regreplace(trim(p.benh_ly),'\\s+',' ');
// MATCH (p:BaiThuoc) WHERE p.hoi_chung IS NOT NULL AND p.hoi_chung <> apoc.text.regreplace(trim(p.hoi_chung),'\\s+',' ') SET p.hoi_chung = apoc.text.regreplace(trim(p.hoi_chung),'\\s+',' ');
// ============================================================================

// ============================================================================
// ⛔ PHẦN 7-NODE DƯỚI ĐÂY ĐÃ LỖI THỜI (graph đã re-import lớn 2026-07-02 chiều: 238 bệnh/587 HC,
//    7 node cũ KHÔNG còn). Dùng scripts/clean_kg_syndromes_v2.cypher THAY THẾ.
//    (Phần TRIM WHITESPACE ở trên vẫn dùng được.)
// ============================================================================
// LÀM SẠCH 7 NODE HoiChung BẨN  (theo baseline diagnose 2026-07-02 SÁNG — STALE)
// ============================================================================
// Graph hiện RẤT sạch: 0 HoiChung trùng tên, 0 mồ côi, 0 BaiThuoc thiếu
// benh_ly/hoi_chung, 0 mismatch. Chỉ còn 7 node tên bẩn (mỗi node = 1 bệnh cụ thể).
//
// 🔴 CHẠY SAU KHI ĐÃ BACKUP (Aura Snapshot). apoc.refactor.mergeNodes KHÔNG đảo được.
// Chạy trong Neo4j Browser / cypher-shell. Mỗi khối đổi 1 node bẩn -> tên sạch:
//   - MERGE node tên sạch (tự tạo nếu chưa có; nếu ĐÃ có -> gộp vào, edge mang benh_ly
//     nên vẫn giữ đúng ngữ cảnh bệnh — app lọc theo r.benh_ly / p.benh_ly).
//   - mergeNodes dồn hết quan hệ, xoá node bẩn.
//   - Đồng bộ p.hoi_chung = h.name để truy hồi bài thuốc không đứt.
//
// LÀM THEO 2 ĐỢT:
//   ĐỢT A (3 ca CƠ HỌC — chỉ bóc tiền tố giai đoạn/đuôi, tên hội chứng lộ rõ): chạy được ngay.
//   ĐỢT B (4 ca CẦN CHUYÊN GIA xác nhận tên): chỉ chạy sau khi bạn/thầy thuốc DUYỆT tên đề xuất.
// ============================================================================

// ---------- Helper: sau MỖI đổi tên, chạy dòng này để đồng bộ bài thuốc ----------
// MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc) WHERE p.hoi_chung <> h.name SET p.hoi_chung = h.name;


// ==================== ĐỢT A — 3 CA CƠ HỌC (an toàn) ====================

// [A1] 'Thể mãn tính - Can kinh thấp nhiệt (đợt cấp)'  ->  'Can kinh thấp nhiệt'   (bệnh: Viêm tai giữa)
MATCH (dirty:HoiChung {name:'Thể mãn tính - Can kinh thấp nhiệt (đợt cấp)'})
MERGE (target:HoiChung {name:'Can kinh thấp nhiệt'})
WITH dirty, target WHERE elementId(dirty) <> elementId(target)
CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
RETURN node.name AS ket_qua;

// [A2] 'Cổ trướng do (Thấp) nhiệt uất huyết ứ'  ->  'Thấp nhiệt uất huyết ứ'        (bệnh: Cổ trướng)
MATCH (dirty:HoiChung {name:'Cổ trướng do (Thấp) nhiệt uất huyết ứ'})
MERGE (target:HoiChung {name:'Thấp nhiệt uất huyết ứ'})
WITH dirty, target WHERE elementId(dirty) <> elementId(target)
CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
RETURN node.name AS ket_qua;

// [A3] 'Thể mãn tính - Thận hư (âm hư hỏa viêm)'  ->  'Thận âm hư'                   (bệnh: Viêm tai giữa)
MATCH (dirty:HoiChung {name:'Thể mãn tính - Thận hư (âm hư hỏa viêm)'})
MERGE (target:HoiChung {name:'Thận âm hư'})
WITH dirty, target WHERE elementId(dirty) <> elementId(target)
CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
RETURN node.name AS ket_qua;

// Đồng bộ bài thuốc sau đợt A:
MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc) WHERE p.hoi_chung <> h.name SET p.hoi_chung = h.name;


// ==================== ĐỢT B — 4 CA CẦN CHUYÊN GIA XÁC NHẬN ====================
// Tên sạch dưới đây là ĐỀ XUẤT dựa trên bài thuốc + triệu chứng trong CSV, KHÔNG phải
// phán quyết y khoa. Hãy đối chiếu với thầy thuốc/tài liệu trước khi chạy. Sửa 'target' nếu cần.
//
// [B1] 'Ung thư dạ dày, đau thượng vị, ăn kém, ợ hơi, gầy sút'   (bệnh: Vị ái)
//      TC: "...thể khí kết thương âm..." | bài: Hành khí tiểu nham thang
//      -> ĐỀ XUẤT: 'Khí kết thương âm'
// MATCH (dirty:HoiChung {name:'Ung thư dạ dày, đau thượng vị, ăn kém, ợ hơi, gầy sút'})
// MERGE (target:HoiChung {name:'Khí kết thương âm'})
// WITH dirty, target WHERE elementId(dirty) <> elementId(target)
// CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
// RETURN node.name;

// [B2] 'Phòng tái phát (bổ can thận khu phong trừ thấp)'          (bệnh: Viêm khớp dạng thấp)
//      bài: Độc hoạt ký sinh thang (kinh điển cho Can Thận hư + phong hàn thấp tý)
//      -> ĐỀ XUẤT: 'Can thận hư'   (hoặc 'Can thận hư kèm phong hàn thấp tý')
// MATCH (dirty:HoiChung {name:'Phòng tái phát (bổ can thận khu phong trừ thấp)'})
// MERGE (target:HoiChung {name:'Can thận hư'})
// WITH dirty, target WHERE elementId(dirty) <> elementId(target)
// CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
// RETURN node.name;

// [B3] 'Biến chứng - Đau ngực (Hung tý)'                          (bệnh: Cường giáp/Basedow)
//      TC: "...can khí uất trệ, nhiệt đàm tắc kinh lạc" | bài: sơ can thông lạc thanh nhiệt hoá đàm
//      -> ĐỀ XUẤT: 'Can uất đàm nhiệt'   (hoặc 'Can khí uất trệ, đàm nhiệt ứ trở')
// MATCH (dirty:HoiChung {name:'Biến chứng - Đau ngực (Hung tý)'})
// MERGE (target:HoiChung {name:'Can uất đàm nhiệt'})
// WITH dirty, target WHERE elementId(dirty) <> elementId(target)
// CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
// RETURN node.name;

// [B4] 'Kinh nghiệm (mới hoặc lâu khó cử động)'                   (bệnh: Kiên tý)
//      bài: "Phương kinh nghiệm dưỡng huyết thư cân" (dưỡng huyết, thư cân)
//      -> ĐỀ XUẤT: 'Khí huyết hư'   (huyết hư không nuôi cân mạch)  — ca mơ hồ nhất, cân nhắc kỹ
// MATCH (dirty:HoiChung {name:'Kinh nghiệm (mới hoặc lâu khó cử động)'})
// MERGE (target:HoiChung {name:'Khí huyết hư'})
// WITH dirty, target WHERE elementId(dirty) <> elementId(target)
// CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
// RETURN node.name;

// Đồng bộ bài thuốc sau đợt B (chạy sau khi đã bỏ comment & chạy các khối B):
// MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc) WHERE p.hoi_chung <> h.name SET p.hoi_chung = h.name;


// ==================== VERIFY (chạy sau cùng) ====================
// (1) 7 tên bẩn không còn:
// MATCH (h:HoiChung)
// WHERE h.name IN [
//   'Ung thư dạ dày, đau thượng vị, ăn kém, ợ hơi, gầy sút',
//   'Phòng tái phát (bổ can thận khu phong trừ thấp)',
//   'Thể mãn tính - Can kinh thấp nhiệt (đợt cấp)',
//   'Biến chứng - Đau ngực (Hung tý)',
//   'Cổ trướng do (Thấp) nhiệt uất huyết ứ',
//   'Kinh nghiệm (mới hoặc lâu khó cử động)',
//   'Thể mãn tính - Thận hư (âm hư hỏa viêm)'
// ] RETURN h.name;                         // kỳ vọng: rỗng (nếu chạy hết A+B)
//
// (2) Không còn mismatch bài thuốc:
// MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc) WHERE p.hoi_chung <> h.name RETURN count(p);  // = 0
//
// (3) Chạy lại: python scripts/kg_maintenance.py diagnose   -> "HoiChung bẩn" giảm còn (7 - số ca đã xử lý)
