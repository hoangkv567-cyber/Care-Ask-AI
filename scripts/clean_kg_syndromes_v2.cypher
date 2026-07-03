// ============================================================================
// LÀM SẠCH HoiChung BẨN — v2 (theo diagnose 2026-07-02 SAU khi graph đã RE-IMPORT lớn)
// Graph mới: BenhLy 238, HoiChung 587, TrieuChung 4685, BaiThuoc 1057 | 39 node "bẩn".
// File clean_7_dirty_hoichung.cypher CŨ đã LỖI THỜI (7 node cũ không còn) — bỏ.
// ============================================================================
// 🔴 ĐÃ BACKUP (Aura Snapshot) rồi mới chạy. apoc.refactor.mergeNodes KHÔNG đảo được.
// Chạy trong Neo4j Browser / cypher-shell.
//
// PHÂN LOẠI 39 node bẩn:
//   NHÓM A (9 node) — TRÙNG LẶP THẬT: tên sau khi bóc = 1 HỘI CHỨNG CHUẨN ĐÃ TỒN TẠI.
//                     Gộp = dồn quan hệ vào node có sẵn, KHÔNG tạo tên mới. ✅ AN TOÀN — chạy bên dưới.
//   NHÓM B (~27 node) — "BỆNH DANH + phân thể/kỳ/hội chứng": bóc ngoặc sẽ biến BỆNH DANH thành
//                     hội chứng (vd 'Cao lâm', 'Hàn sán', 'Cấp hoàng', các 'kỳ' của sởi, 'Cam tích'...).
//                     ⚠️ KHÔNG tự động hoá — cần rà y lý. Các node này ĐÃ bị grounded scoring loại
//                     (regex chặn tên có ngoặc/số) nên KHÔNG hại chẩn đoán. Để nguyên hoặc sửa tay.
//   NHÓM C (3 node) — 'Sơ nhiệt kỳ (...)', 'Trẻ em - Do hàn', 'Trẻ em - Do nhiệt' → sửa tay.
//
// ------------------- NHÓM A: 9 MERGE AN TOÀN -------------------
// (mỗi khối: gộp node bẩn vào node hội chứng chuẩn cùng tên đã tồn tại)

UNWIND [
  ['Phế khí hư (Phế khí hư nhược)', 'Phế khí hư'],
  ['Phong nhiệt (nhiễm trùng)', 'Phong nhiệt'],
  ['Phong nhiệt (thể nhẹ)', 'Phong nhiệt'],
  ['Âm hư (mạn tính)', 'Âm hư'],
  ['Huyết ứ (do chấn thương)', 'Huyết ứ'],
  ['Ứ trệ (khí trệ huyết ứ)', 'Ứ trệ'],
  ['Nhiệt độc (nung mủ)', 'Nhiệt độc'],
  ['Can vị bất hòa (can khí uất trệ)', 'Can vị bất hòa'],
  ['Thấp nhiệt độc thịnh (thể nặng) / Khí dinh (huyết) lưỡng phần nhiệt độc', 'Thấp nhiệt độc thịnh']
] AS pair
MATCH (dirty:HoiChung {name: pair[0]})
MATCH (target:HoiChung {name: pair[1]})
WITH dirty, target WHERE elementId(dirty) <> elementId(target)
CALL apoc.refactor.mergeNodes([target, dirty], {properties:'discard', mergeRels:true}) YIELD node
RETURN node.name AS ket_qua;

// ------------------- Đồng bộ BaiThuoc.hoi_chung (BẮT BUỘC sau merge) -------------------
MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
WHERE p.hoi_chung <> h.name
SET p.hoi_chung = h.name;

// ------------------- VERIFY -------------------
// (1) 9 node bẩn nhóm A không còn (mong = rỗng):
MATCH (h:HoiChung) WHERE h.name IN [
  'Phế khí hư (Phế khí hư nhược)','Phong nhiệt (nhiễm trùng)','Phong nhiệt (thể nhẹ)',
  'Âm hư (mạn tính)','Huyết ứ (do chấn thương)','Ứ trệ (khí trệ huyết ứ)',
  'Nhiệt độc (nung mủ)','Can vị bất hòa (can khí uất trệ)',
  'Thấp nhiệt độc thịnh (thể nặng) / Khí dinh (huyết) lưỡng phần nhiệt độc'
] RETURN h.name;
// (2) mismatch bài thuốc (mong = 0):
MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc) WHERE p.hoi_chung <> h.name RETURN count(p);
// (3) rồi chạy lại: python scripts/kg_maintenance.py diagnose  -> "node bẩn" giảm 9 (39 -> 30)

// ============================================================================
// NHÓM B/C — ĐỂ RÀ TAY (KHÔNG chạy tự động). Xem đầy đủ:
//   python scripts/kg_maintenance.py normalize-names   (dry-run, xem đề xuất + phần "CẦN SỬA TAY")
// Nếu muốn giữ nguyên: các node này không hại chẩn đoán (đã bị grounded scoring loại).
// ============================================================================
