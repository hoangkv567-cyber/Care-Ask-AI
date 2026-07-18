// ROLLBACK CÓ NHẮM — gỡ ĐÚNG phần import "Viêm đại tràng × Tỳ khí hư" (2026-07-17)
//
// ⚠ ĐỪNG dùng `kg_import_csv_gaps.py --undo` để hoàn tác lần import này!
//   Nhãn _nguon/nguon = 'csv_gap_sync' là nhãn của SCRIPT, KHÔNG phải của từng lần chạy: graph đang
//   mang 40 node + ~494 cạnh cùng nhãn đó từ các lần import TRƯỚC (Ức can tán, Ngân kiều tán,
//   Thủy ngưu giác...). `--undo` sẽ xoá sạch cả những thứ đó chứ không riêng phần này.
//
// Dấu chân của lần import này (đã đối chiếu graph thật): 1 cạnh CHIA_THÀNH, 1 node BaiThuoc
// (kèm 11 cạnh BAO_GỒM + 1 cạnh ĐƯỢC_ĐIỀU_TRỊ_BẰNG), 8 cạnh CÓ_BIỂU_HIỆN.
//
// ⚠ KHÔNG được xoá node HoiChung 'Tỳ khí hư': nó dùng chung với Hư lao, Huyết áp thấp, Viêm yết hầu.
// ⚠ KHÔNG được xoá các node ViThuoc: đều là vị dùng chung toàn kho.
//
// Kiểm tra TRƯỚC khi xoá (phải ra: chia_thanh=1, bai_thuoc=1, co_bieu_hien=8):
//   MATCH (b:BenhLy {name:'Viêm đại tràng'})-[r:CHIA_THÀNH]->(:HoiChung {name:'Tỳ khí hư'})
//   WITH count(r) AS chia_thanh
//   MATCH (p:BaiThuoc) WHERE p.benh_ly='Viêm đại tràng' AND p.hoi_chung='Tỳ khí hư'
//   WITH chia_thanh, count(p) AS bai_thuoc
//   MATCH (:HoiChung {name:'Tỳ khí hư'})-[r2:CÓ_BIỂU_HIỆN {benh_ly:'Viêm đại tràng'}]->(:TrieuChung)
//   RETURN chia_thanh, bai_thuoc, count(r2) AS co_bieu_hien;

// (1) Gỡ node bài thuốc riêng của cặp bệnh×thể này (DETACH cuốn theo BAO_GỒM + ĐƯỢC_ĐIỀU_TRỊ_BẰNG).
MATCH (p:BaiThuoc)
WHERE p.benh_ly = 'Viêm đại tràng' AND p.hoi_chung = 'Tỳ khí hư'
DETACH DELETE p;

// (2) Gỡ các cạnh biểu hiện gắn riêng bệnh này (node TrieuChung giữ lại, xử lý ở bước 4).
MATCH (:HoiChung {name:'Tỳ khí hư'})-[r:CÓ_BIỂU_HIỆN {benh_ly:'Viêm đại tràng'}]->(:TrieuChung)
DELETE r;

// (3) Gỡ cạnh phân thể — CHỈ cạnh, KHÔNG đụng node 'Tỳ khí hư' (dùng chung 3 bệnh khác).
MATCH (:BenhLy {name:'Viêm đại tràng'})-[r:CHIA_THÀNH]->(:HoiChung {name:'Tỳ khí hư'})
DELETE r;

// (4) Dọn node TrieuChung mồ côi do bước (2) để lại — chỉ xoá cái KHÔNG còn quan hệ nào.
MATCH (t:TrieuChung)
WHERE t._nguon = 'csv_gap_sync' AND NOT (t)--()
DELETE t;

// Sau khi chạy: nhớ gỡ luôn dòng CSV tương ứng trong data/Medicine_clean.csv
// ("Viêm đại tràng,Tỳ khí hư,...,Sâm Linh Bạch Truật Tán gia giảm,...") để CSV và graph không lệch nhau.
