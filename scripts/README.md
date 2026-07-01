# scripts/ — Công cụ bảo trì Knowledge Graph

## `kg_maintenance.py`
Bảo trì graph Neo4j: chẩn đoán, gộp node trùng, làm sạch node bẩn, thêm quan hệ phối ngũ.
Kết nối qua `.env` (dùng `src/config_loader.py`, không hardcode secret). **Mặc định DRY-RUN.**

### Chuẩn bị
1. Đã có `.env` với `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD` (xem `.env.example`).
2. Cài phụ thuộc: `pip install -r requirements.txt` (cần `neo4j`, `pyyaml`).
3. **BACKUP trước khi `--apply`**: Neo4j Aura → Snapshot; hoặc `CALL apoc.export.cypher.all(...)`.

### Lệnh (chạy từ thư mục gốc dự án)
```bash
# 1) Chẩn đoán (CHỈ ĐỌC — luôn chạy trước để xem số liệu thật)
python scripts/kg_maintenance.py diagnose

# 2) Gộp HoiChung trùng tên (cần APOC — Aura có sẵn)
python scripts/kg_maintenance.py dedupe-hoichung          # xem trước
python scripts/kg_maintenance.py dedupe-hoichung --apply  # thực thi

# 3) Rà HoiChung "bẩn" (rác NLP)
python scripts/kg_maintenance.py clean-dirty                     # liệt kê
python scripts/kg_maintenance.py clean-dirty --apply             # gắn cờ _flagged_dirty (đảo được)
python scripts/kg_maintenance.py clean-dirty --apply --delete    # XÓA hẳn (nguy hiểm — node bẩn có data thật thì ĐỪNG)

# 3b) Chuẩn hoá tên bẩn (bóc bài thuốc/giai đoạn khỏi tên hội chứng) — GIỮ nguyên quan hệ
python scripts/kg_maintenance.py normalize-names          # xem old->new + cảnh báo trùng
python scripts/kg_maintenance.py normalize-names --apply  # đổi tên; trùng tên -> MERGE bằng APOC

# 4) Thêm phối ngũ Thập bát phản / Thập cửu úy (chỉ tạo khi cả 2 vị thuốc đã có trong graph)
python scripts/kg_maintenance.py add-phoi-ngu          # xem cặp sẽ tạo/bỏ qua
python scripts/kg_maintenance.py add-phoi-ngu --apply

# 5) Chuẩn hoá nguồn DongY.hội_chứng (để re-import không tái nhiễm tên bẩn)
python scripts/kg_maintenance.py normalize-source
python scripts/kg_maintenance.py normalize-source --apply
```

## `kg_enrich_symptoms.py`
Làm giàu symptom list cho các HoiChung CỐT LÕI còn thưa (vd Tỳ khí hư) → grounded-scoring chọn
được hội chứng đặc hiệu thay vì hội chứng chung (Khí hư). Cạnh thêm gắn `nguon:'enrich'`, KHÔNG có
benh_ly nên chỉ giúp chấm điểm, không ảnh hưởng truy hồi bài thuốc.

⚠️ RÀ SOÁT bảng `SYNDROME_SYMPTOMS` trong file (tri thức Đông y) theo chuyên môn TRƯỚC khi --apply.
```bash
python scripts/kg_enrich_symptoms.py           # xem trước kế hoạch
python scripts/kg_enrich_symptoms.py --apply   # tạo cạnh
python scripts/kg_enrich_symptoms.py --undo    # gỡ hết cạnh enrich (đảo được)
# rồi kiểm tra lại:
python scripts/test_grounded_scoring.py
```

## `test_grounded_scoring.py`
Test grounded syndrome-scoring độc lập (chỉ cần Neo4j). Xem hội chứng nào lên top cho 1 bộ triệu chứng.
```bash
python scripts/test_grounded_scoring.py
python scripts/test_grounded_scoring.py "mệt mỏi, đại tiện lỏng, hằn răng"
```

### Lưu ý thiết kế
- **Dedupe theo `name`** (không theo `(name, benh_ly)`): model hiện tại dùng chung HoiChung theo tên,
  ngữ cảnh bệnh nằm trên thuộc tính quan hệ (`r.benh_ly`, `p.benh_ly/hoi_chung`). Tách theo
  `(name, benh_ly)` sẽ **phá query `HoiChung {name: $syn}`** của app → không làm ở đây.
- **Heuristic node bẩn** chỉnh trong hằng `_DIRTY_HOICHUNG_WHERE`. Nên chạy `--apply` (gắn cờ) và
  review thủ công trước khi `--delete`.
- **Phối ngũ** dùng dữ liệu cổ điển; tên vị thuốc có thể khác dị bản trong DB → script báo cáo
  cặp bị bỏ qua (thiếu vị thuốc) để bạn bổ sung alias nếu cần.
