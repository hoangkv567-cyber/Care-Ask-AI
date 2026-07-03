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

## `audit_csv_dirt.py`  (CHỈ ĐỌC — chạy TRƯỚC mọi việc làm sạch)
Audit độ bẩn của `data/Medicine_clean.csv` (triệu_chứng + hội_chứng) và **mô phỏng regression**.
Không sửa CSV, không chạm DB. Ghi báo cáo `data/csv_dirt_audit.txt`.
```bash
python scripts/audit_csv_dirt.py
```
Kết quả hiện tại (992 dòng): triệu_chứng có ~427 dòng chứa field bẩn (run-on 493, mash mạch+lưỡi 133,
prose≥8từ 751, ngoặc lệch cặp 62); hội_chứng 555 distinct → 496 CLEAN, 23 ngoặc-chú-thích, 12 dấu ' - ',
11 ghép-hợp-lệ, 9 nghi-staging, 3 dấu '/', 1 >40 ký tự.

## `eval_disease_matching.py`  (BỘ ĐO metrics — phủ toàn 992 bệnh, không cần Neo4j)
Đo chất lượng `_find_matching_diseases` bằng self-consistency của CSV (không cần nhãn chuyên gia).
Dùng để so sánh TRƯỚC/SAU khi đổi thuật toán khớp.
```bash
python scripts/eval_disease_matching.py
```
Chỉ số: **M1 Self@top1** (đưa full triệu chứng → bệnh là #1), **M2 SpecRecall** (đưa chỉ triệu chứng
đặc hiệu → recall/hạng), **M3 OverMatch** (input toàn generic → số ứng viên, muốn ~0), **M4 Clinical**
(4 ca forbid/expect). Baseline hiện tại: M1 85.3%, M2 in-cand 86.4% / top3 86.3%, M3 [3,2,0,0], M4 4/4.

**IDF symptom-weighting** (đã thử & tinh chỉnh qua bộ đo này): thêm cổng `_idf_peak_min=4.0` trong
`_find_matching_diseases` — bệnh chỉ được đặt tên nếu khớp ≥1 triệu chứng đủ đặc hiệu (idf≥4.0, xuất
hiện ở ≲18 bệnh). Kết luận thực nghiệm: IDF chỉ cải thiện **marginal** (bớt over-match generic mà không
giảm recall / không phá ca lâm sàng); đẩy ngưỡng cao hơn (≥5) thì recall tụt & ca lâm sàng vỡ → matcher
chồng-lấn-phẳng đã gần trần độ chính xác, các guard chuyên khoa + generic-filter mới là phần gánh chính.

## `test_disease_matching.py`  (BỘ TEST HỒI QUY — chạy sau mọi thay đổi tầng khớp bệnh)
Khoá lại các fix đã làm cho `_find_matching_diseases` (bộ lọc generic-symptom + guard chuyên khoa:
nam khoa, sản khoa/động thai, Hung tý). CHỈ ĐỌC CSV, không cần Neo4j. Exit 0 nếu tất cả PASS.
```bash
python scripts/test_disease_matching.py
```
Mỗi ca kiểm `forbid` (bệnh danh vô lý TUYỆT ĐỐI không được khớp — Áp xe gan/Bạch tiển/Dương nuy/Hung
tý/Động thai cho ca hô hấp) + `expect` (bệnh hợp lý phải có — COPD; Hung tý cho ca đau ngực thật;
Dương nuy cho ca liệt dương thật; Động thai cho thai phụ). **Chạy test này TRƯỚC/SAU mọi chỉnh sửa
`_find_matching_diseases` hoặc `_validate_disease_safety`.**

## ⚠️ RUNBOOK LÀM SẠCH AN TOÀN (đọc kỹ — đã kiểm chứng đối kháng)

**KẾT LUẬN QUAN TRỌNG (bằng chứng thật):** KHÔNG tách/split cột `triệu_chứng` trong CSV một cách
ngây thơ. `src/fusion_pipeline.py` đọc `data/Medicine_clean.csv` **lúc runtime** và
`_find_matching_diseases` tính `match_ratio = matched_count / len(field tách theo ',')`. Tách thêm
run-on trên `.`/`;` làm **phình mẫu số → tụt ratio dưới ngưỡng 0.30 → bệnh đúng biến mất im lặng**.
`audit_csv_dirt.py` đo được **135 dòng** sẽ tăng ngưỡng matched tối thiểu nếu tách (vd row914 flip
0.444→0.286). Muốn tách symptom thì PHẢI re-tune ngưỡng 0.30 + test recall trên bộ ca có nhãn TRƯỚC.

**3 NGUỒN phải đồng bộ, không dọn một nửa:** (a) `data/Medicine_clean.csv` (matching bệnh danh,
runtime), (b) graph Neo4j (TrieuChung/HoiChung — syndrome scoring & truy hồi), (c)
`data/mapping/symptom_to_syndrome.json` (dựng `tongue_symptoms_list` cho vọng chẩn). Repo **KHÔNG có
script re-import CSV→graph** — đây là blocker cho vòng khép kín; `normalize-source` chỉ vá node staging
`:DongY` của pipeline import NGOÀI repo.

**Thứ tự AN TOÀN:**
1. `python scripts/kg_maintenance.py diagnose` (read-only) — chốt baseline, lưu output.
2. `python scripts/audit_csv_dirt.py` (read-only) — chốt dirt CSV + regression.
3. **BACKUP graph** trước MỌI `--apply`: Aura Snapshot hoặc `CALL apoc.export.cypher.all(...)`. Lưu ý:
   backup Neo4j **KHÔNG** bảo vệ việc bạn sửa file CSV — coi thay CSV là thay đổi PRODUCTION.
4. DRY-RUN (mặc định) các lệnh graph & ĐỌC kỹ phần "CẦN SỬA TAY":
   `dedupe-hoichung` (an toàn — model shared-by-name, mergeRels), `clean-dirty`, `normalize-names`.
5. **Review tay các ca mơ hồ** (báo cáo audit + dry-run): hội chứng ghép hợp lệ >40 ký tự KHÔNG được
   xoá; `normalize-names` có thể làm mất vế phân biệt ('Cam tích (thể vừa) - Tỳ hư tích trệ' → 'Cam
   tích'); dấu '/' và ' - ' phải phân loại staging-rác vs ghép-hợp-lệ bằng tay.
6. `--apply` theo thứ tự đảo-được-trước: `clean-dirty --apply` (chỉ gắn cờ, ĐẢO ĐƯỢC) → kiểm tra →
   `dedupe-hoichung --apply` → `normalize-names --apply` (tự đồng bộ `BaiThuoc.hoi_chung`). Chỉ
   `clean-dirty --apply --delete` khi đã chắc chắn. **Đây là hành động thủ công của bạn sau khi backup.**
7. Sau `--apply`: chạy lại `diagnose` + `test_grounded_scoring.py`, xác nhận mismatch `BaiThuoc.hoi_chung`=0,
   số node bẩn giảm, và **chạy lại chẩn đoán trên vài ca thật** để chắc bộ khớp bệnh (gồm bộ lọc
   generic-symptom mới) không regress.

**KHÔNG BAO GIỜ:** ghi đè `data/Medicine_clean.csv` bằng bản đã split mà chưa re-tune + test recall;
`--delete` node HoiChung nhóm size>40 khi chưa có allowlist ghép-hợp-lệ; đổi tên `BenhLy` (phá join
bài thuốc).

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
