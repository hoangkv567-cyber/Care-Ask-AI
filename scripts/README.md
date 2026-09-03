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

---

# Dữ liệu Tây y (TayY) + ánh xạ ICD-10 + khối "Tây y tham khảo"

## Sơ đồ pipeline (mọi bước đều KHÔNG sửa tay TayY_clean.csv)

```
data/TayY.csv (8.900 dòng, dịch máy — CHỈ ĐỌC)
  └─ clean_tayy_csv.py            -> data/TayY_clean.csv (8.724) + TayY_quarantine.csv + report
       └─ build_icd10_vn_catalog.py -> data/icd10/vn_icd10_tt06_2026.jsonl (15.844 mã TT06/2026-BYT)
       └─ map_tayy_icd10.py         -> proposals.jsonl (top-5 BM25 + 45 mã Tier A exact kép)
       └─ llm_adjudicate_tayy_dups.py -> data/tayy_dup_adjudication.jsonl   [LLM, gợi ý]
       └─ llm_propose_tayy_icd10.py   -> data/icd10/tayy_icd10_llm_proposals.jsonl [LLM, gợi ý]
       └─ review_tayy_icd10.py        -> data/tayy_icd10_reviews.csv  [CHUYÊN GIA — nguồn thật]
       └─ apply_tayy_icd10.py         -> materialize 2 cột icd10_code + trạng_thái vào CSV
       └─ build_tayy_reference_index.py -> data/tayy_reference_index.json (app đọc)
```

## Ba trạng thái kiểm duyệt (cột `trạng_thái_kiểm_duyệt`)
- `chua_kiem_duyet` — mã máy (Tier A hoặc LLM đồng thuận) hoặc chưa có mã. Hiển thị luôn kèm nhãn.
- `da_kiem_duyet`  — chuyên gia approve qua `review_tayy_icd10.py import --apply`.
- `khong_anh_xa`   — chuyên gia phán không ánh xạ được 1 mã đơn.

## Quy tắc bất di bất dịch
1. **LLM chỉ là máy đề xuất.** Mã LLM materialize ở bậc DƯỚI Tier A, trạng thái vẫn
   `chua_kiem_duyet`; chuyên gia phủ quyết qua review sidecar (expert > Tier A > LLM > none).
2. **Không sửa nội dung TayY_clean.csv** (chỉ 2 cột mapping được apply đổi): disease_id khóa
   theo vị trí dòng, `row_sha256` phủ mọi cột nội dung — sửa là vô hiệu proposals 108MB + review.
3. **4 cột cấm** (`tỉ_lệ_chữa_khỏi*`, `đề_xuất_thuốc`, `thuốc_phổ_biến`, `thông_tin_thuốc`)
   không bao giờ vào index/payload/câu trả lời (khóa bằng `test_tayy_reference.py`).
4. **Khối "Tây y tham khảo (ICD-10)"** (key `tayy_reference`) CHỈ role bác sĩ
   (api._strip_clinical_internals cắt); KHÔNG bao giờ vào prompt LLM biện luận.

## Thứ tự lệnh chuẩn
```bash
python scripts/llm_adjudicate_tayy_dups.py --vote-provider dashscope --vote-model qwen-plus
python scripts/llm_propose_tayy_icd10.py --priority P2_validate_tier_a --vote-provider dashscope --vote-model qwen-plus
python scripts/llm_propose_tayy_icd10.py --priority P1_manual_mapping --vote-provider dashscope --vote-model qwen-plus
python scripts/llm_propose_tayy_icd10.py --priority P0_duplicate_name --adjudication data/tayy_dup_adjudication.jsonl --vote-provider dashscope --vote-model qwen-plus
python scripts/llm_propose_tayy_icd10.py --emit-hints     # hints CSV cho reviewer
python scripts/review_tayy_icd10.py validate
python scripts/apply_tayy_icd10.py                        # dry-run -> soi stats -> --apply
python scripts/test_icd10_vn_catalog.py && python scripts/test_tayy_icd10_mapping.py
python scripts/build_tayy_reference_index.py --write
python scripts/test_tayy_reference.py && python scripts/test_payload_phanquyen.py
```
(Requesty đang 402 hết số dư nên phiếu chính chạy DashScope `qwen-plus`; nạp lại tiền thì bỏ
2 cờ `--vote-provider/--vote-model` để về mặc định Requesty.)

## Quy trình duyệt của chuyên gia
1. Mở `data/icd10/tayy_icd10_review_queue.csv` (8.724 dòng, cột `decision` trống) cạnh
   `data/icd10/tayy_icd10_llm_hints.csv` (gợi ý mã LLM + agreement + cờ exact_signal_agreement).
2. Điền `decision` (approved/rejected/needs_more_information/unmappable_*) + `icd10_code`
   (chỉ khi approved) + `reviewer` + `reviewed_at` (ISO-8601 có múi giờ) + `review_note`.
3. `python scripts/review_tayy_icd10.py import --input <file> [--apply]` rồi
   `python scripts/apply_tayy_icd10.py --apply`.

## Quy trình TƯƠNG LAI (expert-gated, chưa code): materialize phân xử trùng tên
Đưa tên phân biệt/gộp dòng vào CSV = re-clean từ TayY.csv với adjudication ĐÃ người duyệt
-> re-map toàn bộ -> chấp nhận vô hiệu mọi sidecar cũ. KHÔNG làm bán tự động.

## Neo4j: import Tây y + cầu Đông–Tây (`kg_import_tayy.py`)

Đưa 8.322 bệnh Tây y (từ `data/tayy_reference_index.json`) lên graph bằng label RIÊNG
`BenhTayY`/`TrieuChungTayY` — TUYỆT ĐỐI không đụng `TrieuChung`/`BenhLy` (vocab trích
triệu chứng quét tất TrieuChung; cổng an toàn khớp chuỗi con tên BenhLy). Cầu nối:
`TƯƠNG_ĐƯƠNG` (triệu chứng, exact + substring cap fanout 5), `TƯƠNG_ỨNG_TÂY_Y`
(53 cạnh xác nhận từ tên ngoặc + disease_tcm_names.json), `LIÊN_QUAN_TRIỆU_CHỨNG`
(602 cạnh máy đề xuất top-3, có score). 3 loại cạnh liên kết xóa-tạo-lại mỗi lần
apply — không bao giờ stale.

```bash
python scripts/kg_import_tayy.py                  # dry-run + worklist tên miss
python scripts/kg_import_tayy.py --apply          # ghi (idempotent)
python scripts/kg_import_tayy.py --export-links   # -> data/dongtay_links.json (app đọc)
python scripts/kg_import_tayy.py --undo           # gỡ sạch theo _nguon
python scripts/test_kg_tayy_import.py             # bất biến vocab + offline logic
```

Sau khi index đổi (build_tayy_reference_index.py --write): chạy lại `--apply` (diff
theo chk per-item) rồi `--export-links`. Chatbot QA đã biết label mới qua
data/graph_schema.txt (câu trả lời dính Tây y luôn kèm nhãn tham khảo).

## Thuốc Tây y tham khảo + khối bệnh nhân rút gọn (policy 2026-08-19)

- `build_tayy_drugs.py` → `data/tayy_drugs.json` (CHỈ cột thuốc phổ biến, lọc rác đề-tài
  + Đông y trá hình hoàn/thang/tán/cao/đan — ngoại lệ 'viên nén phân tán' là thuốc Tây thật).
- Khối "Tây y tham khảo": BÁC SĨ thấy bản đầy đủ + dòng 💊 thuốc tham khảo (nhãn "KHÔNG
  phải đơn thuốc"); ROLE THƯỜNG nhận bản RÚT GỌN qua `api._reduce_tayy_reference` — chỉ
  bệnh `da_kiem_duyet` + mã ICD + nhãn + `TAYY_KHUYEN_CAO` đi khám (allow-list 3 field,
  field mới không tự lọt). Cột `đề_xuất_thuốc`/`thông_tin_thuốc` vẫn CẤM TUYỆT ĐỐI.
- Test: `test_tayy_drugs.py` (bộ lọc), `test_tayy_reference.py` + `test_payload_phanquyen.py`
  (ranh giới 2 role, khóa giá trị thuốc trong payload đã strip).
