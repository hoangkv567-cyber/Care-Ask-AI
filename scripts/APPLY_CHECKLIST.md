# ✅ CHECKLIST LÀM SẠCH KG — chạy `--apply` an toàn

> ### 🟢 CẬP NHẬT theo baseline thực tế (diagnose 2026-07-02)
> Graph SẠCH hơn nhiều so với lo ngại ban đầu: **0 HoiChung trùng tên** (→ `dedupe-hoichung` là
> NO-OP, bỏ qua), **0 mồ côi, 0 BaiThuoc thiếu benh_ly/hoi_chung, 0 mismatch, 0 HoiChung thiếu bài
> thuốc**. Chỉ còn: **7 node HoiChung bẩn** + **84/5893 cạnh CÓ_BIỂU_HIỆN thiếu benh_ly (1.4%)**.
> → **KHÔNG dùng `normalize-names --apply` tự động** cho 7 node này (nó đặt sai tên 2 ca). Thay vào đó
> dùng **`scripts/clean_7_dirty_hoichung.cypher`** (đã soạn riêng, tách 3 ca cơ học vs 4 ca cần chuyên
> gia duyệt). 84 cạnh NULL benh_ly: LOW-PRIORITY — đường truy hồi chặt của app đã lọc theo
> `r.benh_ly=b.name` nên bỏ qua chúng; có thể để nguyên.


> Chạy TẤT CẢ lệnh từ **thư mục gốc dự án**. Cần `.env` có `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD`
> (xem `.env.example`) và `pip install -r requirements.txt`. Aura có sẵn APOC.
> Mọi lệnh `kg_maintenance.py` **mặc định DRY-RUN**; chỉ ghi khi thêm `--apply`.

---

## GIAI ĐOẠN 0 — Chốt baseline (CHỈ ĐỌC, không rủi ro)

```bash
# 0.1  Số liệu graph hiện tại — LƯU LẠI output để so sánh sau
python scripts/kg_maintenance.py diagnose  > baseline_diagnose_truoc.txt

# 0.2  Audit dirt CSV + mô phỏng regression (đã có sẵn báo cáo data/csv_dirt_audit.txt)
python scripts/audit_csv_dirt.py
```
Ghi lại các con số: HoiChung trùng tên, HoiChung bẩn, HoiChung mồ côi, số `BaiThuoc.hoi_chung`
lệch `HoiChung.name`. Đây là mốc để đối chiếu sau khi apply.

---

## GIAI ĐOẠN 1 — 🔴 BACKUP (BẮT BUỘC — mergeNodes/delete KHÔNG đảo được)

**Cách A (khuyên dùng): Aura Console → Snapshot**
1. Vào Neo4j Aura Console → chọn instance → tab **Snapshots** → **Take snapshot**.
2. Đợi snapshot "Completed". Đây là điểm khôi phục toàn bộ nếu có sự cố.

**Cách B: export ra file Cypher (nếu không tạo được snapshot)**
```cypher
// chạy trong Neo4j Browser
CALL apoc.export.cypher.all('backup_kg.cypher', {format:'cypher-shell'})
```
> ⚠️ Backup Neo4j **KHÔNG** bảo vệ file `data/Medicine_clean.csv`. Ở checklist này ta **KHÔNG đụng CSV**
> (tách symptom sẽ phá matching — xem README). Nếu sau này sửa CSV: `cp data/Medicine_clean.csv data/Medicine_clean.bak.csv` trước.

---

## GIAI ĐOẠN 2 — Xem trước (DRY-RUN, chưa ghi gì)

```bash
python scripts/kg_maintenance.py dedupe-hoichung      # các nhóm HoiChung trùng tên sẽ gộp
python scripts/kg_maintenance.py clean-dirty          # danh sách HoiChung "bẩn" + bậc (số quan hệ)
python scripts/kg_maintenance.py normalize-names      # bảng old->new + mục "CẦN SỬA TAY"
```

**🩺 REVIEW TAY trước khi sang giai đoạn 3 — quy tắc bắt buộc:**
- Trong `clean-dirty`: nếu có node **bậc cao (nhiều quan hệ)** hoặc tên **ghép hợp lệ >40 ký tự**
  (vd *"Can khí uất kết kèm đờm trệ"*) → **KHÔNG** được nằm trong diện `--delete`. Chỉ gắn cờ.
- Trong `normalize-names`: kiểm mỗi cặp `old -> new` xem có **mất vế phân biệt** không
  (vd *"Cam tích (thể vừa) - Tỳ hư tích trệ"* → *"Cam tích"* là MẤT thông tin → bỏ qua, sửa tay).
- Các tên `' - '` / `'/'` bị đánh **"CẦN SỬA TAY"** (vd *"Cấp tính - Phong nhiệt"*,
  *"Can uất hoá hoả / Thấp nhiệt nung nấu"*): để nguyên, xử lý bằng tay sau (xem Phụ lục).

---

## GIAI ĐOẠN 3 — Apply theo thứ tự ĐẢO-ĐƯỢC-TRƯỚC

```bash
# 3.1  Gắn cờ node bẩn (CHỈ set _flagged_dirty=true — ĐẢO ĐƯỢC, chưa xoá gì)
python scripts/kg_maintenance.py clean-dirty --apply
python scripts/kg_maintenance.py diagnose            # kiểm: chưa mất node, chỉ có cờ

# 3.2  Gộp HoiChung trùng tên (an toàn: model dùng chung theo name, mergeRels giữ hết quan hệ)
python scripts/kg_maintenance.py dedupe-hoichung --apply

# 3.3  Chuẩn hoá tên bẩn (tự đồng bộ BaiThuoc.hoi_chung = HoiChung.name)
python scripts/kg_maintenance.py normalize-names --apply
```

> ⛔ **CHƯA** chạy `clean-dirty --apply --delete`. Chỉ xoá hẳn khi đã chắc từng node bẩn KHÔNG mang
> dữ liệu thật (bài thuốc/triệu chứng). Nếu quyết xoá, chạy lại `clean-dirty` (dry-run) để soát lần cuối.

**(Tuỳ chọn) làm giàu triệu chứng cho hội chứng cốt lõi thưa — RÀ bảng tri thức trong file trước:**
```bash
python scripts/kg_enrich_symptoms.py          # xem trước
python scripts/kg_enrich_symptoms.py --apply  # (đảo được: --undo)
```

**(Tuỳ chọn) thêm phối ngũ Thập bát phản / Thập cửu úy:**
```bash
python scripts/kg_maintenance.py add-phoi-ngu          # xem cặp
python scripts/kg_maintenance.py add-phoi-ngu --apply  # (đảo được: xoá cạnh TƯƠNG_PHẢN/TƯƠNG_ÚY)
```

---

## GIAI ĐOẠN 4 — Verify (BẮT BUỘC sau khi apply)

```bash
# 4.1  So sánh với baseline: node bẩn giảm, HoiChung trùng = 0, mismatch BaiThuoc.hoi_chung = 0
python scripts/kg_maintenance.py diagnose  > baseline_diagnose_sau.txt

# 4.2  Grounded scoring còn đúng không (chọn hội chứng đặc hiệu, không loạn)
python scripts/test_grounded_scoring.py "mệt mỏi, đại tiện lỏng, hằn răng"

# 4.3  QUAN TRỌNG: chạy lại app trên 2-3 ca THẬT (kể cả ca hô hấp trước đây ra "Áp xe gan")
#      để chắc bộ khớp bệnh (gồm bộ lọc generic-symptom) KHÔNG regress.
```
Đối chiếu `baseline_diagnose_truoc.txt` vs `_sau.txt`:
- ✅ `HoiChung trùng tên` → 0
- ✅ `BaiThuoc.hoi_chung <> HoiChung.name` → 0 (normalize-names tự đồng bộ)
- ✅ số node HoiChung giảm đúng bằng số node trùng đã gộp
- ✅ số bài thuốc / triệu chứng KHÔNG giảm bất thường (nếu giảm mạnh → có node thật bị xoá nhầm → ROLLBACK)

---

## 🔙 ROLLBACK nếu sai
- Gỡ cờ dirty (đảo 3.1): `MATCH (h:HoiChung) WHERE h._flagged_dirty REMOVE h._flagged_dirty`
- Gỡ enrich: `python scripts/kg_enrich_symptoms.py --undo`
- Gỡ phối ngũ: `MATCH ()-[r:TƯƠNG_PHẢN|TƯƠNG_ÚY]->() DELETE r`
- Mất node do dedupe/normalize/delete (KHÔNG đảo bằng script): **khôi phục từ Snapshot/backup ở Giai đoạn 1**.

---

## Phụ lục — ~24 hội chứng "CẦN SỬA TAY" (làm bằng Cypher, sau khi backup)
Các tên `' - '` / `'/'` / staging mà tool cố ý KHÔNG tự đổi (tránh phá hội chứng ghép hợp lệ).
Xem danh sách phân loại trong `data/csv_dirt_audit.txt` (mục 3). Ví dụ đổi tên 1 node + đồng bộ bài thuốc:
```cypher
// Ví dụ: 'Cấp tính - Phong nhiệt'  ->  'Phong nhiệt'  (CHỈ khi bạn chắc về y lý)
MATCH (dirty:HoiChung {name:'Cấp tính - Phong nhiệt'})
MERGE (target:HoiChung {name:'Phong nhiệt'})
WITH dirty, target WHERE elementId(dirty) <> elementId(target)
CALL apoc.refactor.mergeNodes([target,dirty],{properties:'discard',mergeRels:true}) YIELD node
RETURN node;
// rồi đồng bộ:
MATCH (h:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc) WHERE p.hoi_chung <> h.name SET p.hoi_chung = h.name;
```
> ⚠️ Tên có `'/'` (2 hội chứng) như *"Can uất hoá hoả / Thấp nhiệt nung nấu"*: **đừng** tách thành 2 node
> (phá lookup `HoiChung {name:$syn}`). Chọn 1 tên chuẩn hoặc giữ nguyên.
