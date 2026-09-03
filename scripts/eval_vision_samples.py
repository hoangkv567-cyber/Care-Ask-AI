#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/eval_vision_samples.py — ĐO tầng vọng chẩn trên bộ mẫu ảnh có nhãn.

VÌ SAO CÓ FILE NÀY: trước đây tầng đọc ảnh là tầng DUY NHẤT không đo được. Mọi tầng khác đều khóa
bằng test và chứng minh bằng đột biến; riêng vọng chẩn thì ảnh bị xóa ngay sau mỗi lượt chẩn
(api.py: "Đã dọn dẹp file tạm"), nên sửa prompt xong chỉ có thể ĐOÁN là tốt lên hay xấu đi.
Hệ quả đo được trong thực tế: một ảnh có quầng thâm rõ bị trả 'không', một ảnh lưỡi rìa nhẵn lại
bị đọc thành 'có dấu răng' — không ca nào kiểm chứng lại được vì ảnh đã mất.

CÁCH DÙNG
  1. Bật lưu mẫu, chạy app như thường ngày:      set TCM_SAVE_VISION_SAMPLES=1
     Mỗi ca sinh một thư mục trong data/vision_samples/ gồm ảnh + vision.json + nhan.json (trống).
  2. Người có chuyên môn mở nhan.json, điền nhãn ĐÚNG (để trống = chưa gán nhãn, sẽ bị bỏ qua).
  3. Đo mô hình hiện tại:            python scripts/eval_vision_samples.py
     Chạy lại VLM trên chính ảnh cũ: python scripts/eval_vision_samples.py --rerun
     So hai lần chạy:                python scripts/eval_vision_samples.py --rerun --save-as thu-nghiem-A
                                     python scripts/eval_vision_samples.py --compare thu-nghiem-A

  --rerun là chỗ tạo ra giá trị thật: sửa prompt xong chạy lại TRÊN CÙNG BỘ ẢNH rồi so, thay vì
  đoán. Không có --rerun thì chỉ chấm lại vision.json đã lưu sẵn.

⚠ Bộ mẫu chứa ảnh KHUÔN MẶT người bệnh — data/vision_samples/ đã bị .gitignore chặn. Đừng commit,
đừng đẩy lên dịch vụ ngoài.
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "data", "vision_samples")

# Trường nào thuộc ảnh nào — dùng để chấm đúng phương thức, không lẫn lưỡi sang mặt.
TONGUE_KEYS = ("than_luoi", "reu_mau", "reu_day", "reu_chat", "dau_rang", "vet_nut", "luoi_beu")
FACE_KEYS = ("sac_mat", "go_ma_do", "phu", "ban_do", "quang_tham", "trang_diem")


def _load(p, mac_dinh=None):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except Exception:
        return mac_dinh if mac_dinh is not None else {}


def _doc_mau(d):
    """Trả (nhãn_đúng, vision_đã_lưu, đường_dẫn_ảnh). Bỏ qua mẫu chưa gán nhãn."""
    nhan = _load(os.path.join(d, "nhan.json"))
    if not nhan:
        return None
    co_nhan = any(str(v).strip() for nhom in ("tongue", "face")
                  for v in (nhan.get(nhom) or {}).values())
    if not co_nhan:
        return None
    anh = {}
    for ten in ("face", "tongue"):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = os.path.join(d, ten + ext)
            if os.path.exists(p):
                anh[ten] = p
                break
    return nhan, _load(os.path.join(d, "vision.json")), anh


def _du_doan_tu_vision(vd):
    """Bóc dự đoán theo TRƯỜNG từ vision.json. Ưu tiên khối 'structured' (đường JSON tất định);
    nếu ca đó chạy đường prose thì không có structured -> trả rỗng, và mẫu bị tính là KHÔNG ĐO
    ĐƯỢC chứ không tính là sai (tránh đổ lỗi oan cho prompt JSON)."""
    st = (vd or {}).get("structured") or {}
    if isinstance(st, dict) and ("tongue" in st or "face" in st):
        return {"tongue": st.get("tongue") or {}, "face": st.get("face") or {}}
    # Một số bản lưu gộp phẳng -> tách theo khóa
    return {"tongue": {k: st[k] for k in TONGUE_KEYS if k in st},
            "face": {k: st[k] for k in FACE_KEYS if k in st}}


def _chay_lai(anh):
    """Gọi lại VLM trên chính ảnh cũ với prompt HIỆN TẠI — đây là chỗ đo được thay đổi prompt."""
    from src.config_loader import load_config
    from src.cloud_vlm_client import create_vision_client
    if not hasattr(_chay_lai, "_c"):
        _chay_lai._c = create_vision_client(load_config())
    out = {}
    for mod, p in (("tongue", anh.get("tongue")), ("face", anh.get("face"))):
        if not p:
            continue
        r = _chay_lai._c.diagnose_image_structured(p, modality=mod)
        out[mod] = (r or {}).get("data") or {}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="gọi lại VLM trên ảnh cũ (đo prompt hiện tại)")
    ap.add_argument("--save-as", help="lưu kết quả lượt này để so về sau")
    ap.add_argument("--compare", help="so với một lượt đã lưu")
    args = ap.parse_args()

    if not os.path.isdir(SAMPLE_DIR):
        print(f"Chưa có {SAMPLE_DIR}. Bật TCM_SAVE_VISION_SAMPLES=1 rồi chạy app để thu mẫu.")
        return 0
    thu_muc = sorted(d for d in os.listdir(SAMPLE_DIR)
                     if os.path.isdir(os.path.join(SAMPLE_DIR, d)))
    if not thu_muc:
        print("Bộ mẫu rỗng.")
        return 0

    dung, sai, bo_qua = 0, 0, 0
    per_key = {}
    chi_tiet = {}
    for ten in thu_muc:
        d = os.path.join(SAMPLE_DIR, ten)
        got = _doc_mau(d)
        if not got:
            bo_qua += 1
            continue
        nhan, vd, anh = got
        du_doan = _chay_lai(anh) if args.rerun else _du_doan_tu_vision(vd)
        chi_tiet[ten] = du_doan
        for nhom, keys in (("tongue", TONGUE_KEYS), ("face", FACE_KEYS)):
            for k in keys:
                mong = str((nhan.get(nhom) or {}).get(k, "")).strip().lower()
                if not mong:
                    continue                      # chưa gán nhãn trường này
                thuc = str((du_doan.get(nhom) or {}).get(k, "")).strip().lower()
                if not thuc:
                    bo_qua += 1
                    continue                      # không đo được (ca chạy đường prose)
                ok = (thuc == mong)
                per_key.setdefault(k, [0, 0])
                per_key[k][0 if ok else 1] += 1
                if ok:
                    dung += 1
                else:
                    sai += 1
                    print(f"  [SAI] {ten}  {nhom}.{k}: mô hình='{thuc}'  đúng='{mong}'")

    tong = dung + sai
    print(f"\nĐÚNG {dung}/{tong}" + (f" = {100.0*dung/tong:.1f}%" if tong else "")
          + f"  | mẫu/trường bỏ qua (chưa gán nhãn hoặc không đo được): {bo_qua}")
    if per_key:
        print("\nTheo từng dấu — cột SAI mới là chỗ đáng sửa prompt:")
        for k, (a, b) in sorted(per_key.items(), key=lambda x: -x[1][1]):
            print(f"   {k:<12} đúng {a:<3} sai {b}")

    if args.save_as:
        p = os.path.join(SAMPLE_DIR, f"_luot-{args.save_as}.json")
        io.open(p, "w", encoding="utf-8").write(json.dumps(chi_tiet, ensure_ascii=False, indent=1))
        print(f"\nĐã lưu lượt này: {p}")
    if args.compare:
        cu = _load(os.path.join(SAMPLE_DIR, f"_luot-{args.compare}.json"))
        n_doi = 0
        for ten, moi in chi_tiet.items():
            for nhom in ("tongue", "face"):
                for k, v in (moi.get(nhom) or {}).items():
                    v_cu = ((cu.get(ten) or {}).get(nhom) or {}).get(k)
                    if v_cu is not None and str(v_cu) != str(v):
                        n_doi += 1
                        print(f"  [ĐỔI] {ten} {nhom}.{k}: '{v_cu}' -> '{v}'")
        print(f"\nSo với lượt '{args.compare}': {n_doi} trường đổi kết quả.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
