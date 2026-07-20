#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/build_batcuong_golden.py — Sinh/cập nhật data/batcuong_golden.json.

VÌ SAO CÓ HARNESS NÀY: suốt 9 vòng vá, eval_gold và eval_disease_matching (M1 76.3% / M4 4/4)
BẤT BIẾN HOÀN TOÀN — tức 41 file test KHÔNG hề phủ tầng Bát Cương. Ba hồi quy thật đã xảy ra
(mất parity dấu nhiệt, xóa mất biện luận, ĐẢO CỰC Bát Cương) đều chỉ được phát hiện khi người
dùng dán kết quả ra, không phải bởi test.

CHẠY ĐƯỢC OFFLINE — đo được: Bát Cương chốt TRƯỚC lượt gọi LLM sinh prose, nên chặn LLM bằng
ĐÚNG MỘT seam (qa_pipeline.client.chat) thì Mục 1/2 vẫn nguyên giá trị.
    offline ~4s/ca, 0 lượt gọi LLM   |   live 15-83s/ca, 3 lượt
    đã kiểm: offline == live trên cả 3 ca hồi quy; và tất định qua 3 lần chạy.
Vẫn CẦN Neo4j (metadata hội chứng).

⚠ KHÔNG chốt golden cho ca mà lời khai quá mơ hồ: khi LLM bị chặn, nhánh suy giảm mềm dựng Mục 1/2
theo đường KHÁC production. Script tự loại các ca đó (thiếu bệnh danh hoặc thiếu core).

Chạy:  PYTHONIOENCODING=utf-8 python scripts/build_batcuong_golden.py [--limit N]
Sinh xong PHẢI đọc tay file JSON trước khi commit — golden là thứ đóng băng hành vi, chốt sai
thì khóa luôn cái sai.
"""
import argparse
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import logging  # noqa: E402
logging.disable(logging.WARNING)

GOLDEN = os.path.join(ROOT, "data", "batcuong_golden.json")

# Ca HỒI QUY — mỗi ca neo vào một lỗi THẬT đã xảy ra. Đây là phần giá trị nhất của corpus:
# chúng là bằng chứng sống rằng harness bắt được đúng lớp lỗi đã từng lọt.
REGRESSION_CASES = [
    {"id": "reg_khat_tran_duonghu",
     "raw": ("34 tuổi, nữ giới, bệnh vài ngày, tiểu tiện trong dài, tinh thần uể oải, nước tiểu "
             "trong, khát nước, tiểu đêm, đau lưng, mỏi gối, rêu trắng mỏng, lưỡi bệu"),
     "source": "hồi quy #1 — đảo cực: 'khát nước' TRẦN dựng trục Nhiệt trên nền dương hư",
     "note": "Khát TRẦN + dấu hàn định tính -> CẤM dựng trục Nhiệt. Phải ra Hàn."},
    {"id": "reg_khat_lanh_duonghu",
     "raw": ("34 tuổi, nữ giới, bệnh vài ngày, thích uống nước lạnh, tiểu tiện trong dài, nước "
             "tiểu trong, rêu trắng mỏng, khát nước, lưỡi bệu, tiểu đêm, đau lưng, mỏi gối"),
     "source": "hồi quy #3 — đảo cực: mở lựa chọn 渴喜冷飲 phá tiền đề bản vá #1, 'Hàn' bị XÓA",
     "note": "6 dấu hàn + 1 dấu nhiệt THẬT -> phải ra 'Hàn Nhiệt Thác Tạp', KHÔNG được rụng Hàn."},
    {"id": "vdt_ty_khi_hu",
     "raw": ("bệnh mới mắc, ngực bụng đầy tức, đại tiện lỏng, buồn nôn, đau bụng, mệt mỏi, "
             "nôn mửa, ăn kém, ợ chua, rêu trắng mỏng, lưỡi bệu"),
     "source": "ca mối nối Mục 4 — dấu trệ trung tiêu KHÔNG được lật trục Hư/Thực",
     "note": "72/407 dòng HƯ trong KB sẽ bị lật oan nếu nạp dấu trung tiêu vào thuc_kws."},
]

_RX = {"bat_cuong": r"thuộc chứng[^\n:]*:\s*([^\n]+)",
       "core": r"cốt lõi[^\n:]*:\s*([^\n]+)",
       "benh_danh": r"\*\*bệnh danh:\*\*\s*([^\n]+)"}


def norm(s: str) -> str:
    """Chuẩn hóa giá trị bóc ra. Cắt tiêu đề mục dính cuối dòng ('### 3. ...') — chạy LIVE hay
    OFFLINE cho ra cùng nội dung nhưng khác phần đuôi này, không chuẩn hóa thì golden đỏ oan."""
    s = re.split(r'#{2,}', s or "")[0]
    return re.sub(r'\s+', ' ', s.replace('*', '')).strip(' .')


def extract(answer: str) -> dict:
    low = (answer or "").lower()
    out = {}
    for k, rx in _RX.items():
        m = re.search(rx, low)
        out[k] = norm(m.group(1)) if m else ""
    return out


def make_pipeline():
    """Khởi tạo pipeline và CHẶN LLM bằng đúng một seam."""
    for line in io.open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    from src.fusion_pipeline import TCMFusionPipeline as F
    p = F()

    def _blocked(*a, **k):
        raise RuntimeError("OFFLINE: harness chặn mọi lượt gọi LLM")

    p.qa_pipeline.client.chat = _blocked
    return p


def collect_cases(limit=None):
    """Ca hồi quy + ca lấy từ corpus audit 40 ca (nếu có)."""
    cases = list(REGRESSION_CASES)
    p = os.path.join(ROOT, "data", "audit_sample_report.json")
    if os.path.exists(p):
        for i, c in enumerate(json.load(io.open(p, encoding="utf-8"))):
            raw = (c.get("input") or "").strip()
            if raw:
                cases.append({"id": f"audit{i:02d}_{(c.get('disease') or '')[:18]}".strip("_"),
                              "raw": raw, "source": "data/audit_sample_report.json",
                              "note": ""})
    return cases[:limit] if limit else cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    p = make_pipeline()
    cases = collect_cases(args.limit)
    print(f"Chạy {len(cases)} ca OFFLINE (0 lượt gọi LLM)...\n")
    out, skipped = [], []
    for i, c in enumerate(cases, 1):
        try:
            r = p.run_diagnosis(user_symptoms=c["raw"])
            got = extract((r.get("diagnosis_result") or {}).get("answer") or "")
        except Exception as e:
            skipped.append((c["id"], f"lỗi chạy: {type(e).__name__}"))
            continue
        # Loại ca thoái hóa: thiếu bệnh danh/core -> nhánh suy giảm mềm, KHÔNG phải đường production
        if not got["benh_danh"] or not got["core"] or not got["bat_cuong"]:
            skipped.append((c["id"], "thoái hóa (thiếu bệnh danh/core/bát cương)"))
            continue
        out.append({"id": c["id"], "raw": c["raw"], "source": c["source"], "note": c["note"],
                    "expect": got})
        print(f"  [{i:>2}/{len(cases)}] {c['id'][:34]:<34} {got['bat_cuong'][:42]}")

    data = {"version": 1,
            "_meta": {
                "sinh_boi": "scripts/build_batcuong_golden.py",
                "che_do": "OFFLINE — chặn qa_pipeline.client.chat, 0 lượt gọi LLM",
                "canh_bao": ("Golden ĐÓNG BĂNG hành vi hiện tại. Chốt sai = khóa luôn cái sai. "
                             "Phải đọc tay trước khi commit, và khi test đỏ phải xác định là HỒI "
                             "QUY hay là CẢI THIỆN có chủ đích trước khi sinh lại."),
                "khong_phu": ("Tầng prose Mục 3/4 (văn do LLM sinh) — harness này KHÔNG bắt được "
                              "hồi quy kiểu 'cắt mất biện luận'. Cần fixture replay riêng."),
                "so_ca": len(out), "bo_qua": len(skipped)},
            "cases": out}
    io.open(GOLDEN, "w", encoding="utf-8").write(
        json.dumps(data, ensure_ascii=False, indent=1))
    print(f"\nĐã ghi {len(out)} ca -> data/batcuong_golden.json")
    if skipped:
        print(f"Bỏ qua {len(skipped)} ca:")
        for cid, why in skipped[:10]:
            print(f"   {cid[:38]:<38} {why}")


if __name__ == "__main__":
    main()
