#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/test_muc5_general_fallback.py — Khóa ngoại lệ TƯ ÂM TIỀM DƯƠNG khi Mục 5 mượn thể tổng quát
(_yin_def_yang_rise_general).

VÌ SAO (audit ổn định 40 ca — ca Mục 5 TRẮNG duy nhất mà bệnh danh vẫn ĐÚNG):
    'Kinh hành đầu thống' × cốt lõi 'Âm hư dương cang'  ->  "chưa có bài thuốc đặc trị"
trong khi KB CÓ sẵn 'Kinh hành đầu thống × Âm hư -> Thanh huyễn bình can thang' — mà đó CHÍNH LÀ
bài tư âm tiềm dương (Sinh địa, Nữ trinh tử, Hạn liên thảo + Tang diệp, Cúc hoa, Ngưu tất).

CƠ CHẾ: tầng "mượn thể tổng quát" đòi token DƯ của cốt lõi chỉ được là từ ĐỊNH VỊ TẠNG. Token dư
ở đây là {dương, cang} — chạm tập token bệnh lý -> chặn.

CÁI CHỐT ĐÓ ĐÚNG và PHẢI GIỮ: nó chặn 'Khí hư huyết trệ' mượn bài 'Khí hư' (mượn xong BỎ RƠI
huyết trệ — một tà thực cần trị riêng). Ngoại lệ chỉ đúng cho DƯƠNG CANG vì dương cang KHÔNG phải
tà ngoại lai mà là HỆ QUẢ của chính âm hư; tư âm thì dương tự tiềm, không bỏ sót gì.

PHẠM VI ĐÃ ĐO trên KB: đúng 2 hội chứng cốt lõi ('Âm hư dương cang', 'Âm hư dương thịnh') × 19
bệnh có dòng 'Âm hư'. 'Âm hư hỏa vượng' KHÔNG được mở (hỏa là tà) — nhóm (2) khóa điều này.

Chạy:  PYTHONIOENCODING=utf-8 python scripts/test_muc5_general_fallback.py   (exit != 0 nếu fail)
Không cần Neo4j/LLM.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.fusion_pipeline import TCMFusionPipeline as F  # noqa: E402


def _toks(s):
    return set(re.findall(r'[^\W\d_]+', (s or "").lower()))


def _chk(label, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return bool(cond)


def test_cho_muon():
    print("== (1) CHO mượn: token dư là DƯƠNG BỐC (hệ quả của âm hư) ==")
    n = f = 0
    for core, hc in (("Âm hư dương cang", "Âm hư"),
                     ("Âm hư dương thịnh", "Âm hư"),
                     ("Âm hư dương vượng", "Âm hư"),
                     ("Can âm hư dương thượng cang", "Can âm hư")):
        ok = F._yin_def_yang_rise_general(_toks(core), _toks(hc))
        n += _chk(f"'{core}' mượn được '{hc}'", ok)
        f += (not ok)
    return n, f


def test_khong_cho_muon():
    """Ranh giới — nếu nhóm này lọt, hệ sẽ kê bài BỎ RƠI tà thực."""
    print("\n== (2) KHÔNG cho mượn ==")
    n = f = 0
    for core, hc, why in (
        ("Âm hư hỏa vượng", "Âm hư", "'hỏa' là TÀ, không phải hệ quả thuần của âm hư"),
        ("Âm hư thấp nhiệt", "Âm hư", "thấp+nhiệt là tà thực, bỏ rơi thì trị thiếu"),
        ("Âm hư ứ huyết", "Âm hư", "ứ huyết cần hoạt huyết riêng"),
        ("Khí hư huyết trệ", "Khí hư", "chính là ca mà cái chốt gốc sinh ra để chặn"),
        ("Khí hư đàm trệ", "Khí hư", "đàm trệ là tà thực"),
        ("Can dương thượng cang", "Can", "không có âm hư ở cả hai vế"),
        ("Âm hư", "Âm hư", "trùng khít, không có token dư -> không phải ca mượn"),
    ):
        ok = not F._yin_def_yang_rise_general(_toks(core), _toks(hc))
        n += _chk(f"chặn '{core}' <- '{hc}' ({why})", ok)
        f += (not ok)
    return n, f


def test_pham_vi_tren_kb():
    """Bán kính ảnh hưởng phải còn NHỎ. Nếu ai đó nới _YANG_RISE_TOKS, số này phình lên và test kêu."""
    print("\n== (3) Bán kính trên KB thật ==")
    import csv
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "Medicine_clean.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    kcol = list(rows[0].keys())
    allhc = {(r.get(kcol[1]) or "").strip() for r in rows if (r.get(kcol[1]) or "").strip()}
    cores = [h for h in allhc if F._yin_def_yang_rise_general(_toks(h), {"âm", "hư"})]
    n = f = 0
    ok = len(cores) <= 4
    n += _chk(f"số hội chứng cốt lõi hưởng luật <= 4 (đang {len(cores)})", ok, str(sorted(cores)))
    f += (not ok)
    # không được đụng tới nhánh khí hư
    bad = [h for h in allhc if F._yin_def_yang_rise_general(_toks(h), {"khí", "hư"})]
    ok = not bad
    n += _chk("không hội chứng nào mượn được thể 'Khí hư' qua luật này", ok, str(bad))
    f += (not ok)
    return n, f


def main():
    tp = tf = 0
    for fn in (test_cho_muon, test_khong_cho_muon, test_pham_vi_tren_kb):
        p, q = fn()
        tp += p
        tf += q
    print(f"\n{tp} PASS, {tf} FAIL / {tp + tf}")
    sys.exit(1 if tf else 0)


if __name__ == "__main__":
    main()
