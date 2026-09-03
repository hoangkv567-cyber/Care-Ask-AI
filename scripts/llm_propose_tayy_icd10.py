#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM đề xuất mã ICD-10 cho TayY từ top-5 candidate BM25 của map_tayy_icd10.py.

Vai trò trong pipeline: mapper deterministic chỉ cấp được 45 mã Tier A
(exact-match kép); 7.5k dòng còn lại phải chọn tay từ top-5 candidate. Script này
để LLM làm lượt chọn ĐẦU TIÊN — nhiều vote độc lập, chỉ nhận khi đồng thuận —
và ghi sidecar ``data/icd10/tayy_icd10_llm_proposals.jsonl``. Mã LLM là "máy đề
xuất": apply_tayy_icd10.py materialize nó ở bậc ưu tiên DƯỚI Tier A, trạng thái
vẫn ``chua_kiem_duyet``, và chuyên gia luôn có quyền phủ quyết qua review sidecar.

KHÔNG sửa TayY_clean.csv, KHÔNG sửa review_queue 56MB (hints ghi file CSV riêng).
Mọi record mang row_sha256 + catalog_sha256 — stale là bị loader từ chối.

Chạy (từ repo root; mặc định resume; --limit để thăm dò trước khi chạy full)::

    python scripts/llm_propose_tayy_icd10.py --dry-run
    python scripts/llm_propose_tayy_icd10.py --priority P2_validate_tier_a
    python scripts/llm_propose_tayy_icd10.py --priority P1_manual_mapping --workers 4
    python scripts/llm_propose_tayy_icd10.py --priority P0_duplicate_name \
        --adjudication data/tayy_dup_adjudication.jsonl
    python scripts/llm_propose_tayy_icd10.py --emit-hints
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from llm_batch_common import (  # noqa: E402
    DEFAULT_DASHSCOPE_MODEL,
    DEFAULT_REQUESTY_MODEL,
    BatchLLMError,
    FallbackChat,
    append_jsonl,
    build_fallback_chain,
    load_done,
    load_env,
    utc_now_iso,
)
from map_tayy_icd10 import _proposal_priority  # noqa: E402
from review_tayy_icd10 import (  # noqa: E402
    DEFAULT_CATALOG,
    DEFAULT_CATALOG_META,
    DEFAULT_CLEAN,
    DEFAULT_PROPOSALS,
    CatalogData,
    ContractError,
    _iter_json_records,
    load_catalog,
    load_clean,
)

ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = ROOT / "data" / "icd10" / "tayy_icd10_llm_proposals.jsonl"
DEFAULT_HINTS = ROOT / "data" / "icd10" / "tayy_icd10_llm_hints.csv"
DEFAULT_REPORT = ROOT / "data" / "icd10" / "tayy_icd10_llm_report.json"
DEFAULT_ADJUDICATION = ROOT / "data" / "tayy_dup_adjudication.jsonl"

SCHEMA_VERSION = "tayy-icd10-llm-proposal-v1"
VERDICTS = frozenset(("pick", "none_correct", "outside_top5", "insufficient"))
ALL_PRIORITIES = (
    "P2_validate_tier_a",
    "P1_manual_mapping",
    "P1_low_information",
    "P1_non_disease_or_tcm",
    "P0_duplicate_name",
    "P0_exact_conflict",
)
# P0_exact_conflict (1 dòng) cố ý KHÔNG có trong mặc định: người xử tay.
DEFAULT_PRIORITIES = (
    "P2_validate_tier_a",
    "P1_manual_mapping",
    "P1_low_information",
    "P1_non_disease_or_tcm",
)

SYSTEM_PROMPT = (
    "Bạn là chuyên gia mã hóa bệnh tật ICD-10 (bản tiếng Việt theo Thông tư 06/2026/TT-BYT). "
    "Chỉ trả về đúng một JSON object, không markdown, không văn bản ngoài JSON."
)

USER_TEMPLATE = """Chọn mã ICD-10 morbidity ĐƠN LẺ đúng nhất cho bản ghi bệnh (nguồn dịch máy, có thể nhiễu).

BẢN GHI:
{evidence}
{dup_context}
5 ỨNG VIÊN (từ hệ truy xuất, chưa chắc đúng):
{candidates}

LUẬT:
- Một ứng viên đúng là bệnh này -> verdict "pick" + đúng code của ứng viên đó.
- Chắc chắn mã đúng KHÔNG nằm trong 5 ứng viên -> verdict "outside_top5" + code ICD-10
  đầy đủ (định dạng như A04.3); chỉ dùng khi rất tự tin.
- Bản ghi không đủ thông tin để định bệnh -> "insufficient".
- Bản ghi không phải MỘT bệnh gán được đúng MỘT mã (đề mục chung, hội chứng mơ hồ,
  nội dung không phải bệnh) -> "none_correct".
- Không chọn mã bị đánh dấu eligible=False; ưu tiên mã cụ thể (4 ký tự) hơn mã .9 chung.

Trả về JSON: {{"verdict": "pick|none_correct|outside_top5|insufficient",
  "code": "A04.3 hoặc null", "reason": "ngắn gọn, <=30 từ"}}"""


# ---------------------------------------------------------------- prompt build --
def _clip(text: Any, cap: int) -> str:
    return str(text or "").strip().replace("\n", " ")[:cap]


def _evidence_block(record: dict[str, Any]) -> str:
    evidence = record.get("evidence") or {}
    components = record.get("components") or {}
    lines = [f"- Tên bệnh: {_clip(record.get('name'), 160)}"]
    lead = _clip((components.get("lead") or {}).get("raw"), 200)
    if lead:
        lines.append(f"- Ý mô tả đầu: {lead}")
    desc = _clip(evidence.get("description"), 400)
    if desc:
        lines.append(f"- Mô tả: {desc}")
    symptoms = [_clip(s, 60) for s in (evidence.get("symptoms") or [])[:12]]
    if symptoms:
        lines.append(f"- Triệu chứng: {', '.join(symptoms)}")
    tests = [_clip(t, 60) for t in (evidence.get("tests") or [])[:6]]
    if tests:
        lines.append(f"- Xét nghiệm: {', '.join(tests)}")
    department = _clip(evidence.get("department"), 120)
    if department:
        lines.append(f"- Khoa điều trị: {department}")
    return "\n".join(lines)


def _candidates_block(record: dict[str, Any]) -> str:
    lines = []
    for candidate in record.get("candidates") or []:
        eligible = bool(
            ((candidate.get("eligibility") or {}).get("auto_single_code_eligible"))
        )
        lines.append(
            f"{candidate.get('rank')}. {candidate.get('code')} — {_clip(candidate.get('title_vi'), 160)}"
            f" | EN: {_clip(candidate.get('title_en'), 120)}"
            f" | Nhóm: {_clip(candidate.get('category_vi'), 100)} | eligible={eligible}"
        )
        guidance = _clip(candidate.get("guidance_vi"), 200)
        if guidance:
            lines.append(f"   Hướng dẫn: {guidance}")
    return "\n".join(lines)


def _dup_context_block(disease_id: str, adjudication: dict[str, dict] | None) -> str:
    if not adjudication or disease_id not in adjudication:
        return ""
    info = adjudication[disease_id]
    lines = ["NGỮ CẢNH TRÙNG TÊN (đã phân xử bằng máy, tham khảo):"]
    if info.get("distinct_name"):
        lines.append(f"- Bản ghi này thuộc cụm bệnh: {info['distinct_name']!r}")
    peers = [p for p in info.get("cluster_ids", []) if p != disease_id]
    if peers:
        lines.append(f"- Cùng cụm (cùng bệnh): {', '.join(peers)}")
    others = info.get("other_cluster_names") or []
    if others:
        lines.append(
            "- Các cụm KHÁC BỆNH nhưng trùng tên: " + "; ".join(repr(n) for n in others if n)
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- adjudication --
def load_adjudication_index(
    path: Path, row_hashes: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """disease_id -> {cluster_ids, distinct_name, other_cluster_names}.

    Chỉ nhận record hash còn khớp và consensus != no_consensus; record stale bị
    bỏ qua kèm cảnh báo (sidecar là gợi ý, không phải phán quyết)."""
    index: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise ContractError(f"Không thấy adjudication sidecar: {path}")
    stale = 0
    for _, record in _iter_json_records(path, "adjudication sidecar"):
        if record.get("schema_version") != "tayy-dup-adjudication-v1":
            continue
        hashes = record.get("row_sha256") or {}
        ids = record.get("disease_ids") or []
        if not ids or any(row_hashes.get(i) != hashes.get(i) for i in ids):
            stale += 1
            continue
        consensus = record.get("consensus") or {}
        if consensus.get("verdict") in (None, "no_consensus"):
            continue
        clusters = consensus.get("clusters") or []
        names = [str(c.get("distinct_name") or "").strip() for c in clusters]
        for cluster_index, cluster in enumerate(clusters):
            cluster_ids = [str(x) for x in (cluster.get("ids") or [])]
            other_names = [n for i, n in enumerate(names) if i != cluster_index and n]
            for disease_id in cluster_ids:
                index[disease_id] = {
                    "cluster_ids": cluster_ids,
                    "distinct_name": names[cluster_index],
                    "other_cluster_names": other_names,
                }
    if stale:
        print(f"[CẢNH BÁO] {stale} record adjudication stale (hash lệch) — bỏ qua")
    return index


# --------------------------------------------------------------------- voting --
def _validate_vote(
    payload: dict, candidate_codes: set[str], catalog: CatalogData
) -> dict[str, Any]:
    verdict = str(payload.get("verdict") or "").strip()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} không hợp lệ")
    raw_code = payload.get("code")
    code = catalog.canonical_code(raw_code) if raw_code not in (None, "", "null") else ""
    vote: dict[str, Any] = {"reason": _clip(payload.get("reason"), 240)}
    if verdict in ("pick", "outside_top5"):
        if not code:
            raise ValueError(f"verdict={verdict} nhưng code không hợp lệ: {raw_code!r}")
        record = catalog.records[code]
        if record.get("auto_single_code_eligible") is not True:
            # Mã tồn tại nhưng catalog CẤM gán đơn lẻ (dagger/asterisk/cần chi tiết
            # hơn — vd A18.0 lao xương khớp cần mã kép). Máy thường ĐÚNG về bệnh,
            # sai về loại mã -> giữ làm bằng chứng cho reviewer thay vì vứt phiếu.
            vote.update({"verdict": "ineligible_code", "code": "", "ma_muon": code})
            return vote
        if verdict == "pick" and code not in candidate_codes:
            verdict = "outside_top5"      # LLM nói 'pick' nhưng mã ngoài bảng
    else:
        code = ""
    vote.update({"verdict": verdict, "code": code})
    return vote


def _outcome_key(vote: dict[str, Any]) -> str:
    if vote.get("verdict") == "ineligible_code":
        return f"inel:{vote.get('ma_muon')}"      # đồng thuận theo đúng mã máy muốn gán
    return vote["code"] if vote["code"] else vote["verdict"]


def propose_one(
    record: dict[str, Any],
    priority: str,
    catalog: CatalogData,
    voters: list[tuple[BatchChat, int]],
    tiebreaker: tuple[BatchChat, int],
    adjudication: dict[str, dict] | None,
) -> dict[str, Any]:
    disease_id = record["disease_id"]
    candidate_codes = {
        catalog.canonical_code(c.get("code")) for c in (record.get("candidates") or [])
    }
    candidate_codes.discard("")
    user_prompt = USER_TEMPLATE.format(
        evidence=_evidence_block(record),
        dup_context=_dup_context_block(disease_id, adjudication),
        candidates=_candidates_block(record),
    )

    votes: list[dict[str, Any]] = []

    def run_vote(chat: BatchChat, seed: int) -> dict[str, Any] | None:
        try:
            payload = chat.chat_json(SYSTEM_PROMPT, user_prompt, seed=seed)
            vote = _validate_vote(payload, candidate_codes, catalog)
        except (BatchLLMError, ValueError) as exc:
            votes.append(
                {"provider": chat.provider, "model": chat.model, "seed": seed,
                 "error": str(exc)[:300]}
            )
            return None
        vote.update({"provider": chat.provider, "model": chat.model, "seed": seed})
        votes.append(vote)
        return vote

    valid = [v for chat, seed in voters if (v := run_vote(chat, seed))]
    if not (len(valid) == 2 and _outcome_key(valid[0]) == _outcome_key(valid[1])):
        tb = run_vote(*tiebreaker)
        if tb:
            valid.append(tb)

    tally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in valid:
        tally[_outcome_key(vote)].append(vote)
    winners = max(tally.values(), key=len, default=[])

    if len(winners) >= 2:
        code = winners[0]["code"]
        if code:
            kind = "in_top5" if code in candidate_codes else "outside_top5"
        elif winners[0]["verdict"] == "ineligible_code":
            kind = "ineligible_code"      # bệnh cần mã kép/chi tiết hơn — chuyên gia phán
        else:
            kind = "none" if winners[0]["verdict"] == "none_correct" else "insufficient"
        consensus = {
            "kind": kind,
            "code": code or None,
            "agreement": f"{len(winners)}/{len(valid)}",
        }
        if kind == "ineligible_code":
            consensus["ma_de_xuat_khong_du_dieu_kien"] = winners[0].get("ma_muon")
    else:
        consensus = {"kind": "no_consensus", "code": None, "agreement": f"1/{len(valid)}" if valid else "0"}

    # Đồng thuận LLM trùng mã exact đơn-tín-hiệu của mapper -> reviewer duyệt nhanh.
    exact = (record.get("components") or {}).get("exact") or {}
    name_codes = exact.get("name_codes") or []
    lead_codes = exact.get("lead_codes") or []
    exact_signal_agreement = bool(
        consensus["code"]
        and (
            (len(name_codes) == 1 and consensus["code"] == name_codes[0])
            or (len(lead_codes) == 1 and consensus["code"] == lead_codes[0])
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "disease_id": disease_id,
        "name": record.get("name") or "",
        "row_sha256": record.get("row_sha256") or record.get("row_fingerprint") or "",
        "catalog_sha256": catalog.sha256,
        "base_mapper_version": str(
            (record.get("provenance") or {}).get("mapper_version") or ""
        ),
        "priority": priority,
        "votes": votes,
        "consensus": consensus,
        "exact_signal_agreement": exact_signal_agreement,
        "created_at": utc_now_iso(),
    }


# ---------------------------------------------------------------------- hints --
def emit_hints(output: Path, hints_path: Path) -> None:
    records = load_done(output, "disease_id")
    hints_path.parent.mkdir(parents=True, exist_ok=True)
    with hints_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["disease_id", "tên_bệnh", "llm_code", "kind", "agreement",
             "exact_signal_agreement", "goi_y_decision", "reason"]
        )
        suggested = {
            "none": "unmappable_no_single_code",
            "insufficient": "needs_more_information",
            "ineligible_code": "unmappable_no_single_code",
        }
        for disease_id in sorted(records):
            record = records[disease_id]
            consensus = record.get("consensus") or {}
            reasons = [
                v.get("reason", "") for v in record.get("votes", []) if v.get("reason")
            ]
            llm_code = consensus.get("code") or ""
            if not llm_code and consensus.get("ma_de_xuat_khong_du_dieu_kien"):
                llm_code = f"({consensus['ma_de_xuat_khong_du_dieu_kien']} — cần mã kép/chi tiết hơn)"
            writer.writerow(
                [
                    disease_id,
                    record.get("name", ""),
                    llm_code,
                    consensus.get("kind") or "",
                    consensus.get("agreement") or "",
                    "yes" if record.get("exact_signal_agreement") else "",
                    suggested.get(consensus.get("kind"), ""),
                    (reasons[0] if reasons else "")[:200],
                ]
            )
    print(f"[OK] Hints cho reviewer: {hints_path} ({len(records)} dòng)")


# ----------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM chọn mã ICD-10 từ top-5 candidate -> sidecar máy đề xuất (không sửa CSV)."
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--catalog-meta", type=Path, default=DEFAULT_CATALOG_META)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--priority",
        action="append",
        choices=ALL_PRIORITIES,
        default=None,
        help=f"Lặp cờ để chọn nhiều; mặc định: {', '.join(DEFAULT_PRIORITIES)}",
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=None,
        help="Sidecar phân xử trùng tên; BẮT BUỘC khi chạy P0_duplicate_name",
    )
    parser.add_argument("--limit", type=int, default=0, help="Chỉ xử lý N dòng mới (0 = tất cả)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Chỉ đếm khối lượng, không gọi LLM")
    parser.add_argument("--emit-hints", action="store_true", help="Chỉ xuất hints CSV từ sidecar rồi thoát")
    parser.add_argument("--hints", type=Path, default=DEFAULT_HINTS)
    parser.add_argument(
        "--vote-provider",
        choices=("requesty", "dashscope"),
        default="requesty",
        help="Provider cho 2 phiếu chính (Requesty 402 hết số dư -> dùng dashscope)",
    )
    parser.add_argument(
        "--vote-model",
        default=DEFAULT_REQUESTY_MODEL,
        help="Model cho 2 phiếu chính; 'A,B' để phiếu 1 dùng A, phiếu 2 dùng B (đa dạng hơn)",
    )
    parser.add_argument("--ds-model", default=DEFAULT_DASHSCOPE_MODEL,
                        help="Model tie-break (luôn DashScope)")
    args = parser.parse_args(argv)

    if args.emit_hints:
        emit_hints(args.output, args.hints)
        return 0

    load_env(ROOT)
    clean = load_clean(args.clean)
    catalog = load_catalog(args.catalog, args.catalog_meta)

    priorities = tuple(args.priority) if args.priority else DEFAULT_PRIORITIES
    adjudication: dict[str, dict] | None = None
    if "P0_duplicate_name" in priorities:
        if not args.adjudication:
            raise ContractError(
                "Chạy P0_duplicate_name bắt buộc có --adjudication "
                "data/tayy_dup_adjudication.jsonl (sinh bởi llm_adjudicate_tayy_dups.py)"
            )
        adjudication = load_adjudication_index(args.adjudication, clean.row_hashes)
        print(f"Adjudication nạp: {len(adjudication)} dòng thuộc cụm đã phân xử")

    done = load_done(args.output, "disease_id")

    def is_fresh(record: dict, row_hash: str) -> bool:
        if (
            record.get("row_sha256") != row_hash
            or record.get("catalog_sha256") != catalog.sha256
        ):
            return False
        # no_consensus do PHIẾU LỖI (mạng/quota — <2 phiếu hợp lệ) không phải kết quả
        # thật -> coi như chưa chạy để lần sau retry; bất đồng THẬT (>=2 phiếu hợp lệ
        # nhưng không có phe đa số) thì giữ, dành cho người duyệt.
        consensus = record.get("consensus") or {}
        if consensus.get("kind") != "no_consensus":
            return True
        valid_votes = sum(1 for v in record.get("votes") or [] if "verdict" in v)
        return valid_votes >= 2

    # Quét streaming: chọn record cần chạy theo priority, hash còn khớp, chưa có
    # kết quả fresh. KHÔNG giữ toàn bộ 108MB trong RAM — chỉ giữ record được chọn.
    pending: list[tuple[str, dict[str, Any]]] = []
    stats: Counter = Counter()
    for _, record in _iter_json_records(args.proposals, "proposal sidecar"):
        disease_id = str(record.get("disease_id") or "")
        row_hash = str(record.get("row_sha256") or record.get("row_fingerprint") or "")
        if disease_id not in clean.by_id or row_hash != clean.row_hashes.get(disease_id):
            stats["stale_row"] += 1
            continue
        prov_catalog = str((record.get("provenance") or {}).get("catalog_sha256") or "")
        if prov_catalog and prov_catalog != catalog.sha256:
            stats["stale_catalog"] += 1
            continue
        priority = _proposal_priority(record)
        stats[f"priority::{priority}"] += 1
        if priority not in priorities:
            continue
        # exact-conflict (kể cả khi ưu tiên bị nhãn P0_duplicate_name nuốt): danh
        # tính nhập nhằng, người xử tay — không hỏi LLM, loader cũng không nhận mã.
        if ((record.get("components") or {}).get("exact") or {}).get("conflict"):
            stats["exact_conflict_bo_qua"] += 1
            continue
        if priority == "P0_duplicate_name" and (
            adjudication is None or disease_id not in adjudication
        ):
            stats["dup_khong_co_phan_xu"] += 1
            continue
        if disease_id in done and is_fresh(done[disease_id], row_hash):
            stats["da_co_ket_qua"] += 1
            continue
        pending.append((priority, record))

    print(f"Ưu tiên chạy: {', '.join(priorities)}")
    for key in sorted(stats):
        print(f"  {key:<32} {stats[key]}")
    print(f"Cần chạy: {len(pending)} dòng")
    if args.limit > 0:
        pending = pending[: args.limit]
        print(f"--limit: chạy {len(pending)} dòng lần này")
    if args.dry_run or not pending:
        if args.dry_run:
            print("[DRY-RUN] Không gọi LLM.")
        return 0

    # Mỗi worker một bộ client riêng (httpx.Client không chia sẻ giữa thread khi
    # đang stream); vote structure: 2 phiếu Requesty + tie-break DashScope.
    local = threading.local()

    def get_voters() -> tuple[list[tuple[BatchChat, int]], tuple[BatchChat, int]]:
        if not hasattr(local, "voters"):
            models = [m.strip() for m in args.vote_model.split(",") if m.strip()]
            if len(models) == 1:
                models = models * 2
            # Mỗi phiếu là một CHUỖI dự phòng: model cạn quota (402/403) -> tự
            # chuyển model kế; record vote ghi model THẬT đã dùng.
            local.voters = [
                (FallbackChat(build_fallback_chain(args.vote_provider, models[0]),
                              temperature=0.15), 42),
                (FallbackChat(build_fallback_chain(args.vote_provider, models[1]),
                              temperature=0.35), 1337),
            ]
            local.tiebreaker = (
                FallbackChat(build_fallback_chain("dashscope", args.ds_model),
                             temperature=0.2), 7)
        return local.voters, local.tiebreaker

    run_stats: Counter = Counter()
    lock = threading.Lock()

    def task(item: tuple[str, dict[str, Any]]) -> str:
        priority, record = item
        voters, tiebreaker = get_voters()
        result = propose_one(record, priority, catalog, voters, tiebreaker, adjudication)
        append_jsonl(args.output, result)
        with lock:
            run_stats[f"kind::{result['consensus']['kind']}"] += 1
        return f"{record['disease_id']} -> {result['consensus']['kind']}" + (
            f" {result['consensus']['code']}" if result["consensus"]["code"] else ""
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, item) for item in pending]
        for future in as_completed(futures):
            completed += 1
            try:
                message = future.result()
            except Exception as exc:      # noqa: BLE001 — 1 dòng hỏng không giết cả batch
                message = f"[LỖI] {exc}"
                with lock:
                    run_stats["task_error"] += 1
            if completed % 25 == 0 or completed == len(pending):
                print(f"[{completed}/{len(pending)}] {message}")

    # Report tổng trên toàn bộ sidecar fresh.
    all_records = load_done(args.output, "disease_id")
    fresh = {
        k: r for k, r in all_records.items()
        if k in clean.row_hashes and is_fresh(r, clean.row_hashes[k])
    }
    kinds = Counter((r.get("consensus") or {}).get("kind") for r in fresh.values())
    exact_agree = sum(1 for r in fresh.values() if r.get("exact_signal_agreement"))
    report = {
        "schema_version": SCHEMA_VERSION,
        "tổng_record_fresh": len(fresh),
        "consensus_kind": dict(sorted(kinds.items())),
        "exact_signal_agreement": exact_agree,
        "chạy_lần_này": dict(sorted(run_stats.items())),
        "ghi_chú": [
            "Mã LLM là máy đề xuất — materialize qua apply_tayy_icd10.py ở bậc dưới "
            "Tier A, trạng thái vẫn chua_kiem_duyet.",
            "kind=none/insufficient/no_consensus KHÔNG bao giờ đổ mã vào CSV.",
        ],
        "cập_nhật": utc_now_iso(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {args.output} ({len(fresh)} record fresh)")
    print(f"[OK] {args.report}: {dict(sorted(kinds.items()))}")
    print("Tiếp theo: python scripts/llm_propose_tayy_icd10.py --emit-hints  # hints cho reviewer")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[LỖI CONTRACT] {exc}", file=sys.stderr)
        raise SystemExit(1)
