#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM phân xử 958 dòng trùng tên (cờ ``trung_ten_khac_noi_dung``) trong TayY_clean.csv.

Dịch máy phá tên bệnh nên "trùng tên" có thể là (a) 2 bản dịch cùng một bệnh,
(b) các bệnh KHÁC nhau chung tên hỏng, hoặc (c) nhóm mixed ('Ung thư' = 11 bệnh).
Script hỏi LLM PHÂN HOẠCH từng nhóm thành các cụm cùng-một-bệnh, vote đồng thuận,
ghi sidecar gợi ý — TUYỆT ĐỐI KHÔNG sửa TayY_clean.csv (disease_id khóa theo vị
trí dòng và row_sha256 phủ mọi cột nội dung; sửa CSV là vô hiệu 108MB proposal
ICD + mọi review chuyên gia — xem map_tayy_icd10.py:539-546, review_tayy_icd10.py:112-135).

Consumer của sidecar:
* scripts/llm_propose_tayy_icd10.py — bơm ngữ cảnh cụm khi đề xuất mã ICD cho
  dòng P0_duplicate_name;
* scripts/build_tayy_reference_index.py — ẩn dòng thua trong cụm cùng-bệnh,
  hiện tên phân biệt máy đề xuất cho cụm khác-bệnh.

Vote: 2 lượt Requesty qwen3-30b (temperature/seed khác nhau); lệch thì lượt 3
DashScope qwen3-max phân xử; không đạt >=2 phiếu cùng phân hoạch -> no_consensus
(để người xử). Mọi record mang row_sha256 từng thành viên — consumer đối chiếu,
lệch là bỏ qua record (dữ liệu gợi ý, không phải phán quyết chuyên gia).

Chạy (từ repo root; mặc định resume, bỏ nhóm đã có record khớp hash)::

    python scripts/llm_adjudicate_tayy_dups.py --limit 5     # thăm dò
    python scripts/llm_adjudicate_tayy_dups.py               # full 423 nhóm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from review_tayy_icd10 import (  # noqa: E402
    ContractError,
    load_clean,
)

ROOT = SCRIPT_DIR.parent
DEFAULT_CLEAN = ROOT / "data" / "TayY_clean.csv"
DEFAULT_OUTPUT = ROOT / "data" / "tayy_dup_adjudication.jsonl"
DEFAULT_REPORT = ROOT / "data" / "tayy_dup_adjudication_report.json"

SCHEMA_VERSION = "tayy-dup-adjudication-v1"
DUPLICATE_NAME_FLAG = "trung_ten_khac_noi_dung"

SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân loại bệnh học, phân xử dữ liệu bệnh Tây y bị dịch máy "
    "làm hỏng tên. Chỉ trả về đúng một JSON object, không markdown, không giải thích ngoài JSON."
)

USER_TEMPLATE = """Các bản ghi dưới đây CÙNG TÊN BỆNH "{name}" nhưng có thể là những bệnh khác nhau
bị dịch máy phá tên, hoặc nhiều bản dịch của cùng một bệnh.

Nhiệm vụ: PHÂN HOẠCH toàn bộ bản ghi thành các cụm, mỗi cụm = đúng một bệnh thực tế.
- Mỗi bản ghi phải thuộc đúng một cụm (không thiếu, không thừa, không trùng).
- Cụm có >=2 bản ghi: chọn "keep" = id bản ghi có nội dung đầy đủ/chính xác nhất.
- Nếu có >1 cụm: đặt "distinct_name" tiếng Việt phân biệt cho TỪNG cụm (ngắn gọn,
  đúng thuật ngữ y khoa thông dụng, không bịa; ví dụ 'Ung thư' tách thành
  'Ung thư phổi', 'Ung thư dạ dày'...). Nếu chỉ có 1 cụm, "distinct_name" để "".
- "confidence": "high" nếu chắc chắn, "low" nếu dữ liệu mơ hồ.

BẢN GHI:
{members}

Trả về JSON đúng schema:
{{"clusters": [{{"ids": ["TAYY-00001"], "keep": "TAYY-00001", "distinct_name": ""}}],
  "confidence": "high"}}"""


def _parse_list(raw: str, cap: int) -> list[str]:
    try:
        value = json.loads(raw or "[]")
        if isinstance(value, list):
            return [str(x) for x in value[:cap]]
    except json.JSONDecodeError:
        pass
    return []


def _member_block(row: dict[str, str]) -> str:
    symptoms = _parse_list(row.get("triệu_chứng") or "", 12)
    lines = [f"- id: {row['disease_id']}"]
    desc = (row.get("mô_tả_bệnh") or "").strip().replace("\n", " ")
    if desc:
        lines.append(f"  mô tả: {desc[:600]}")
    if symptoms:
        lines.append(f"  triệu chứng: {', '.join(symptoms)}")
    khoa = (row.get("khoa_điều_trị") or "").strip()
    if khoa:
        lines.append(f"  khoa: {khoa[:120]}")
    cause = (row.get("nguyên_nhân") or "").strip().replace("\n", " ")
    if cause:
        lines.append(f"  nguyên nhân: {cause[:300]}")
    return "\n".join(lines)


def _validate_vote(payload: dict, member_ids: set[str]) -> list[dict[str, Any]]:
    """Phân hoạch phải phủ đúng tập id; keep thuộc cụm. Sai -> ValueError."""
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("clusters phải là list")
    # qwen-plus hay chép thêm cụm rỗng từ ví dụ schema -> lọc trước khi validate.
    clusters = [c for c in clusters if isinstance(c, dict) and c.get("ids")]
    if not clusters:
        raise ValueError("clusters phải là list khác rỗng")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise ValueError("mỗi cluster phải là object")
        ids = cluster.get("ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("cluster.ids phải là list khác rỗng")
        ids = [str(x).strip() for x in ids]
        for disease_id in ids:
            if disease_id not in member_ids:
                raise ValueError(f"id lạ trong cluster: {disease_id}")
            if disease_id in seen:
                raise ValueError(f"id xuất hiện 2 lần: {disease_id}")
            seen.add(disease_id)
        keep = str(cluster.get("keep") or "").strip()
        if len(ids) >= 2:
            if keep not in ids:
                raise ValueError(f"keep {keep!r} không thuộc cụm {ids}")
        else:
            keep = ids[0]
        normalized.append(
            {
                "ids": sorted(ids),
                "keep": keep,
                "distinct_name": str(cluster.get("distinct_name") or "").strip(),
            }
        )
    if seen != member_ids:
        missing = sorted(member_ids - seen)
        raise ValueError(f"phân hoạch thiếu id: {missing}")
    return normalized


def _partition_key(clusters: list[dict[str, Any]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(c["ids"]) for c in clusters)


def _verdict(clusters: list[dict[str, Any]], n_members: int) -> str:
    if len(clusters) == 1:
        return "same_disease"
    if len(clusters) == n_members:
        return "different_disease"
    return "mixed"


def adjudicate_group(
    name: str,
    rows: list[dict[str, str]],
    voters: list[tuple[BatchChat, int]],
    tiebreaker: tuple[BatchChat, int],
    max_votes: int,
) -> dict[str, Any]:
    member_ids = {row["disease_id"] for row in rows}
    user_prompt = USER_TEMPLATE.format(
        name=name, members="\n".join(_member_block(row) for row in rows)
    )

    votes: list[dict[str, Any]] = []

    def run_vote(chat: BatchChat, seed: int) -> dict[str, Any] | None:
        try:
            payload = chat.chat_json(SYSTEM_PROMPT, user_prompt, seed=seed)
            clusters = _validate_vote(payload, member_ids)
        except (BatchLLMError, ValueError) as exc:
            votes.append(
                {
                    "provider": chat.provider,
                    "model": chat.model,
                    "seed": seed,
                    "error": str(exc)[:300],
                }
            )
            return None
        vote = {
            "provider": chat.provider,
            "model": chat.model,
            "seed": seed,
            "clusters": clusters,
            "confidence": "high" if payload.get("confidence") == "high" else "low",
        }
        votes.append(vote)
        return vote

    valid: list[dict[str, Any]] = []
    for chat, seed in voters[:max_votes]:
        vote = run_vote(chat, seed)
        if vote:
            valid.append(vote)

    consensus: dict[str, Any]
    if max_votes == 1:
        # Chế độ thăm dò: nhận 1 phiếu duy nhất (không dùng cho chạy thật).
        winners = valid[:1]
    else:
        keys = [_partition_key(v["clusters"]) for v in valid]
        if len(valid) == 2 and keys[0] == keys[1]:
            winners = valid
        else:
            tb_vote = run_vote(*tiebreaker)
            if tb_vote:
                valid.append(tb_vote)
            tally: dict[frozenset[frozenset[str]], list[dict[str, Any]]] = defaultdict(list)
            for vote in valid:
                tally[_partition_key(vote["clusters"])].append(vote)
            winners = max(tally.values(), key=len, default=[])
            if len(winners) < 2:
                winners = []

    if winners:
        # Trong phe thắng, lấy phân hoạch của phiếu confidence cao hơn (rồi theo
        # thứ tự chạy) làm đại diện keep/distinct_name.
        best = sorted(
            winners, key=lambda v: (0 if v["confidence"] == "high" else 1)
        )[0]
        consensus = {
            "clusters": best["clusters"],
            "verdict": _verdict(best["clusters"], len(member_ids)),
            "agreement": f"{len(winners)}/{len([v for v in votes if 'clusters' in v])}",
            "confidence": best["confidence"],
        }
    else:
        consensus = {"clusters": [], "verdict": "no_consensus", "agreement": "0", "confidence": "low"}

    return {"votes": votes, "consensus": consensus}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM phân xử nhóm trùng tên TayY -> sidecar gợi ý (không sửa CSV)."
    )
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=0, help="Chỉ xử lý N nhóm mới (0 = tất cả)")
    parser.add_argument(
        "--votes",
        type=int,
        choices=(1, 3),
        default=3,
        help="3 = 2 phiếu + tie-break (mặc định); 1 = một phiếu (chỉ để thăm dò)",
    )
    parser.add_argument(
        "--vote-provider",
        choices=("requesty", "dashscope"),
        default="requesty",
        help="Provider cho 2 phiếu chính (Requesty 402 hết số dư -> dùng dashscope)",
    )
    parser.add_argument("--vote-model", default=DEFAULT_REQUESTY_MODEL)
    parser.add_argument("--ds-model", default=DEFAULT_DASHSCOPE_MODEL,
                        help="Model tie-break (luôn DashScope)")
    args = parser.parse_args(argv)

    load_env(ROOT)
    clean = load_clean(args.clean)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clean.rows:
        flags = (row.get("cờ_chất_lượng") or "").split(";")
        if DUPLICATE_NAME_FLAG in flags:
            groups[(row.get("tên_bệnh") or "").casefold()].append(row)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"Nhóm trùng tên: {len(groups)} (tổng {sum(len(v) for v in groups.values())} dòng)")

    done = load_done(args.output, "group_key")

    def is_fresh(record: dict) -> bool:
        hashes = record.get("row_sha256") or {}
        ids = record.get("disease_ids") or []
        if not ids or any(
            clean.row_hashes.get(disease_id) != hashes.get(disease_id) for disease_id in ids
        ):
            return False
        # no_consensus do PHIẾU LỖI (mạng/quota — <2 phiếu hợp lệ) không phải kết quả
        # thật -> coi như chưa chạy để lần sau retry; bất đồng THẬT (>=2 phiếu hợp lệ
        # nhưng không cùng phân hoạch) thì giữ, dành cho người xử.
        if (record.get("consensus") or {}).get("verdict") != "no_consensus":
            return True
        valid_votes = sum(1 for v in record.get("votes") or [] if "clusters" in v)
        return valid_votes >= 2

    pending = [
        (key, rows)
        for key, rows in sorted(groups.items())
        if not (key in done and is_fresh(done[key]))
    ]
    print(f"Đã có record hợp lệ: {len(groups) - len(pending)}; cần chạy: {len(pending)}")
    if args.limit > 0:
        pending = pending[: args.limit]

    vote_models = [m.strip() for m in args.vote_model.split(",") if m.strip()]
    if len(vote_models) == 1:
        vote_models = vote_models * 2
    # Mỗi phiếu là một CHUỖI dự phòng: model cạn quota (402/403) -> tự chuyển
    # model kế; record vote ghi model THẬT đã dùng.
    voters = [
        (FallbackChat(build_fallback_chain(args.vote_provider, vote_models[0]),
                      temperature=0.15), 42),
        (FallbackChat(build_fallback_chain(args.vote_provider, vote_models[1]),
                      temperature=0.35), 1337),
    ]
    tiebreaker = (FallbackChat(build_fallback_chain("dashscope", args.ds_model),
                               temperature=0.2), 7)

    stats: Counter = Counter()
    for index, (key, rows) in enumerate(pending, 1):
        result = adjudicate_group(
            key, rows, voters, tiebreaker, max_votes=1 if args.votes == 1 else 2
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "group_key": key,
            "disease_ids": sorted(row["disease_id"] for row in rows),
            "row_sha256": {
                row["disease_id"]: clean.row_hashes[row["disease_id"]] for row in rows
            },
            "votes": result["votes"],
            "consensus": result["consensus"],
            "adjudicated_at": utc_now_iso(),
        }
        append_jsonl(args.output, record)
        verdict = result["consensus"]["verdict"]
        stats[verdict] += 1
        print(f"[{index}/{len(pending)}] {key!r} ({len(rows)} dòng) -> {verdict}")

    # Report tổng trên TOÀN BỘ sidecar hiện có (kể cả các lần chạy trước).
    all_records = load_done(args.output, "group_key")
    fresh = {k: r for k, r in all_records.items() if is_fresh(r)}
    verdicts = Counter(r["consensus"]["verdict"] for r in fresh.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "nguồn": {"clean": "data/TayY_clean.csv", "số_nhóm_trùng_tên": len(groups)},
        "đã_phân_xử": len(fresh),
        "verdict": dict(sorted(verdicts.items())),
        "chạy_lần_này": dict(sorted(stats.items())),
        "ghi_chú": [
            "Sidecar là GỢI Ý máy — không phải phán quyết chuyên gia; consumer phải "
            "đối chiếu row_sha256, lệch là bỏ qua record.",
            "KHÔNG sửa TayY_clean.csv theo sidecar này (phá lineage proposal/review ICD); "
            "materialize thật là quy trình expert-gated re-clean trong tương lai.",
        ],
        "cập_nhật": utc_now_iso(),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] {args.output} ({len(fresh)}/{len(groups)} nhóm hợp lệ)")
    print(f"[OK] {args.report}: {dict(sorted(verdicts.items()))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[LỖI CONTRACT] {exc}", file=sys.stderr)
        raise SystemExit(1)
