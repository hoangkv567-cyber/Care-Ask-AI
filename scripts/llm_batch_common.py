#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hạ tầng dùng chung cho các script LLM-batch offline (đề xuất dữ liệu).

KHÔNG dùng ``qa_pipeline.client`` của app: client đó bắt buộc Neo4j env và bỏ
qua tham số ``model`` (src/qa_system.py:65-124) nên không chọn được model theo
vote. Ở đây gọi thẳng 2 endpoint OpenAI-compatible mà repo đã dùng:

* DashScope compatible-mode  (key ``DASHSCOPE_API_KEY``)
* Requesty router            (key ``REQUESTY_API_KEY``)

Nguyên tắc chung cho mọi script dùng module này:
* Kết quả LLM là "máy đề xuất" — ghi sidecar riêng, KHÔNG bao giờ sửa
  TayY_clean.csv hay review sidecar của chuyên gia.
* Checkpoint: ghi JSONL append + fsync từng record; chạy lại tự resume.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
REQUESTY_URL = "https://router.requesty.ai/v1/chat/completions"

PROVIDERS = {
    "dashscope": (DASHSCOPE_URL, "DASHSCOPE_API_KEY"),
    "requesty": (REQUESTY_URL, "REQUESTY_API_KEY"),
}

# Model mặc định khớp config/config.yaml (dashscope.model / requesty.model).
DEFAULT_REQUESTY_MODEL = "nebius/qwen/qwen3-30b-a3b-instruct-2507"
DEFAULT_DASHSCOPE_MODEL = "qwen3-max-2026-01-23"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class BatchLLMError(RuntimeError):
    """Gọi LLM thất bại sau khi đã cạn retry (mạng/quota/JSON hỏng)."""


class QuotaExhaustedError(BatchLLMError):
    """402/403 — quota/số dư của model đã CẠN HẲN; chờ/backoff không cứu được."""


# Sổ model chết dùng CHUNG toàn tiến trình (mọi thread/FallbackChat cùng thấy):
# một model 402/403 một lần là mọi phiếu sau bỏ qua nó luôn, khỏi đốt retry.
_DEAD_LOCK = threading.Lock()
_DEAD_MODELS: set[tuple[str, str]] = set()


def _mark_dead(provider: str, model: str, reason: str) -> None:
    with _DEAD_LOCK:
        if (provider, model) not in _DEAD_MODELS:
            _DEAD_MODELS.add((provider, model))
            print(f"[QUOTA] {provider}/{model} chết ({reason}) — chuyển model dự phòng")


def _is_dead(provider: str, model: str) -> bool:
    with _DEAD_LOCK:
        return (provider, model) in _DEAD_MODELS


def load_env(root: Path) -> None:
    """Nạp .env theo đúng pattern scripts/build_batcuong_golden.py:87-90."""
    env_path = Path(root) / ".env"
    if not env_path.is_file():
        return
    for line in io.open(env_path, encoding="utf-8"):
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _extract_json(content: str) -> dict:
    """Bóc JSON object từ trả lời LLM (chịu ```json fence và văn bao quanh)."""
    text = (content or "").strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    # Cứu cánh: cắt từ '{' đầu tới '}' cân bằng cuối cùng.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for index in range(start, len(text)):
            ch = text[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        value = json.loads(candidate)
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Không bóc được JSON object từ trả lời LLM: {text[:200]!r}")


class BatchChat:
    """Client 1 provider + 1 model cho một 'vote'. Thread-safe (httpx.Client)."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        temperature: float = 0.1,
        timeout: float = 90.0,
        max_retries: int = 3,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"provider {provider!r} không hợp lệ; cần dashscope|requesty")
        url, env_key = PROVIDERS[provider]
        token = (os.environ.get(env_key) or "").strip()
        if not token:
            raise BatchLLMError(f"Thiếu {env_key} trong môi trường/.env")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._url = url
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def chat_json(self, system: str, user: str, *, seed: int | None = None) -> dict:
        """Một lượt hỏi, trả JSON object. Retry 429/5xx/mạng với backoff 2/8/30s;
        JSON hỏng được nhắc lại đúng 1 lần trong cùng attempt."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        if self.provider == "dashscope":
            # Model hybrid (qwen3-32b/14b/235b...) bắt buộc tắt thinking khi gọi
            # non-streaming — thiếu tham số này là 400. Model thường bỏ qua nó.
            payload["enable_thinking"] = False
        if seed is not None:
            payload["seed"] = seed

        backoffs = [2, 8, 30]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                content = self._post(payload)
                try:
                    return _extract_json(content)
                except ValueError:
                    # Nhắc "chỉ trả JSON" đúng một lần rồi mới tính là lỗi attempt.
                    retry_payload = dict(payload)
                    retry_payload["messages"] = payload["messages"] + [
                        {"role": "assistant", "content": content[:2000]},
                        {
                            "role": "user",
                            "content": "Trả lời KHÔNG hợp lệ. Chỉ trả về đúng một JSON object, "
                                       "không markdown, không giải thích.",
                        },
                    ]
                    return _extract_json(self._post(retry_payload))
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
        raise BatchLLMError(
            f"{self.provider}/{self.model}: cạn {self.max_retries} lượt retry: {last_error}"
        )

    def _post(self, payload: dict) -> str:
        resp = self._client.post(self._url, json=payload)
        if resp.status_code in (400, 401, 402, 403, 404):
            # Lỗi VĨNH VIỄN cho model/key này (cạn quota, sai tên model, sai key)
            # — retry/backoff không cứu được; FallbackChat sẽ chuyển model kế.
            raise QuotaExhaustedError(
                f"{self.provider}/{self.model}: HTTP {resp.status_code} "
                f"({resp.text[:120]})"
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()


# Thang dự phòng DashScope: quota free tính RIÊNG THEO TỪNG PHIÊN BẢN model
# (qwen-plus-latest / qwen-plus-2025-09-11 / ... mỗi cái 1M token riêng — user
# cung cấp danh sách từ console ModelStudio 2026-08-17). Xếp theo chất lượng
# cho bài chọn-mã-y-khoa: họ plus/max trước, model mở lớn giữa, flash cuối.
# CHỈ model text non-thinking (thinking đốt quota vào chuỗi suy nghĩ).
DASHSCOPE_FALLBACK_LADDER = (
    "qwen-plus-latest",
    "qwen3.5-plus",
    "qwen-plus-2025-09-11",
    "qwen3.6-plus",
    "qwen-plus-2025-07-28",
    "qwen3.7-plus",
    "qwen3.7-plus-2026-05-26",
    "qwen-plus-2025-04-28",
    "qwen3.5-plus-2026-02-15",
    "qwen3.7-max-2026-06-08",
    "qwen3.7-max-preview",
    "qwen3.6-max-preview",
    "qwen3-max-preview",
    "qwen3-max-2025-09-23",
    "qwen3.7-max-2026-05-20",
    "qwen3.7-max-2026-05-17",
    "qwen-max",
    "deepseek-v3.2",
    "glm-5.1",
    "glm-5.2",
    "qwen3-32b",
    "qwen3.6-27b",
    "qwen3.5-27b",
    "qwen3-235b-a22b",
    "qwen3.5-122b-a10b",
    "qwen3.5-397b-a17b",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3.7-flash",
    "qwen3.6-flash",
    "qwen3.5-flash",
    "qwen-flash-2025-07-28",
    "deepseek-v4-flash-0731",
    "qwen3.5-35b-a3b",
    "qwen3.8-2.4t-a95b",
    "kimi-k2.7-code",
    # Đợt bổ sung 2026-08-18 (ảnh console user): quota còn nguyên + các model
    # hybrid mở được nhờ enable_thinking=false.
    "qwen3-max",
    "qwen3.8-max",
    "qwen3.7-max",
    "deepseek-v4-pro-0813",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "qwen3.6-plus-2026-04-02",
    "qwen3.5-plus-2026-04-20",
    "qwen-plus-2025-07-14",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-next-80b-a3b-instruct",
    "qwen3.6-35b-a3b",
    "qwen3-32b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3.7-flash-2026-07-15",
    "qwen3.5-flash-2026-02-23",
    "qwen3.6-flash-2026-04-16",
    "qwen3-coder-plus",
    "qwen3-coder-next",
    "qwen3-coder-flash",
)


def build_fallback_chain(provider: str, model: str) -> list[tuple[str, str]]:
    """Chuỗi dự phòng: model được yêu cầu trước, rồi TOÀN BỘ thang DashScope
    (mỗi phiên bản 1 quota riêng), cuối cùng Requesty (nếu chưa có)."""
    chain: list[tuple[str, str]] = [(provider, model)]
    for fallback in DASHSCOPE_FALLBACK_LADDER:
        if ("dashscope", fallback) not in chain:
            chain.append(("dashscope", fallback))
    if not any(p == "requesty" for p, _ in chain):
        chain.append(("requesty", DEFAULT_REQUESTY_MODEL))
    return chain


class FallbackChat:
    """Client 1 phiếu với CHUỖI model dự phòng: model nào trả 402/403 (cạn
    quota/số dư) bị đánh dấu chết toàn cục và phiếu tự chuyển model kế tiếp.
    Sau mỗi call thành công, ``.provider``/``.model`` phản ánh model THẬT đã
    dùng — record vote ghi đúng nguồn gốc."""

    def __init__(
        self,
        chain: list[tuple[str, str]],
        *,
        temperature: float = 0.1,
        timeout: float = 90.0,
        max_retries: int = 3,
    ) -> None:
        if not chain:
            raise ValueError("chain rỗng")
        self._chain = list(chain)
        self._kw = dict(temperature=temperature, timeout=timeout, max_retries=max_retries)
        self._clients: dict[tuple[str, str], BatchChat] = {}
        self.provider, self.model = self._chain[0]

    def chat_json(self, system: str, user: str, *, seed: int | None = None) -> dict:
        last_error: Exception | None = None
        for provider, model in self._chain:
            if _is_dead(provider, model):
                continue
            key = (provider, model)
            client = self._clients.get(key)
            if client is None:
                try:
                    client = self._clients[key] = BatchChat(provider, model, **self._kw)
                except BatchLLMError as exc:      # thiếu API key trong .env
                    last_error = exc
                    _mark_dead(provider, model, "thiếu API key")
                    continue
            try:
                out = client.chat_json(system, user, seed=seed)
                self.provider, self.model = provider, model
                return out
            except QuotaExhaustedError as exc:
                last_error = exc
                _mark_dead(provider, model, "402/403")
                continue
        raise BatchLLMError(
            f"Mọi model trong chuỗi dự phòng đều cạn quota/chết: {self._chain}: {last_error}"
        )

    def close(self) -> None:
        for client in self._clients.values():
            client.close()


_APPEND_LOCK = threading.Lock()


def append_jsonl(path: Path, record: dict) -> None:
    """Append 1 record + fsync = checkpoint bền qua crash. Thread-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def load_done(path: Path, key: str) -> dict[str, dict]:
    """Đọc JSONL đã có -> {record[key]: record} để resume (record sau đè record trước)."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} dòng {line_no}: JSON lỗi: {exc}") from exc
            if isinstance(record, dict) and record.get(key):
                out[str(record[key])] = record
    return out


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
