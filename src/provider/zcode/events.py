"""ZCode 事件层：错误分类 + Anthropic Messages SSE → 中立 Event。

错误语义移植自 free-llm-router `src/providers/zcode.ts`：

- `3009`（model concurrency limit）是**并发配额**，与剩余额度无关 ——
  「等一下就能过」，与 429 的另一种语义（额度用尽）处置完全不同，必须分开。
- `1005`（exceed quota limit）/ `1113`（余额不足）是账号维度额度耗尽 ——
  要换号。**必须排除 3009**：并发限流同样返回 429，误判会把完全可用的
  账号错标成「当日用尽」。
- `3012`（unusual activity）是风控拦截 —— 不可重试，有账号冷却惩罚
  （30 分钟；24h 内第 3 次起 24h；5 次停用），也不该换号加重。
- 关键词兜底必须**窄**（`exceed quota limit` / `余额不足`）：`quota` /
  `balance` 之类泛词会误伤模型正文里恰好讨论「额度」的内容。

SSE 侧：上游是标准 Anthropic Messages 流（message_start / content_block_* /
message_delta / message_stop / error），映射到中立 Event：
`text_delta` → CONTENT，`thinking_delta` → REASONING，`input_json_delta` →
TOOL_CALLS 增量分片，`message_delta.usage` → USAGE，stop_reason → FINISH。

流内状态（工具块序号映射、usage 累计）由 `StreamState` 承载，**每个流一份**、
由消费方（client.stream_chat）创建并显式传入 —— Anthropic 的块序号在
`content_block_start` 与 `input_json_delta` 之间跨帧引用，没有流内状态就
无法把分片对回正确的工具调用。
"""

from __future__ import annotations

import json
from typing import Any

from ...provider.base import (
    ErrKind,
    Event,
    EventKind,
    UpstreamProtocolViolation,
    Usage,
)

CONCURRENCY_CODE = "3009"
QUOTA_CODE = "1005"
BALANCE_CODE = "1113"
RISK_CODE = "3012"
TOKEN_INVALID_CODE = "1002"


def is_concurrency_limited(body: str) -> bool:
    return CONCURRENCY_CODE in body or "concurrency limit" in body.lower()


def is_quota_exhausted(body: str) -> bool:
    if is_concurrency_limited(body):
        return False
    if BALANCE_CODE in body or "余额不足" in body:
        return True
    if QUOTA_CODE in body:
        return True
    lowered = body.lower()
    return "exceed quota limit" in lowered or "quota exhausted" in lowered


def is_risk_blocked(body: str) -> bool:
    return RISK_CODE in body or "unusual activity" in body.lower()


def classify_error(status: int, text: str) -> ErrKind:
    if is_risk_blocked(text):
        return ErrKind.OTHER                   # 风控：账号级累计冷却，不加重
    if is_concurrency_limited(text):
        return ErrKind.CONCURRENCY
    if is_quota_exhausted(text):
        return ErrKind.PLAN
    if TOKEN_INVALID_CODE in text:
        return ErrKind.DEAD
    if status in (401, 403):
        return ErrKind.DEAD
    if status == 402:
        return ErrKind.CREDIT
    if status == 429:
        return ErrKind.SOFT
    if status >= 500:
        return ErrKind.SOFT
    if status == 400:
        return ErrKind.INVALID
    return ErrKind.OTHER


def classify_status(status: int, body: bytes = b"") -> ErrKind:
    """HTTP 状态 + body → ErrKind（UpstreamHTTPError.classify_status 绑定点）。"""
    return classify_error(status, body.decode("utf-8", errors="replace"))


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class StreamState:
    """一个 Anthropic 流的转换状态（每流一份，无并发共享）。"""

    __slots__ = ("tool_ordinals", "next_tool_ordinal", "saw_tool_call")

    def __init__(self) -> None:
        # Anthropic 块序号 → (块类型, 工具调用序号)
        self.tool_ordinals: dict[int, tuple[str, int]] = {}
        self.next_tool_ordinal = 0
        self.saw_tool_call = False


def _tool_start_event(state: StreamState, index: int, block: dict[str, Any]) -> Event:
    """tool_use 块开始：发首片（id + name，arguments 置空串）。"""
    ordinal = state.next_tool_ordinal
    state.next_tool_ordinal += 1
    state.saw_tool_call = True
    state.tool_ordinals[index] = ("tool_use", ordinal)
    return Event(kind=EventKind.TOOL_CALLS, tool_calls=[{
        "index": ordinal,
        "id": str(block.get("id") or ""),
        "type": "function",
        "function": {"name": str(block.get("name") or ""), "arguments": ""},
    }])


def _finish_reason(stop_reason: str) -> str:
    """Anthropic stop_reason → OpenAI finish_reason。

    有工具调用时必须报 `tool_calls`（客户端继续执行工具的依据），
    不论上游 stop_reason 说什么。
    """
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    return "stop"


def parse_frame(frame: SSEFrame, state: StreamState) -> list[Event]:
    """一个 SSE 帧 → 中立事件列表。"""
    event_name = frame.event
    try:
        payload = json.loads(frame.data)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    kind = payload.get("type") or event_name
    if kind in ("ping", "pong"):
        return []

    if kind == "error":
        error = payload.get("error")
        error = error if isinstance(error, dict) else payload
        message = error.get("message") if isinstance(error.get("message"), str) else ""
        return [Event(kind=EventKind.ERROR, error_message=message or frame.data[:300],
                      error_kind=classify_error(200, frame.data))]

    if kind == "message_start":
        message = payload.get("message")
        message = message if isinstance(message, dict) else {}
        usage = message.get("usage")
        if isinstance(usage, dict):
            # message_start 只有 input 侧计数；output 在 message_delta
            return [Event(kind=EventKind.USAGE, usage=Usage(
                input_tokens=_int_or_none(usage.get("input_tokens")),
                cached_tokens=_int_or_none(usage.get("cache_read_input_tokens")),
            ))]
        return []

    if kind == "content_block_start":
        index = _int_or_none(payload.get("index")) or 0
        block = payload.get("content_block")
        block = block if isinstance(block, dict) else {}
        if block.get("type") == "tool_use":
            return [_tool_start_event(state, index, block)]
        return []

    if kind == "content_block_delta":
        index = _int_or_none(payload.get("index")) or 0
        delta = payload.get("delta")
        delta = delta if isinstance(delta, dict) else {}
        delta_type = delta.get("type")
        if delta_type == "text_delta" and delta.get("text"):
            return [Event(kind=EventKind.CONTENT, content=delta["text"])]
        if delta_type == "thinking_delta" and delta.get("thinking"):
            return [Event(kind=EventKind.REASONING, content=delta["thinking"])]
        if delta_type == "input_json_delta":
            partial = delta.get("partial_json")
            if not (isinstance(partial, str) and partial):
                return []
            entry = state.tool_ordinals.get(index)
            if entry is None or entry[0] != "tool_use":
                return []
            return [Event(kind=EventKind.TOOL_CALLS, tool_calls=[{
                "index": entry[1],
                "function": {"arguments": partial},
            }])]
        return []

    if kind == "message_delta":
        delta = payload.get("delta")
        delta = delta if isinstance(delta, dict) else {}
        events: list[Event] = []
        usage = payload.get("usage")
        if isinstance(usage, dict):
            output = _int_or_none(usage.get("output_tokens"))
            if output is not None:
                events.append(Event(kind=EventKind.USAGE,
                                    usage=Usage(output_tokens=output)))
        stop_reason = delta.get("stop_reason")
        if isinstance(stop_reason, str) and stop_reason:
            finish = _finish_reason(stop_reason)
            if state.saw_tool_call:
                finish = "tool_calls"
            events.append(Event(kind=EventKind.FINISH, finish_reason=finish))
        return events

    return []                                       # content_block_stop / message_stop / 未知


def parse_models_payload(data: Any) -> list[dict[str, Any]]:
    """上游 client/configs 的 `builtinModels` → 模型条目列表。

    ⚠️ `builtinModels` 是**对象**（键是序号字串）也可能是数组 —— 两种都认，
    只认数组会对真实形态得出「0 个模型」的假阴性（free-llm-router 实测踩过）。
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("builtinModels")
    if isinstance(raw, dict):
        entries = list(raw.values())
    elif isinstance(raw, list):
        entries = raw
    else:
        return []
    out: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        model_id = item.get("modelId") or item.get("id")
        if not (isinstance(model_id, str) and model_id):
            continue
        capabilities = item.get("capabilities")
        vision = isinstance(capabilities, dict) and capabilities.get("vision") is True
        reasoning = item.get("reasoning")
        levels: list[str] | None = None
        default_level: str | None = None
        if isinstance(reasoning, dict):
            raw_levels = reasoning.get("levels")
            if isinstance(raw_levels, dict) and raw_levels:
                levels = _order_reasoning_levels(list(raw_levels.keys()))
            raw_default = reasoning.get("defaultLevel")
            if isinstance(raw_default, str) and raw_default:
                default_level = raw_default
        context = item.get("contextWindow")
        max_tokens = item.get("maxCompletionTokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            max_tokens = item.get("maxTokens")
        out.append({
            "id": model_id,
            "name": item.get("name") if isinstance(item.get("name"), str) else model_id,
            "context_window": context if isinstance(context, int) and context > 0 else 200_000,
            "max_tokens": max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 32_768,
            "supports_image": vision,
            "reasoning_levels": levels,
            "default_reasoning_level": default_level,
        })
    return out


_REASONING_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _order_reasoning_levels(keys: list[str]) -> list[str]:
    """档位键排成 IDE 展示序（上游 JSON 键序实测是 low, max, high —— 不能直接用）。"""
    known = [level for level in _REASONING_ORDER if level in keys]
    unknown = [key for key in keys if key not in _REASONING_ORDER]
    return known + unknown


def protocol_violation(what: str) -> UpstreamProtocolViolation:
    return UpstreamProtocolViolation(f"zcode {what}")
