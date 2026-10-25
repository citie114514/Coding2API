"""OpenAI 形状 → Anthropic Messages（MiniMax 版消息转换）。

与 zcode 的 anthropic_compat 分开实现，因为 MiniMax 上游对**配对纪律**是
硬 400（实测 `tool call result does not follow tool call (2013)`）：

- 同批 `tool_result` 必须合并进一条 user 消息（harness 把每个工具调用落成
  独立 tool 消息，逐条下发会产出多条连续 user ⇒ 2013）；
- 不得跨 assistant 边界累积（`assistant A / assistant B / user(A+B)` 同样
  2013）。

故用「assistant 进待发区，遇到结果或下一条非工具消息时成对提交」的算法
（free-llm-router `src/minimax-messages.ts:224-355` 同款）。

思考形态决策表（`src/minimax-messages.ts:108-140`）：

| 情形 | 请求体 |
|---|---|
| effort='none'（switchable 模型） | `thinking:{type:'disabled'}` |
| effort='on'（switchable 模型） | `thinking:{type:'adaptive'}` |
| 有档位 effort | `thinking:{type:'adaptive'}` + `output_config.effort` |
| M3.1 前缀（无档位也必须） | `thinking:{type:'adaptive'}`，否则 **400** |
| 其余（M2.7 系） | 整个不发（服务端默认 forced_on） |

`disabled` 只对目录声明 `switchable` 的模型可达：M3.1 传 disabled 硬拒、
M2.7 传 disabled 被静默忽略（用户以为关了、实际没关）。未知档位一律丢弃。
"""

from __future__ import annotations

import json
from typing import Any


class PayloadError(ValueError):
    """请求体转换失败（如图片 URL 非法）——由客户端层转 400 给调用方。"""


def _read_str(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _loose_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _parse_image_data_url(url: str) -> tuple[str, str]:
    """解析 `data:<mime>;base64,<数据>`。

    ⚠️ 只认 `data:` URL，**绝不下载 http(s) 链接**（SSRF 红线）；拿到普通
    URL 时抛错而不是静默丢图 —— 静默丢图会让用户以为模型看到了图片。
    """
    if not url.startswith("data:"):
        raise PayloadError(
            "MiniMax 只接受内联（`data:` URL）图片，不接受 http(s) 图片链接。"
            "请让客户端把图片以 base64 直接内联在 `image_url.url` 里")
    comma = url.find(",")
    if comma < 0:
        raise PayloadError("图片 data URL 缺少逗号分隔符")
    meta = url[5:comma]
    data = url[comma + 1:].strip()
    raw_mime, _, raw_encoding = meta.partition(";")
    media_type = raw_mime.strip() or "image/png"
    if raw_encoding.strip().lower() not in ("", "base64"):
        raise PayloadError(f"图片 data URL 的编码是「{raw_encoding}」，MiniMax 只接受 base64")
    if not data:
        raise PayloadError("图片 data URL 里没有 base64 数据")
    return media_type, data


def _content_to_blocks(content: Any) -> list[dict[str, Any]]:
    """OpenAI 文本/多模态 content → Anthropic 内容块数组。"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in ("text", None):
            text = part.get("text")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = part.get("url") or image_url
            if not isinstance(url, str):
                raise PayloadError("图片块缺少 `image_url.url`")
            media_type, data = _parse_image_data_url(url)
            blocks.append({"type": "image",
                           "source": {"type": "base64", "media_type": media_type,
                                      "data": data}})
            continue
        # 未知块类型：不静默丢、也不打挂整轮 —— 忽略（与 TS 同口径只记日志）
    return blocks


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """`tool_calls[].function.arguments`（字符串）→ 对象。

    解析失败退化为 `{}`（不编造参数），但**不能丢掉整条 tool_use** ——
    丢了会让后续 `tool_result` 变成孤儿块，服务端 400。
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def to_anthropic_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """OpenAI 消息数组 → Anthropic messages（system 必须已由调用方抽走）。"""
    out: list[dict[str, Any]] = []
    pending_assistant: dict[str, Any] | None = None
    pending_tool_results: list[dict[str, Any]] = []

    def commit_pending() -> None:
        nonlocal pending_assistant, pending_tool_results
        if pending_assistant is not None:
            out.append(pending_assistant)
            pending_assistant = None
        if pending_tool_results:
            out.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        if role in ("system", "developer"):
            continue
        if role == "tool":
            tool_call_id = _read_str(raw, ("tool_call_id", "toolCallId"))
            if tool_call_id is None:
                continue
            entry: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_call_id,
                                     "content": _content_to_blocks(raw.get("content"))}
            if raw.get("isError") is True or raw.get("is_error") is True:
                entry["is_error"] = True
            pending_tool_results.append(entry)
            continue
        if role == "assistant":
            # 成对提交上一批，再转入待发区（防跨 assistant 边界累积 → 2013）
            commit_pending()
            blocks = _content_to_blocks(raw.get("content"))
            # 历史里的 thinking 不回传：Anthropic 要求带签名，回传无签名块会被拒
            tool_calls = raw.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = _read_str(call, ("id",))
                    fn = call.get("function")
                    fn = fn if isinstance(fn, dict) else {}
                    name = _read_str(fn, ("name",))
                    if call_id is None or name is None:
                        continue            # 名称为空的 tool_call 会让上游 400 且不指字段
                    blocks.append({"type": "tool_use", "id": call_id, "name": name,
                                   "input": _parse_tool_arguments(fn.get("arguments"))})
            if blocks:
                pending_assistant = {"role": "assistant", "content": blocks}
            continue
        # user（含未知角色按 user 处理）
        commit_pending()
        blocks = _content_to_blocks(raw.get("content"))
        if blocks:
            out.append({"role": "user", "content": blocks})
    commit_pending()                            # 收尾必须提交，否则留下无结果的 tool_use
    return out


def to_anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    """OpenAI tools → Anthropic `input_schema` 形态（兼容已扁平输入）。"""
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        fn = fn if isinstance(fn, dict) else tool
        name = _read_str(fn, ("name",))
        if name is None:
            continue
        entry: dict[str, Any] = {"name": name}
        description = _read_str(fn, ("description",))
        if description is not None:
            entry["description"] = description
        schema = fn.get("parameters")
        if not isinstance(schema, dict):
            schema = fn.get("input_schema")
        entry["input_schema"] = schema if isinstance(schema, dict) else {
            "type": "object", "properties": {}}
        out.append(entry)
    return out


def normalize_effort(raw: Any, model_id: str,
                     entry: dict[str, Any] | None) -> tuple[str | None, bool]:
    """归一思考档位 → (effort, requires_adaptive)。

    ⚠️ requiresAdaptive **按模型 id 前缀判**（不依赖目录条目是否拉到）：
    M3.1 漏发 adaptive 是服务端硬 400。目录未知时不放行任何档位。
    """
    adaptive_prefix = "MiniMax-M3.1"
    requires_adaptive = model_id.startswith(adaptive_prefix) or (
        entry is not None and str(entry.get("id", "")).startswith(adaptive_prefix))
    if not (isinstance(raw, str) and raw.strip()):
        return None, requires_adaptive
    effort = raw.strip()
    declared = set(entry.get("effort_options") or []) if entry else set()
    if entry and entry.get("thinking_mode") == "switchable":
        declared.update(("on", "none"))
    if effort not in declared:
        return None, requires_adaptive
    return effort, requires_adaptive


def build_messages_body(payload: dict[str, Any], model: str,
                        entry: dict[str, Any] | None) -> dict[str, Any]:
    """引擎 payload（OpenAI 形状）→ MiniMax Anthropic Messages 请求体。"""
    raw_messages = payload.get("messages")
    raw_messages = raw_messages if isinstance(raw_messages, list) else []
    system_parts: list[str] = []
    dialogue: list[Any] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                text = content
            else:
                text = "\n".join(
                    block.get("text") for block in _content_to_blocks(content)
                    if isinstance(block.get("text"), str) and block.get("text"))
            if text and text.strip():
                system_parts.append(text)
            continue
        dialogue.append(message)

    body: dict[str, Any] = {"model": model, "stream": True,
                            "messages": to_anthropic_messages(dialogue)}
    if system_parts:
        body["system"] = "\n\n".join(system_parts)

    # max_tokens：Anthropic 必填。上限收敛到目录声明的单次输出上限
    #（客户端可能带着从别家抄来的大值，上游 400 拒整轮且不说原因）。
    requested = _loose_number(payload.get("max_tokens"))
    if requested is None:
        requested = _loose_number(payload.get("max_completion_tokens"))
    ceiling = entry.get("max_output") if entry else None
    if isinstance(requested, (int, float)) and requested > 0 and requested == int(requested):
        max_tokens = int(requested)
        if isinstance(ceiling, (int, float)) and ceiling > 0:
            max_tokens = min(max_tokens, int(ceiling))
        body["max_tokens"] = max_tokens
    elif isinstance(ceiling, (int, float)) and ceiling > 0:
        body["max_tokens"] = int(ceiling)

    temperature = _loose_number(payload.get("temperature"))
    if temperature is not None:
        body["temperature"] = temperature
    top_p = _loose_number(payload.get("top_p"))
    if top_p is not None:
        body["top_p"] = top_p
    stop = payload.get("stop")
    if isinstance(stop, list) and stop:
        body["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    elif isinstance(stop, str) and stop:
        body["stop_sequences"] = [stop]

    tools = to_anthropic_tools(payload.get("tools"))
    choice = payload.get("tool_choice")
    choice_type = choice if isinstance(choice, str) else (
        choice.get("type") if isinstance(choice, dict) else None)
    # Anthropic 没有「禁用工具」：tool_choice='none' 只能整段不发 tools 表达
    tools_disabled = choice_type == "none"
    if tools and not tools_disabled:
        body["tools"] = tools
        if choice_type == "required":
            body["tool_choice"] = {"type": "any"}
        elif choice_type in ("auto", None):
            body["tool_choice"] = {"type": "auto"}
        else:
            fn = choice.get("function") if isinstance(choice, dict) else None
            name = _read_str(fn, ("name",)) if isinstance(fn, dict) else None
            if name is None and isinstance(choice, str):
                name = choice
            body["tool_choice"] = ({"type": "tool", "name": name}
                                   if name else {"type": "auto"})

    effort_input = payload.get("reasoning_effort")
    if not effort_input and isinstance(payload.get("reasoning"), dict):
        effort_input = payload["reasoning"].get("effort")
    effort, requires_adaptive = normalize_effort(effort_input, model, entry)
    if effort == "none":
        body["thinking"] = {"type": "disabled"}
    elif effort == "on":
        body["thinking"] = {"type": "adaptive"}
    elif effort is not None:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": effort}
    elif requires_adaptive:
        body["thinking"] = {"type": "adaptive"}
    return body
