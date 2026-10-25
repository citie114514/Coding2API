"""OpenAI 形状 → Anthropic Messages 请求体转换（ZCode 专用）。

移植自 free-llm-router `src/providers/anthropic.ts` 的请求侧转换，语义逐条对齐：

1. `role:'system'` 抽到顶层 `system`（Anthropic 无 system 角色）；多条 system
   拼接，不丢弃。
2. `role:'tool'` → `role:'user'` 的 `tool_result` 块；连续工具结果合并进同一条
   user 消息。
3. assistant 的 `tool_calls` → `tool_use` 块；`arguments` 字符串 → `input` 对象。
4. 空 assistant 消息丢弃（上游 400）。
5. `tool_choice`：OpenAI `required` → Anthropic `any`（非同名映射）；`none` →
   None 且调用方不下发 tools（Anthropic 无 none 取值）。

身份块（identity_blocks）是上游 3012 风控的准入判据：官方客户端总会带
CLI 身份前缀 + stable 段 + 环境段，且首轮 user 消息插日期上下文块。
逐字复刻，不要「优化」——详见 identity_blocks.py 文件头。
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

from . import identity_blocks as blocks

_UNPARSABLE_KEY = "__unparsableArguments"


def _text_of(content: Any) -> str:
    """OpenAI content（字符串或块数组）里的纯文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "input_text") and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _parse_tool_arguments(raw: Any) -> Any:
    """工具参数 JSON 字符串 → 对象。残缺参数不补 `{}`（让它显式失败）。"""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    trimmed = raw.strip()
    if not trimmed:
        return {}
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return {_UNPARSABLE_KEY: trimmed}


def split_system_messages(wire: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """抽出 system 内容（拼接）；返回 (system, 其余消息)。"""
    parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for message in wire:
        if isinstance(message, dict) and message.get("role") == "system":
            text = _text_of(message.get("content"))
            if text:
                parts.append(text)
            continue
        rest.append(message)
    return ("\n\n".join(parts) if parts else None), rest


def _image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """OpenAI image 块 → Anthropic image 块（只接受 data: URL，不下载 http）。"""
    url = block.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str):
        return None
    prefix = "data:"
    if not url.startswith(prefix):
        return None
    header, _, data = url[len(prefix):].partition(";base64,")
    if not data:
        return None
    media_type = header or "image/png"
    if media_type == "image/*":
        media_type = "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": data}}


def to_anthropic_messages(wire: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI 消息数组 → Anthropic messages（system 必须已抽走）。"""
    out: list[dict[str, Any]] = []

    def push_tool_result(block: dict[str, Any]) -> None:
        last = out[-1] if out else None
        if last is not None and last.get("role") == "user" and isinstance(last.get("content"), list):
            last["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for message in wire:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        if role == "tool":
            tool_use_id = message.get("tool_call_id")
            if not (isinstance(tool_use_id, str) and tool_use_id):
                continue                      # 孤儿工具结果：Anthropic 会 400
            push_tool_result({"type": "tool_result", "tool_use_id": tool_use_id,
                              "content": _text_of(message.get("content"))})
            continue
        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            text = _text_of(message.get("content"))
            if text:
                content_blocks.append({"type": "text", "text": text})
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    fn = fn if isinstance(fn, dict) else {}
                    name = fn.get("name")
                    if not (isinstance(name, str) and name):
                        continue
                    call_id = call.get("id")
                    content_blocks.append({
                        "type": "tool_use",
                        "id": call_id if isinstance(call_id, str) and call_id else "call_x",
                        "name": name,
                        "input": _parse_tool_arguments(fn.get("arguments")),
                    })
            if not content_blocks:
                continue                      # 空 assistant：上游 400
            out.append({"role": "assistant", "content": content_blocks})
            continue
        # user（含多模态）
        content = message.get("content")
        if isinstance(content, str):
            if content:
                out.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        content_blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                content_blocks.append({"type": "text", "text": block["text"]})
                continue
            if block_type in ("image", "image_url", "input_image"):
                image = _image_block(block if "image_url" in block else
                                     {"image_url": block.get("image_url") or block.get("url")})
                if image is not None:
                    content_blocks.append(image)
        if content_blocks:
            out.append({"role": "user", "content": content_blocks})
    return out


def to_anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    """OpenAI 工具表（嵌套 function 或已扁平）→ Anthropic `input_schema` 形态。"""
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        fn = fn if isinstance(fn, dict) else tool
        name = fn.get("name")
        if not (isinstance(name, str) and name):
            continue
        entry: dict[str, Any] = {"name": name}
        description = fn.get("description")
        if isinstance(description, str) and description:
            entry["description"] = description
        parameters = fn.get("parameters")
        entry["input_schema"] = parameters if isinstance(parameters, dict) else {
            "type": "object", "properties": {}}
        out.append(entry)
    return out


def to_anthropic_tool_choice(choice: Any) -> dict[str, Any] | None:
    """OpenAI tool_choice → Anthropic tool_choice；`none` → None（不下发 tools）。"""
    if isinstance(choice, str):
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}             # 非同名映射
        if choice == "none":
            return None
        return {"type": "tool", "name": choice}
    if isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "function":
            fn = choice.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            return {"type": "tool", "name": name} if isinstance(name, str) and name else None
        if choice_type in ("auto", "any", "tool"):
            return choice
    return None


def _with_tool_cache_breakpoint(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给最后一个 tool 打 prompt caching 断点（前缀式缓存 ⇒ 覆盖 system+全部 tools）。"""
    if not tools:
        return tools
    out = [dict(tool) for tool in tools]
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def _format_local_iso_date(now: _dt.datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _build_context_prefix_block(now: _dt.datetime) -> dict[str, Any]:
    """`<system-reminder>` 日期块（逐字复刻官方形态，形态细节是准入判据）。"""
    body = "\n".join([
        blocks.CONTEXT_PREFIX_INTRO,
        f"# currentDate\nToday's date is {_format_local_iso_date(now)}.",
        "",
        blocks.CONTEXT_PREFIX_OUTRO,
    ])
    return {"type": "text", "text": f"<system-reminder>{body}</system-reminder>"}


def _starts_with_system_reminder(content: Any) -> bool:
    if isinstance(content, str):
        return content.startswith("<system-reminder>")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        return isinstance(text, str) and text.startswith("<system-reminder>")
    return False


def _with_context_prefix(messages: list[dict[str, Any]],
                         now: _dt.datetime) -> list[dict[str, Any]]:
    """给首轮 user 消息的 content 数组最前面插日期块（幂等）。"""
    out = [dict(message) for message in messages]
    first = out[0] if out else None
    if first is None or first.get("role") != "user":
        return out
    if _starts_with_system_reminder(first.get("content")):
        return out
    prefix = _build_context_prefix_block(now)
    content = first.get("content")
    if isinstance(content, list):
        first["content"] = [prefix, *content]
    elif isinstance(content, str):
        first["content"] = [prefix, {"type": "text", "text": content}]
    else:
        first["content"] = [prefix]
    return out


def build_system_blocks(caller_system: str | None, *, model: str,
                        cwd: str = "/workspace", platform: str = "win32") -> list[dict[str, Any]]:
    """官方形态的 system 块数组：cliPrefix + stable + environment (+ 调用方)。

    调用方内容**必须追加在最后**（上游检查身份块的前缀结构）。
    只在最后一块打缓存断点（断点上限 4 个，留给 tools）。
    """
    stable = "\n\n".join(blocks.OFFICIAL_STABLE_SECTIONS)
    environment = "\n".join([
        "# Environment",
        "You have been invoked in the following environment:",
        f" - Primary working directory: {cwd if cwd.strip() else '.'}",
        " - Is a git repository: no",
        f" - Platform: {platform}",
        f" - Shell: {'powershell' if platform == 'win32' else 'bash'}",
        f" - OS Version: {platform} x64",
        f" - You are powered by the model named zcode/{model}.",
    ])
    out: list[dict[str, Any]] = [
        {"type": "text", "text": blocks.OFFICIAL_CLI_PREFIX},
        {"type": "text", "text": stable},
        {"type": "text", "text": environment},
    ]
    if isinstance(caller_system, str) and caller_system.strip():
        out.append({"type": "text", "text": caller_system})
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def build_messages_body(payload: dict[str, Any], model: str,
                        now: _dt.datetime | None = None,
                        allowed_efforts: list[str] | None = None) -> dict[str, Any]:
    """引擎 payload（OpenAI 形状）→ ZCode Anthropic Messages 请求体。

    `allowed_efforts`：该模型声明的思考档位（上游 client/configs 下发）。
    未知档位直接下发可能被上游拒（整个请求失败），故只在声明列表内才写；
    也不发默认值（上游有自己的 defaultLevel，不覆盖）。"""
    now = now or _dt.datetime.now().astimezone()
    wire = payload.get("messages")
    wire = wire if isinstance(wire, list) else []
    caller_system, rest = split_system_messages(wire)
    messages = _with_context_prefix(to_anthropic_messages(rest), now)

    body: dict[str, Any] = {
        "model": model,
        # Anthropic 必填字段；缺省 8192（官方客户端同款兜底）
        "max_tokens": 8192,
        "system": build_system_blocks(caller_system, model=model),
        "messages": messages,
        "stream": True,
    }
    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        max_tokens = payload.get("max_completion_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        body["max_tokens"] = max_tokens
    temperature = payload.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        body["temperature"] = temperature
    top_p = payload.get("top_p")
    if isinstance(top_p, (int, float)) and not isinstance(top_p, bool):
        body["top_p"] = top_p
    stop = payload.get("stop")
    if isinstance(stop, list) and stop:
        body["stop_sequences"] = stop
    elif isinstance(stop, str) and stop:
        body["stop_sequences"] = [stop]
    tools = to_anthropic_tools(payload.get("tools"))
    if tools and payload.get("tool_choice") != "none":
        body["tools"] = _with_tool_cache_breakpoint(tools)
    choice = to_anthropic_tool_choice(payload.get("tool_choice"))
    if choice is not None:
        body["tool_choice"] = choice
    # 思考档位下发为 `output_config.effort`（上游 client/configs 逐字下发的写法）。
    effort = payload.get("reasoning_effort")
    if not effort and isinstance(payload.get("reasoning"), dict):
        effort = payload["reasoning"].get("effort")
    if isinstance(effort, str) and effort and (
            allowed_efforts is None or effort in allowed_efforts):
        body["output_config"] = {"effort": effort}
    return body
