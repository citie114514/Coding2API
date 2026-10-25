"""Raccoon（商汤小浣熊）上游客户端：OpenAI 协议 + 积分池。

移植自 free-llm-router `src/providers/raccoon.ts`（+ raccoon-oauth /
raccoon-credits）。要点：

- 上游是 **OpenAI chat/completions** 协议（SSE），帧解析复用 qoder 的共用层。
- 业务失败**不一定**返回 HTTP 4xx：也可能是 200 + 非 0 `code` 信封 ——
  信封解析时状态码只在**没有** `code` 字段时兜底。
- 余额只读 `GET /points/v1/balance`；**绝不调 `desktop/v1/login/points/grant`**
  当查询用 —— 那是写端点（一次性奖励），高频路径上调用会意外消费掉机会。
- 各积分池分开作 packages（每日/奖励/会员/充值的有效期与回补规则都不同）。
- 每日 300 积分由服务端按日自动发放（账单 `biz_type: daily_grant`），
  **没有可调用的签到端点** —— 不实现 checkin 是如实声明，不是缺失。
- `refresh_token` 是**一次性轮换**的：续期时服务端只回新 access_token 的话
  必须保留旧值；绝不能用真实凭据调 /refresh 做任何「验证」。
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ...engine.sse import iter_frames
from ...provider import base
from ...provider.base import ErrKind, Event, Model, Quota
from ...provider.proxy import build_client
from .credential import (
    API_BASE,
    AUTH_PREFIX,
    CLIENT_PLATFORM,
    CLIENT_VERSION,
    DESKTOP_PREFIX,
    LLM_PREFIX,
    POINTS_PREFIX,
    RaccoonCredential,
)

STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
SHORT_TIMEOUT = httpx.Timeout(30.0)

# 业务码 → 不换号集合：100003（手机号加密错）/ 100006（滑块失败）/ 200035
# 是请求本身的问题，换号会重放同样的坏请求并放大风控
_NO_ROTATE_CODES = frozenset({100003, 100006, 200035})
# refresh_token 已失效的业务码（终态）
_RELOGIN_CODES = frozenset({200003})


class UpstreamHTTPError(base.UpstreamHTTPError):
    """Raccoon 上游错误（HTTP 层或信封业务码统一走这里）。"""

    def __init__(self, status: int, body: bytes, *, code: int | None = None, **kwargs) -> None:
        super().__init__(status, body, **kwargs)
        self.code = code

    def kind(self) -> ErrKind:
        if self.code is not None:
            if self.code in _RELOGIN_CODES:
                return ErrKind.DEAD
            text = self.body.decode("utf-8", errors="replace").lower()
            if any(marker in text for marker in ("insufficient", "积分不足", "余额不足")):
                return ErrKind.CREDIT
            if self.code in (100003, 100006, 200035):
                return ErrKind.REQUEST
            return ErrKind.OTHER
        return _classify_status(self.status, self.body)


def _classify_status(status: int, body: bytes) -> ErrKind:
    if status in (401, 403):
        return ErrKind.DEAD                    # token 寿命 ~3h，换号/续期处置
    if status == 402:
        return ErrKind.CREDIT
    if status == 429:
        return ErrKind.SOFT
    if status >= 500:
        return ErrKind.SOFT
    if status == 400:
        return ErrKind.INVALID
    return ErrKind.OTHER


def _headers(credential: RaccoonCredential, *, platform: str = CLIENT_PLATFORM) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {credential.access_token}",
        "X-Org-Code": credential.org_code,     # 个人账号为空串；客户端总是发送
        "X-Raccoon-Language": "zh",
        "X-Client-Version": CLIENT_VERSION,
    }
    if platform:
        headers["X-Client-Platform"] = platform
    if credential.device_id:
        headers["X-Client-Device-ID"] = credential.device_id
    return headers


def _format_multiplier(value: float) -> str:
    return str(float(f"{value:.4f}"))


def display_name(model_id: str, description: str, effective: float, base_mult: float) -> str:
    """展示名：`GLM-5-3 · x0.75` / `· 免费` / `· x0.2→x0.1`。

    ⚠️ 1 倍也要显示（否则用户无法区分「它就是 1 倍」与「没取到倍率」）。
    """
    name = description or model_id
    if effective != effective or effective < 0:            # NaN / 负数
        return name
    if effective == 0:
        return f"{name} · 免费"
    if base_mult > 0 and base_mult > effective:
        return f"{name} · x{_format_multiplier(base_mult)}→x{_format_multiplier(effective)}"
    return f"{name} · x{_format_multiplier(effective)}"


def parse_models_payload(payload: Any) -> list[Model]:
    """`GET /model_catalog` → Model 列表（只取 categories[type=chat]，滤 visible=false）。"""
    if not isinstance(payload, dict) or payload.get("code") != 0:
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    categories = data.get("categories")
    chat = None
    for category in categories if isinstance(categories, list) else []:
        if isinstance(category, dict) and category.get("type") == "chat":
            chat = category
            break
    if chat is None:
        return []
    models = chat.get("models")
    out: list[Model] = []
    seen: set[str] = set()
    for entry in models if isinstance(models, list) else []:
        if not isinstance(entry, dict):
            continue
        # ⚠️ 远端把模型 id 放在 `name` 字段里（不是 `id`）
        model_id = entry.get("name")
        if not (isinstance(model_id, str) and model_id.strip()) or model_id in seen:
            continue
        model_id = model_id.strip()
        if entry.get("visible") is False:
            continue
        seen.add(model_id)
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        description = entry.get("description")
        description = description if isinstance(description, str) and description else model_id
        effective = entry.get("billing_effective_multiplier")
        effective = effective if isinstance(effective, (int, float)) and not isinstance(
            effective, bool) else float("nan")
        base_mult = entry.get("billing_multiplier")
        base_mult = base_mult if isinstance(base_mult, (int, float)) and not isinstance(
            base_mult, bool) else float("nan")
        tags = [t.lower() for t in entry.get("tags", [])
                if isinstance(t, str)] if isinstance(entry.get("tags"), list) else []
        context = params.get("context_window") or entry.get("context_window")
        context = context if isinstance(context, int) and context > 0 else None
        max_tokens = params.get("max_tokens")
        max_tokens = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None
        out.append(Model(
            id=model_id,
            name=display_name(model_id, description, effective, base_mult),
            max_input_tokens=context,
            max_output_tokens=max_tokens,
            supports_images=any(t in tags for t in ("vision", "image", "image-understanding")),
            # 判据用生效价 === 0（实际计价口径）；limited_free 是限免、有时限
            credit_rate=0.0 if effective == 0 else None,
        ))
    return out


def _cleanup_tool_pairing(messages: list[Any]) -> list[Any]:
    """丢弃孤儿 `tool` 消息与名称为空的 tool_calls（上游 400 的两个来源）。"""
    declared: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]:
                declared.add(call["id"])
    out: list[Any] = []
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not (isinstance(call_id, str) and call_id and call_id in declared):
                changed = True
                continue
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            kept = [call for call in message["tool_calls"]
                    if isinstance(call, dict) and isinstance(call.get("function"), dict)
                    and isinstance(call["function"].get("name"), str) and call["function"]["name"]]
            if len(kept) != len(message["tool_calls"]):
                changed = True
                if kept:
                    message["tool_calls"] = kept
                else:
                    message.pop("tool_calls", None)
        out.append(message)
    return out if changed else messages


def prepare_body(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """出站请求体：model 去前缀、强制 stream、清理工具配对、思考通道映射。

    ⚠️ 唯一**有效**的思考控制通道是 `extra_body.thinking`
    （`adaptive` / `enabled` / `disabled`）。`reasoning_effort` 虽被上游接受
    但**无任何可观测效果**，映射后必须删掉（留着让人误以为档位生效了）。
    """
    body = dict(payload)
    body["messages"] = copy.deepcopy(payload.get("messages"))
    body["model"] = model
    body["stream"] = True
    if isinstance(body.get("messages"), list):
        body["messages"] = _cleanup_tool_pairing(body["messages"])
    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and "extra_body" not in body:
        thinking_type = "disabled" if effort in ("none", "off", "disabled") else "enabled"
        body["extra_body"] = {"thinking": {"type": thinking_type}}
    body.pop("reasoning_effort", None)
    return body


@dataclass(slots=True)
class RaccoonClient:
    api_base: str = API_BASE
    _stream_client: httpx.AsyncClient | None = None
    _short_client: httpx.AsyncClient | None = None
    proxy: str | None = None

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")

    def _stream(self) -> httpx.AsyncClient:
        if self._stream_client is None:
            self._stream_client = build_client(timeout=STREAM_TIMEOUT, proxy=self.proxy)
        return self._stream_client

    def _short(self) -> httpx.AsyncClient:
        if self._short_client is None:
            self._short_client = build_client(timeout=SHORT_TIMEOUT, proxy=self.proxy)
        return self._short_client

    async def aclose(self) -> None:
        for client in (self._stream_client, self._short_client):
            if client is not None:
                await client.aclose()

    async def _request(self, credential: RaccoonCredential, path: str, *,
                       method: str = "GET", json_body: dict[str, Any] | None = None,
                       platform: str | None = None) -> dict[str, Any]:
        """带凭证 JSON 请求 + 信封校验；`code != 0` 抛 UpstreamHTTPError。"""
        headers = _headers(credential) if platform is None else _headers(
            credential, platform=platform)
        try:
            response = await self._short().request(
                method, f"{self.api_base}{path}", headers=headers, json=json_body)
        except httpx.HTTPError as error:
            raise base.UpstreamTransportError(str(error)) from error
        try:
            parsed = response.json()
        except ValueError as error:
            raise UpstreamHTTPError(response.status_code,
                                    response.content[:2048]) from error
        if not isinstance(parsed, dict):
            raise UpstreamHTTPError(response.status_code, response.content[:2048])
        code = parsed.get("code")
        # ⚠️ 失败可能带 4xx 也可能是 200 + 非 0 code：状态码只在无 code 字段时兜底
        code_num = code if isinstance(code, int) else (response.status_code
                                                       if response.status_code >= 400 else 0)
        if code_num != 0:
            raise UpstreamHTTPError(response.status_code,
                                    json.dumps(parsed, ensure_ascii=False)[:2048].encode(),
                                    code=code_num,
                                    **base.retry_hints(response.headers))
        return parsed

    async def stream_chat(self, credential: RaccoonCredential, payload: dict[str, Any],
                          model: str) -> AsyncIterator[Event]:
        from ..qoder.events import parse_inner_chunk          # OpenAI 帧 → 中立事件（共用）
        body = prepare_body(payload, model)
        headers = _headers(credential)
        headers["Accept"] = "text/event-stream"
        async with self._stream().stream(
            "POST", f"{self.api_base}{LLM_PREFIX}/chat/completions",
            headers=headers, json=body,
        ) as response:
            if response.status_code >= 400:
                raw = await base.read_body_bounded(response)
                raise UpstreamHTTPError(response.status_code, raw,
                                        **base.retry_hints(response.headers))
            async for frame in iter_frames(response.aiter_bytes()):
                if not frame.data:
                    continue
                for event in parse_inner_chunk(frame.data):
                    yield event

    async def fetch_models(self, credential: RaccoonCredential) -> list[Model]:
        parsed = await self._request(credential, f"{LLM_PREFIX}/model_catalog")
        models = parse_models_payload(parsed)
        if not models:
            # 空目录显式失败，不返回空数组冒充「账号没有模型」
            raise UpstreamHTTPError(200, b'{"error":"raccoon model catalog empty"}')
        return models

    async def probe_quota(self, credential: RaccoonCredential) -> Quota:
        """积分余额。⚠️ 只读 /points/v1/balance，绝不碰 login/points/grant（写端点）。"""
        probed_at = int(time.time())
        parsed = await self._request(credential, f"{POINTS_PREFIX}/balance")
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
        available = data.get("available_points")
        if not (isinstance(available, (int, float)) and not isinstance(available, bool)):
            # 核心字段缺失 ⇒ 形状不对；返回 0 会把「查不到」伪装成「余额为 0」
            raise UpstreamHTTPError(200, b'{"error":"raccoon balance missing available_points"}')

        def num(key: str) -> float | None:
            value = data.get(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(
                value, bool) else None

        packages: list[dict[str, Any]] = []
        reward = num("reward_points")
        daily = num("daily_points")
        monthly = num("monthly_points")
        topup = num("topup_points")
        if reward is not None:
            packages.append({"name": "奖励积分", "total": reward, "used": 0.0, "end": None})
        if daily is not None:
            packages.append({"name": "每日积分（每日刷新）", "total": daily, "used": 0.0,
                             "end": None})
        if monthly is not None and monthly > 0:
            packages.append({"name": "会员积分", "total": monthly, "used": 0.0, "end": None})
        if topup is not None:
            packages.append({"name": "充值积分", "total": topup, "used": 0.0, "end": None})
        if not packages:
            packages.append({"name": "可用积分", "total": float(available), "used": 0.0,
                             "end": None})
        # ⚠️ 不声称任何过期信息：服务端只给四个池的余额，不返回到期字段
        return Quota(remaining=float(available), total=float(available), cycle_end=None,
                     packages=packages, probed_at=probed_at)

    async def enrich(self, credential: RaccoonCredential) -> RaccoonCredential:
        """登录后补 user_id/phone/昵称（失败原样返回，不让已成功的登录失败）。"""
        try:
            parsed = await self._request(credential, f"{AUTH_PREFIX}/user_info")
        except (UpstreamHTTPError, base.UpstreamTransportError):
            return credential
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
        user_id = data.get("id")
        user_id = user_id if isinstance(user_id, str) and user_id else ""
        name = data.get("name")
        name = name if isinstance(name, str) and name else ""
        office = data.get("office_identity")
        office = office if isinstance(office, str) else ""
        phone = data.get("phone")
        phone = phone if isinstance(phone, str) else ""
        if user_id:
            credential.uid = user_id                # 服务端 id 比 JWT sub 更稳定
        if name:
            credential.nickname = name
        if office:
            credential.org_code = office
        if phone:
            credential.phone = phone
        if user_id:
            credential.user_id = user_id
        return credential

    async def refresh(self, credential: RaccoonCredential) -> RaccoonCredential:
        """refresh_token 换新凭据（一次性轮换；服务端只回新 access 时保留旧值）。"""
        if not credential.refresh_token:
            raise base.UpstreamReloginRequired("凭据缺少 refresh_token，请重新登录")
        try:
            response = await self._short().post(
                f"{self.api_base}{AUTH_PREFIX}/refresh",
                headers={"Content-Type": "application/json"},
                json={"refresh_token": credential.refresh_token})
        except httpx.HTTPError as error:
            raise base.UpstreamTransportError(str(error)) from error
        try:
            parsed = response.json()
        except ValueError as error:
            raise UpstreamHTTPError(response.status_code, response.content[:2048]) from error
        code = parsed.get("code") if isinstance(parsed, dict) else None
        if response.status_code == 401 or code == 200003:
            raise base.UpstreamReloginRequired(
                "Raccoon 登录态已过期（refresh_token 已失效），请重新登录（微信扫码）")
        if not isinstance(parsed, dict) or (isinstance(code, int) and code != 0):
            raise UpstreamHTTPError(response.status_code,
                                    json.dumps(parsed, ensure_ascii=False)[:2048].encode(),
                                    code=code if isinstance(code, int) else None)
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
        access = data.get("access_token")
        if not (isinstance(access, str) and access):
            raise UpstreamHTTPError(200, b'{"error":"raccoon refresh missing access_token"}')
        fresh = RaccoonCredential(
            access_token=access,
            refresh_token=data.get("refresh_token") if isinstance(
                data.get("refresh_token"), str) and data.get("refresh_token")
            else credential.refresh_token,
            expires_at=None,
            uid=credential.uid, nickname=credential.nickname, org_code=credential.org_code,
            phone=credential.phone, user_id=credential.user_id,
            device_id=credential.device_id)
        from .credential import jwt_exp_epoch
        fresh.expires_at = jwt_exp_epoch(access)
        return credential.merge_refreshed(fresh)

    async def grant_login_reward(self, credential: RaccoonCredential) -> tuple[bool, float, str]:
        """一次性桌面端登录奖励（幂等；不是每日签到）。返回 (granted, points, detail)。"""
        parsed = await self._request(credential, f"{DESKTOP_PREFIX}/login/points/grant",
                                     method="POST")
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
        if data.get("granted") is not True:
            return False, 0.0, "该账号已领取过桌面端登录奖励（每号一次，幂等命中）"
        popup = data.get("popup") if isinstance(data.get("popup"), dict) else {}
        points = popup.get("points")
        points = float(points) if isinstance(points, (int, float)) and not isinstance(
            points, bool) else 3000.0
        return True, points, f"已领取桌面端登录奖励 {points:g} 积分"
