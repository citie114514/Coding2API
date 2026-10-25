"""ZCode（智谱）上游客户端：Anthropic Messages 协议 + 静态 JWT。

移植自 free-llm-router `src/providers/zcode.ts`（其协议逆向自官方 ZCode CLI）。

关键协议事实（均有上游实测依据，勿「优化」）：

- **凭据静态**：JWT 无 `exp`、无续期端点 → 不实现 refresh；真失效回
  401/1002，由 DEAD 分类处置。
- **`X-Device-Mid` 硬需求**：缺它 billing 全家桶回 `400 code 3001`。
- **模型推理不需要 captcha**（自客户端 3.14.4 起），但**签到 claim 恒索要
  阿里云 captcha**（400/3007，校验前置于 plan 校验）→ 签到会如实报出上游
  拒绝，绝不伪造成功。
- **签到前必须补发 `event/report` 活跃上报**（幂等）：服务端不主动推送活动，
  不补发则 `billing/preview` 恒返回空 `plans` —— 把空列表当「已领取」是
  假结论。
- **每日赠送额度不在 `balances` 桶里**，只在 `plans` 里；`balances: []` 是
  正常态，不是形状异常。
- 首字等待有 20s+ 长尾：流式读超时必须设 None（首字等待由引擎的
  stream_first_event_timeout 兜底），连接超时单独收紧。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...engine.sse import iter_frames
from ...provider import base
from ...provider.base import (
    CheckinResult,
    ErrKind,
    Model,
    Quota,
)
from ...provider.proxy import build_client
from . import anthropic_compat, events as zcode_events
from .credential import ZcodeCredential

ZCODE_HOST = "https://zcode.z.ai"
EP_MESSAGES = "/api/v1/zcode-plan/anthropic/v1/messages"
EP_BALANCE = "/api/v1/zcode-plan/billing/balance"
EP_CLIENT_CONFIGS = "/api/v1/client/configs"
EP_CLAIM = "/api/v1/zcode-plan/billing/claim"
EP_PREVIEW = "/api/v1/zcode-plan/billing/preview"
EP_EVENT_REPORT = "/api/v1/event/report"
EP_OAUTH_INIT = "/api/v1/oauth/cli/init"
EP_OAUTH_POLL = "/api/v1/oauth/cli/poll/{flow_id}"

APP_VERSION_FALLBACK = "3.14.3"

# 流式无总超时防长流截断（上游首字有 20s+ 长尾，read 必须为 None）；
# 短请求 30s 防悬挂。与 kilo/zen 同策略。
STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
SHORT_TIMEOUT = httpx.Timeout(30.0)

# 兜底模型目录（上游 client/configs 的 builtinModels 实测快照）。
# ⚠️ 只放实测可用的两个：GLM-5-Turbo / GLM-5.2 实测返回空响应，列出来
# 比不列更糟。能力字段逐字抄上游（GLM-5.3 的 capabilities 是空对象 ⇒ 无 vision）。
FALLBACK_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "GLM-5.3-Flash", "name": "GLM-5.3-Flash", "context_window": 1_000_000,
     "max_tokens": 128_000, "supports_image": True,
     "reasoning_levels": ["low", "high", "max"], "default_reasoning_level": "max"},
    {"id": "GLM-5.3", "name": "GLM-5.3", "context_window": 1_000_000,
     "max_tokens": 128_000, "supports_image": False,
     "reasoning_levels": ["low", "high", "max"], "default_reasoning_level": "max"},
)


def build_headers(credential: ZcodeCredential, *, authorization: str | None = None,
                  json_body: bool = True) -> dict[str, str]:
    """ZCode 请求头（官方客户端指纹；X-Device-Mid 是硬需求）。"""
    headers = {
        "User-Agent": f"ZCode/{credential.app_version}",
        "HTTP-Referer": ZCODE_HOST,
        "X-ZCode-App-Version": credential.app_version,
        "X-Release-Channel": "stable",
        "X-Client-Language": "zh-CN",
        "X-Client-Timezone": "Asia/Shanghai",
        "X-Device-Mid": credential.device_mid or credential.uid,
        # 官方客户端只在 Windows 上有这条通道，照抄 win32（上游可能按平台分池）
        "X-Platform": "win32",
        "X-Os-Category": "windows",
        "anthropic-version": "2023-06-01",
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    if authorization is not None:
        headers["Authorization"] = authorization
    return headers


class UpstreamHTTPError(base.UpstreamHTTPError):
    """ZCode 上游非 2xx；kind() 走 ZCode 的状态码/业务码规则。"""

    classify_status = staticmethod(zcode_events.classify_status)


def _model_from_entry(entry: dict[str, Any]) -> Model:
    return Model(
        id=entry["id"],
        name=entry.get("name") or entry["id"],
        max_input_tokens=entry.get("context_window"),
        max_output_tokens=entry.get("max_tokens"),
        supports_images=entry.get("supports_image"),
        supports_reasoning=bool(entry.get("reasoning_levels")) or None,
        default_effort=entry.get("default_reasoning_level"),
    )


@dataclass(slots=True)
class ZcodeClient:
    host: str = ZCODE_HOST
    _stream_client: httpx.AsyncClient | None = None
    _short_client: httpx.AsyncClient | None = None
    proxy: str | None = None

    def __post_init__(self) -> None:
        self.host = self.host.rstrip("/")

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

    async def stream_chat(self, credential: ZcodeCredential, payload: dict[str, Any],
                          model: str) -> AsyncIterator[Event]:
        """POST Anthropic Messages 并逐事件产出中立 Event。非 2xx 抛 UpstreamHTTPError。"""
        entry = next((m for m in FALLBACK_MODELS if m["id"] == model), None)
        body = anthropic_compat.build_messages_body(
            payload, model, allowed_efforts=entry and entry.get("reasoning_levels"))
        headers = build_headers(credential, authorization=f"Bearer {credential.jwt}")
        state = zcode_events.StreamState()
        async with self._stream().stream(
            "POST", f"{self.host}{EP_MESSAGES}", json=body, headers=headers,
        ) as response:
            if response.status_code >= 400:
                raw = await base.read_body_bounded(response)
                raise UpstreamHTTPError(response.status_code, raw,
                                        **base.retry_hints(response.headers))
            async for frame in iter_frames(response.aiter_bytes()):
                for event in zcode_events.parse_frame(frame, state):
                    yield event

    async def fetch_models(self, credential: ZcodeCredential) -> list[Model]:
        """拉模型目录（远端 client/configs，失败回退兜底表）。"""
        url = (f"{self.host}{EP_CLIENT_CONFIGS}"
               f"?app_version={credential.app_version}&platform=unknown")
        try:
            response = await self._short().get(
                url, headers=build_headers(credential, json_body=False,
                                           authorization=f"Bearer {credential.jwt}"))
        except httpx.HTTPError:
            return [_model_from_entry(entry) for entry in FALLBACK_MODELS]
        if response.status_code >= 400:
            return [_model_from_entry(entry) for entry in FALLBACK_MODELS]
        try:
            data = response.json()
        except ValueError:
            return [_model_from_entry(entry) for entry in FALLBACK_MODELS]
        entries = zcode_events.parse_models_payload(data.get("data") if
                                                    isinstance(data, dict) else None)
        if not entries:
            return [_model_from_entry(entry) for entry in FALLBACK_MODELS]
        return [_model_from_entry(entry) for entry in entries]

    async def probe_quota(self, credential: ZcodeCredential) -> Quota:
        """查余额（billing/balance）→ Quota。

        ⚠️ 响应可能没有 `data` 包裹（实测顶层形态），顶层与 data 取并集；
        `balances: []` 是正常态 —— 每日赠送额度只在 `plans` 里，桶与 plans
        都为空时如实返回 0，只有「连字段都没有」才算探测失败。
        """
        probed_at = int(time.time())
        try:
            response = await self._short().get(
                f"{self.host}{EP_BALANCE}",
                headers=build_headers(credential, json_body=False,
                                      authorization=f"Bearer {credential.jwt}"))
        except httpx.HTTPError as error:
            raise UpstreamTransportErrorZcode(str(error)) from error
        if response.status_code >= 400:
            raise UpstreamHTTPError(response.status_code,
                                    await base.read_body_bounded(response),
                                    **base.retry_hints(response.headers))
        try:
            parsed = response.json()
        except ValueError as error:
            raise zcode_events.protocol_violation("balance response is not JSON") from error
        if not isinstance(parsed, dict):
            raise zcode_events.protocol_violation("balance response is not an object")
        nested = parsed.get("data")
        nested = nested if isinstance(nested, dict) else {}
        display_mode = parsed.get("displayMode") or nested.get("displayMode")
        balances = parsed.get("balances") if parsed.get("balances") is not None \
            else nested.get("balances")
        plans = parsed.get("plans") if parsed.get("plans") is not None else nested.get("plans")
        if balances is None and plans is None:
            raise zcode_events.protocol_violation("balance response missing balances/plans")
        if display_mode == "enterprise":
            # 企业版不下发额度数字、只给外部链接 ⇒ 探测失败（不是「已耗尽」）
            return Quota(probe_failed=True, probed_at=probed_at)

        def num(value: Any) -> float | None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return None

        packages: list[dict[str, Any]] = []
        expiry_ladder: list[tuple[int, float]] = []
        total = 0.0
        remaining = 0.0
        earliest_expiry: int | None = None
        raw_buckets = balances if isinstance(balances, list) else []
        for item in raw_buckets:
            if not isinstance(item, dict):
                continue
            show_name = item.get("show_name")
            show_name = show_name if isinstance(show_name, str) and show_name else "ZCode 额度"
            unit = item.get("unit_type")
            unit = unit if isinstance(unit, str) and unit else "token"
            amount = num(item.get("available_units"))
            if amount is None:
                amount = num(item.get("remaining_units")) or 0.0
            bucket_total = num(item.get("total_units")) or 0.0
            expires_at = num(item.get("expires_at"))
            # 上游到期时间是 Unix 秒 → epoch 秒（本服务统一口径）
            expiry = int(expires_at) if expires_at and expires_at > 0 else None
            packages.append({"name": f"{show_name}（{unit}）", "total": bucket_total,
                             "used": max(0.0, bucket_total - amount), "end": expiry})
            remaining += amount
            total += bucket_total
            if expiry is not None:
                expiry_ladder.append((expiry, amount))
                if earliest_expiry is None or expiry < earliest_expiry:
                    earliest_expiry = expiry
        if not packages:
            # `balances: []` 是正常态；顺带把 plans 里的可领额度算进展示
            raw_plans = plans if isinstance(plans, list) else []
            for item in raw_plans:
                if not isinstance(item, dict):
                    continue
                plan_id = item.get("plan_id")
                if not (isinstance(plan_id, str) and plan_id):
                    continue
                name = item.get("name")
                name = name if isinstance(name, str) and name else "ZCode 每日额度"
                amount = num(item.get("amount"))
                label = f"可领：{name}（{amount:g}）" if amount is not None else f"可领：{name}"
                packages.append({"name": label, "total": 0.0, "used": 0.0, "end": None})
            if not packages:
                packages.append({"name": "额度桶为空（当前没有可用额度，也没有可领活动）",
                                 "total": 0.0, "used": 0.0, "end": None})
        return Quota(remaining=remaining, total=total or None, cycle_end=earliest_expiry,
                     expiry_ladder=expiry_ladder or None, packages=packages,
                     probed_at=probed_at)

    async def checkin(self, credential: ZcodeCredential) -> CheckinResult:
        """每日签到：补活跃上报 → 查 preview → 尝试 claim。

        ⚠️ 诚实契约：claim 恒索要阿里云 captcha（400/3007）—— 如实把上游
        原话报出来，绝不伪造成功。preview（不需要 captcha）能正常给出
        「有没有可领的」。
        """
        headers = build_headers(credential)
        # ⓪ 补发客户端活跃上报（幂等）：不做这步，preview 永远是空列表。
        # 上报按 device_mid + 日期去重；失败不阻塞（下次再补）。
        for event in ("app_launch", "app_daily_active"):
            try:
                await self._short().post(
                    f"{self.host}{EP_EVENT_REPORT}",
                    headers=headers,
                    json={"event": event, "device_mid": credential.device_mid or credential.uid,
                          "platform": "win32", "app_version": credential.app_version},
                )
            except httpx.HTTPError:
                continue
        # ① 查可领取计划（不需要 captcha；必须带 app_version + platform 查询参数）
        claimable: list[str] = []
        try:
            preview = await self._short().get(
                f"{self.host}{EP_PREVIEW}?app_version={credential.app_version}&platform=win32",
                headers=build_headers(credential, json_body=False))
            if preview.status_code < 400:
                body = preview.json() if preview.content else {}
                data = body.get("data") if isinstance(body, dict) else None
                data = data if isinstance(data, dict) else {}
                plans = data.get("plans")
                if isinstance(plans, list):
                    claimable = [plan["plan_id"] for plan in plans
                                 if isinstance(plan, dict)
                                 and isinstance(plan.get("plan_id"), str) and plan["plan_id"]]
        except (httpx.HTTPError, ValueError):
            pass
        if not claimable:
            # 正常状态（今天已领过 / 没有活动），不是错误
            return CheckinResult(ok=True, already_checked_in=True, credit=0,
                                 message="没有可领取的每日积分（可能今天已领取）")
        # ② 尝试领取。上游若哪天放宽 captcha 校验，这里自然开始工作。
        response = await self._short().post(
            f"{self.host}{EP_CLAIM}", headers=headers, json={"plan_id": claimable[0]})
        text = response.text if response.content else ""
        code: int | None = None
        msg = ""
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                raw_code = parsed.get("code")
                if isinstance(raw_code, int):
                    code = raw_code
                elif isinstance(raw_code, str) and raw_code.lstrip("-").isdigit():
                    code = int(raw_code)
                raw_msg = parsed.get("msg") or parsed.get("message")
                if isinstance(raw_msg, str):
                    msg = raw_msg
        except ValueError:
            pass
        if code == 0:
            return CheckinResult(ok=True, credit=0,
                                 message=f"签到成功（{len(claimable)} 个待领计划）")
        raise UpstreamHTTPError(response.status_code, response.content,
                                **base.retry_hints(response.headers))


class UpstreamTransportErrorZcode(base.UpstreamTransportError):
    """ZCode 传输层失败（SOFT：链路瞬态，不累计错误数）。"""


@dataclass(slots=True)
class ZcodeProvider:
    """Provider 协议实现（细接口）。

    凭证生命周期：import_credential（JSON 导入）+ start_auth/poll_auth（设备码
    登录由 main.py 挂的 ZcodeOAuth 承接 poll 轨道，这里只留导入）。无 refresh。
    """

    client: ZcodeClient = field(default_factory=ZcodeClient)
    pacer: Any | None = None

    id: str = "zcode"

    def import_credential(self, raw: dict) -> dict:
        return credential_data_from_raw(raw)

    def classify(self, status: int, body: bytes) -> ErrKind:
        return zcode_events.classify_status(status, body)

    async def probe_quota(self, credential_data: dict) -> Quota:
        return await self.client.probe_quota(credential_from(credential_data))

    async def list_models(self, credential_data: dict) -> list[Model]:
        return await self.client.fetch_models(credential_from(credential_data))

    async def stream_chat(self, credential_data: dict, payload: dict,
                          model: str) -> AsyncIterator[Event]:
        if self.pacer is not None:
            await self.pacer.wait_turn("zcode")
        try:
            async for event in self.client.stream_chat(
                    credential_from(credential_data), payload, model):
                yield event
        finally:
            if self.pacer is not None:
                self.pacer.release("zcode")

    async def checkin(self, credential_data: dict) -> CheckinResult:
        return await self.client.checkin(credential_from(credential_data))

    async def aclose(self) -> None:
        await self.client.aclose()


def credential_from(data: dict) -> ZcodeCredential:
    return ZcodeCredential.from_dict(data)


def credential_data_from_raw(raw: dict) -> dict:
    from .credential import credential_from_raw
    return credential_from_raw(raw).to_dict()
