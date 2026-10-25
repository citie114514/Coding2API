"""MiniMax Code（中国版）上游客户端。

移植自 free-llm-router `src/providers/minimax.ts` + `minimax-credits.ts`。
协议要点：

- 推理端点是 **Anthropic Messages**（`agent.minimax.cn/mavis/api/v1/llm/v1/messages`），
  请求头**刻意不加 `anthropic-version`**（实测只带 Authorization 即 200，
  照官方文档补那个头是猜测）。
- **账号域与业务域不同 host 不可混用**：OAuth 挂 `account.minimax.cn`，
  业务挂 `agent.minimax.cn`。
- **业务码在 `base_resp.status_code` 且 HTTP 状态码不可信**（实测
  `invalid timezone_id` 也是 HTTP 200）—— 业务判据只能信 body。
- 余额 = Σ `details[].remaining_amount`（**字符串** `"800.00"`）；
  `total_count` 是记录**条数**不是余额（生产数据推翻过的误读，勿改回去）。
- 签到幂等判据是响应体 `claim_result`（1=真领取、2=已领过）；
  `points` 是总数（已含 `bonus_points`），**不得相加**（相加虚高一倍）。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ..zcode import events as anthropic_sse          # Anthropic SSE → 中立 Event（共用）
from ...engine.sse import iter_frames
from ...provider import base
from ...provider.base import CheckinResult, ErrKind, Event, Model, Quota
from ...provider.proxy import build_client
from . import compat
from .credential import MinimaxCredential

API_HOST = "https://agent.minimax.cn"
ACCOUNT_HOST = "https://account.minimax.cn"
EP_INFER = "/mavis/api/v1/llm/v1/messages"
EP_MODELS = "/mavis/api/v1/models"
EP_SIGNIN_CLAIM = "/minimax-cloud/api/v1/signin/claim"
EP_CREDIT_DETAILS = "/minimax-cloud/api/v1/credit/details"
EP_TOKEN = "/oauth2/token"

REGION = "cn"
BUILD_ENV = "prod"
DEFAULT_TIMEZONE_ID = "Asia/Shanghai"
CLIENT_ID = "mcode-public"
AUDIENCE = "agent-backend"
SCOPE = "agent.default"

STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
SHORT_TIMEOUT = httpx.Timeout(30.0)

# 兜底模型目录（含 M3.1-Flash-Preview —— 它不在客户端内置静态表里，但正是
# 官方界面上被选中的模型）。只有 M3.1-Flash-Preview 有档位，其余不编。
FALLBACK_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "MiniMax-M3.1-Flash-Preview", "name": "M3.1-Flash-Preview",
     "context_window": 1_000_000, "max_output": 128_000, "supports_image": True,
     "effort_options": ["default", "low", "medium", "high", "xhigh", "max"],
     "default_effort": "default", "thinking_mode": "forced_on"},
    {"id": "MiniMax-M3", "name": "M3", "context_window": 1_000_000,
     "max_output": 128_000, "supports_image": True, "thinking_mode": "switchable"},
    {"id": "MiniMax-M2.7-highspeed", "name": "M2.7-highspeed",
     "context_window": 200_000, "max_output": 128_000, "supports_image": False,
     "thinking_mode": "forced_on"},
    {"id": "MiniMax-M2.7", "name": "M2.7", "context_window": 200_000,
     "max_output": 128_000, "supports_image": False, "thinking_mode": "forced_on"},
)

# 额度 / 限流文案标记（中英双通道；上游额度不足不一定 402，业务错误走
# HTTP 200 + base_resp，聊天路径之外只有文案兜底可用）
CREDIT_MARKERS = ("insufficient", "payment required", "积分不足", "额度不足", "余额不足")
RATE_MARKERS = ("rate limit", "too many requests", "频率限制", "超出限制")
QUOTA_MARKERS = ("quota", "balance", "credit", "exceeded")


def _classify_status(status: int, body: bytes) -> ErrKind:
    text = body.decode("utf-8", errors="replace").lower()
    if status == 401:
        return ErrKind.DEAD
    if status == 402 or any(marker in text for marker in CREDIT_MARKERS):
        return ErrKind.CREDIT
    if status == 429 or any(marker in text for marker in RATE_MARKERS):
        return ErrKind.SOFT
    if any(marker in text for marker in QUOTA_MARKERS):
        return ErrKind.CREDIT
    if status >= 500:
        return ErrKind.SOFT
    if status == 400:
        # Anthropic 形状的错误体；消息里的 (2013) 是请求/配对错误
        return ErrKind.INVALID
    return ErrKind.OTHER


class UpstreamHTTPError(base.UpstreamHTTPError):
    """MiniMax 上游非 2xx。"""

    classify_status = staticmethod(_classify_status)


def _business_headers(token: str) -> dict[str, str]:
    """业务端点只需 Bearer（实测签到/积分/目录均如此）。"""
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _infer_headers(token: str) -> dict[str, str]:
    """推理端点头（Anthropic Messages）。刻意不加 anthropic-version（实测 200）。"""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
            "Accept": "text/event-stream"}


def _model_from_entry(entry: dict[str, Any]) -> Model:
    return Model(
        id=entry["id"], name=entry.get("name") or entry["id"],
        max_input_tokens=entry.get("context_window") or None,
        max_output_tokens=entry.get("max_output") or None,
        supports_images=entry.get("supports_image"),
        supports_reasoning=bool(entry.get("effort_options")) or None,
        default_effort=entry.get("default_effort"),
    )


def _entry_of(entries: tuple[dict[str, Any], ...], model_id: str) -> dict[str, Any] | None:
    return next((e for e in entries if e["id"] == model_id), None)


def _normalize_remote_model(raw: Any, model_id: str | None = None) -> dict[str, Any] | None:
    """归一远端条目（复刻 asar parseModel）。"""
    if not isinstance(raw, dict):
        return None
    entry_id = model_id or _read_str(raw, ("id",)) or _read_str(raw, ("name",))
    if not entry_id:
        return None
    name = _read_str(raw, ("name",)) or entry_id
    limit = raw.get("limit") if isinstance(raw.get("limit"), dict) else {}
    options = [v for v in raw.get("context_window_options", [])
               if isinstance(v, int) and not isinstance(v, bool) and v > 0] \
        if isinstance(raw.get("context_window_options"), list) else []
    # 窗口：档位表最大档优先（M3.1 的 limit.context=512K 但档位表到 1M）
    context = max(options) if options else limit.get("context")
    context = context if isinstance(context, int) and context > 0 else 0
    modalities = raw.get("modalities") if isinstance(raw.get("modalities"), dict) else {}
    inputs = modalities.get("input") if isinstance(modalities.get("input"), list) else []
    effort_options = [v for v in raw.get("effort_options", [])
                      if isinstance(v, str) and v.strip()] \
        if isinstance(raw.get("effort_options"), list) else []
    raw_default = _read_str(raw, ("default_effort",))
    default_effort = raw_default if raw_default in effort_options else None
    thinking = raw.get("thinking_config")
    thinking_mode = _read_str(thinking, ("mode",)) if isinstance(thinking, dict) else None
    return {"id": entry_id, "name": name, "context_window": context,
            "max_output": limit.get("output") if isinstance(limit.get("output"), int)
            and limit.get("output") > 0 else 0,
            "supports_image": "image" in inputs,
            "effort_options": effort_options, "default_effort": default_effort,
            "thinking_mode": thinking_mode}


def _read_str(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_remote_models(payload: Any) -> list[dict[str, Any]]:
    """远端目录响应 → 模型条目（models 是**对象**，key 即长名）。"""
    if not isinstance(payload, dict):
        return []
    providers = payload.get("providers")
    if not isinstance(providers, list):
        return []
    minimax = next((p for p in providers if isinstance(p, dict)
                    and _read_str(p, ("providerId",)) == "minimax"), None)
    if minimax is None:
        return []
    config = minimax.get("config")
    if not isinstance(config, dict):
        return []
    models = config.get("models")
    if not isinstance(models, dict):
        return []
    out = [entry for entry in (_normalize_remote_model(raw, model_id)
                               for model_id, raw in models.items())
           if entry is not None]
    order = config.get("model_order")
    if isinstance(order, list):
        index = {mid: i for i, mid in enumerate(order) if isinstance(mid, str)}
        out.sort(key=lambda e: index.get(e["id"], len(order)))
    return out


@dataclass(slots=True)
class MinimaxClient:
    api_host: str = API_HOST
    account_host: str = ACCOUNT_HOST
    _stream_client: httpx.AsyncClient | None = None
    _short_client: httpx.AsyncClient | None = None
    proxy: str | None = None

    def __post_init__(self) -> None:
        self.api_host = self.api_host.rstrip("/")
        self.account_host = self.account_host.rstrip("/")

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

    async def stream_chat(self, credential: MinimaxCredential, payload: dict[str, Any],
                          model: str) -> AsyncIterator[Event]:
        """POST Anthropic Messages 并产出中立 Event（SSE 解析复用 zcode 的共用层）。"""
        entry = _entry_of(FALLBACK_MODELS, model)
        body = compat.build_messages_body(payload, model, entry)
        async with self._stream().stream(
            "POST", f"{self.api_host}{EP_INFER}",
            headers=_infer_headers(credential.access_token), json=body,
        ) as response:
            if response.status_code >= 400:
                raw = await base.read_body_bounded(response)
                raise UpstreamHTTPError(response.status_code, raw,
                                        **base.retry_hints(response.headers))
            state = anthropic_sse.StreamState()
            async for frame in iter_frames(response.aiter_bytes()):
                for event in anthropic_sse.parse_frame(frame, state):
                    yield event

    async def fetch_models(self, credential: MinimaxCredential) -> list[Model]:
        """远端目录优先（必须带 region/buildEnv），0 条或失败回退兜底表。"""
        url = (f"{self.api_host}{EP_MODELS}?region={REGION}&buildEnv={BUILD_ENV}")
        try:
            response = await self._short().get(
                url, headers=_business_headers(credential.access_token))
        except httpx.HTTPError:
            return [_model_from_entry(e) for e in FALLBACK_MODELS]
        if response.status_code >= 400:
            return [_model_from_entry(e) for e in FALLBACK_MODELS]
        try:
            entries = parse_remote_models(response.json())
        except ValueError:
            entries = []
        source = entries or list(FALLBACK_MODELS)
        return [_model_from_entry(e) for e in source]

    async def _request_envelope(self, url: str, credential: MinimaxCredential,
                                *, method: str = "GET",
                                json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        """业务请求 + 拆信封。业务失败抛 UpstreamHTTPError（body 带业务码）。"""
        try:
            response = await self._short().request(
                method, url, headers=_business_headers(credential.access_token),
                json=json_body)
        except httpx.HTTPError as error:
            raise base.UpstreamTransportError(str(error)) from error
        if response.status_code in (401, 403):
            raise UpstreamHTTPError(response.status_code, response.content,
                                    **base.retry_hints(response.headers))
        try:
            record = response.json()
        except ValueError as error:
            raise UpstreamHTTPError(response.status_code, response.content[:2048],
                                    **base.retry_hints(response.headers)) from error
        if not isinstance(record, dict):
            raise UpstreamHTTPError(response.status_code, response.content[:2048])
        base_resp = record.get("base_resp")
        base_resp = base_resp if isinstance(base_resp, dict) else {}
        status_code = base_resp.get("status_code")
        status_msg = base_resp.get("status_msg") if isinstance(base_resp.get("status_msg"), str) else ""
        # ⚠️ 业务码在 base_resp.status_code 且 HTTP 状态不可信（200 也会带业务错误）
        if isinstance(status_code, int) and status_code != 0:
            raise UpstreamHTTPError(200, json.dumps(
                {"base_resp": base_resp}, ensure_ascii=False).encode())
        _ = status_msg
        return record

    async def probe_quota(self, credential: MinimaxCredential) -> Quota:
        """积分明细 → Quota。余额 = Σ details[].remaining_amount。"""
        probed_at = int(time.time())
        record = await self._request_envelope(
            f"{self.api_host}{EP_CREDIT_DETAILS}", credential)
        data = record.get("data") if isinstance(record.get("data"), dict) else record
        details = data.get("details")
        if not isinstance(details, list) and not isinstance(data.get("total_count"), int):
            raise UpstreamHTTPError(200, b'{"error":"minimax balance shape unexpected"}')
        packages: list[dict[str, Any]] = []
        expiry_ladder: list[tuple[int, float]] = []
        remaining = 0.0
        earliest: int | None = None

        def loose(value: Any) -> float | None:
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

        for item in details or []:
            if not isinstance(item, dict):
                continue
            amount = loose(item.get("remaining_amount"))
            if amount is None:
                continue
            remaining += amount
            expiry = None
            raw_expiry = item.get("expire_at_ms")
            if isinstance(raw_expiry, (int, float)) and not isinstance(raw_expiry, bool) \
                    and raw_expiry > 0:
                expiry = int(raw_expiry / 1000) if raw_expiry >= 1e12 else int(raw_expiry)
            # 明细名不编造：credit_type 语义未实测，用序号占位
            packages.append({"name": f"积分包 {len(packages) + 1}", "total": amount,
                             "used": 0.0, "end": expiry})
            if expiry is not None:
                expiry_ladder.append((expiry, amount))
                if earliest is None or expiry < earliest:
                    earliest = expiry
        return Quota(remaining=remaining, total=remaining or None, cycle_end=earliest,
                     expiry_ladder=expiry_ladder or None, packages=packages or None,
                     probed_at=probed_at)

    async def checkin(self, credential: MinimaxCredential) -> CheckinResult:
        """每日签到。timezone_id 是 query 参数且必填；幂等判据 claim_result。"""
        timezone_id = credential.timezone_id or DEFAULT_TIMEZONE_ID
        record = await self._request_envelope(
            f"{self.api_host}{EP_SIGNIN_CLAIM}?timezone_id={timezone_id}",
            credential, method="POST", json_body={})
        data = record.get("data") if isinstance(record.get("data"), dict) else record
        claim_result = data.get("claim_result")
        if claim_result == 2:
            return CheckinResult(ok=True, already_checked_in=True, credit=0,
                                 message="今日已签到（服务端判定为重复领取）")
        if claim_result != 1:
            # 判据不明确时不虚报成功
            raise UpstreamHTTPError(200, json.dumps(
                {"error": f"invalid claim_result {claim_result!r}"},
                ensure_ascii=False).encode())

        def loose(value: Any) -> float | None:
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

        # points 是总数（已含 bonus_points），不相加
        gained = loose(data.get("points")) or loose(data.get("bonus_points")) or 0.0
        return CheckinResult(ok=True, credit=gained,
                             message=f"签到成功，获得 {gained:g} 积分（时区 {timezone_id}）")

    async def refresh(self, credential: MinimaxCredential) -> MinimaxCredential:
        """refresh_token grant 续期（x-www-form-urlencoded，非 JSON）。

        invalid_grant / 400 / 401 是终态（refresh_token 本身失效）→
        UpstreamReloginRequired；传输层失败可重试。
        """
        if not credential.refresh_token:
            raise base.UpstreamReloginRequired(
                "MiniMax 凭据缺少 refresh_token，无法自动续期，请重新登录")
        try:
            response = await self._short().post(
                f"{self.account_host}{EP_TOKEN}",
                headers={"Accept": "application/json",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "refresh_token", "refresh_token": credential.refresh_token,
                      "client_id": CLIENT_ID, "scope": SCOPE, "audience": AUDIENCE})
        except httpx.HTTPError as error:
            raise base.UpstreamTransportError(str(error)) from error
        if response.status_code in (400, 401):
            raise base.UpstreamReloginRequired(
                "MiniMax 的 refresh_token 已失效，请重新登录")
        if response.status_code >= 400:
            raise UpstreamHTTPError(response.status_code, response.content,
                                    **base.retry_hints(response.headers))
        try:
            record = response.json()
        except ValueError as error:
            raise base.UpstreamReloginRequired(
                "MiniMax 续期响应不是 JSON，请重新登录") from error
        if not isinstance(record, dict) or not (isinstance(record.get("access_token"), str)
                                                and record["access_token"]):
            raise base.UpstreamReloginRequired(
                "MiniMax 续期响应缺少 access_token，请重新登录")
        expires_in = record.get("expires_in")
        expires_at = (int(time.time() + expires_in)
                      if isinstance(expires_in, (int, float))
                      and not isinstance(expires_in, bool) and expires_in > 0
                      else credential.expires_at)
        fresh = MinimaxCredential(
            access_token=record["access_token"],
            refresh_token=record.get("refresh_token") if isinstance(
                record.get("refresh_token"), str) and record.get("refresh_token")
            else credential.refresh_token,
            expires_at=expires_at, uid=credential.uid, nickname=credential.nickname,
            scope=credential.scope, timezone_id=credential.timezone_id)
        return credential.merge_refreshed(fresh)


@dataclass(slots=True)
class MinimaxProvider:
    """Provider 协议实现（细接口）。refresh 走 refresh_token grant。"""

    client: MinimaxClient
    pacer: Any | None = None

    id: str = "minimax"

    def import_credential(self, raw: dict) -> dict:
        from .credential import credential_from_raw
        return credential_from_raw(raw).to_dict()

    def credential_from(self, data: dict) -> MinimaxCredential:
        return MinimaxCredential.from_dict(data)

    def classify(self, status: int, body: bytes) -> ErrKind:
        return _classify_status(status, body)

    async def probe_quota(self, credential_data: dict) -> Quota:
        return await self.client.probe_quota(self.credential_from(credential_data))

    async def list_models(self, credential_data: dict) -> list[Model]:
        return await self.client.fetch_models(self.credential_from(credential_data))

    async def stream_chat(self, credential_data: dict, payload: dict,
                          model: str) -> AsyncIterator[Event]:
        if self.pacer is not None:
            await self.pacer.wait_turn("minimax")
        try:
            async for event in self.client.stream_chat(
                    self.credential_from(credential_data), payload, model):
                yield event
        finally:
            if self.pacer is not None:
                self.pacer.release("minimax")

    async def checkin(self, credential_data: dict) -> CheckinResult:
        return await self.client.checkin(self.credential_from(credential_data))
    def checkin_scope(self, credential_data: dict) -> str:
        """签到隔离键：渠道|uid（uid 缺失返回空串，调用方回落凭证 ID）。"""
        uid = self.credential_from(credential_data).uid
        return f"minimax|{uid}" if uid else ""

    async def refresh(self, credential_data: dict) -> dict:
        """refresh_token grant 续期（RefreshTask 调用约定：async → 新 dict）。"""
        credential = self.credential_from(credential_data)
        fresh = await self.client.refresh(credential)
        return fresh.to_dict()

    async def aclose(self) -> None:
        await self.client.aclose()
