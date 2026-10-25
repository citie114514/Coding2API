"""ZCode 设备码登录（poll 轨道）。

移植自 free-llm-router `src/providers/zcode.ts` 的登录流程：

- `POST /api/v1/oauth/cli/init`（Bearer 是**我们自己生成的 CLI 会话密钥**，
  官方同款做法）→ `{flow_id, authorize_url, poll_interval_sec}`；
- 用户在浏览器打开 authorize_url 完成授权；
- `GET /api/v1/oauth/cli/poll/{flow_id}`（Bearer 同一会话密钥）→
  `pending` / `ready`（token + user）/ `failed`。

这条路径**不经任何本地回调**，纯服务端轮询即可走完 —— 完美贴合本服务的
poll 轨道。故意绕开故障的 `POST /api/v1/oauth/token`（自 2026-09-28 起稳定
500/code 2007，token 由轮询直接返回，完全不经过它）。

state 编码：flow_id + flow_secret + 过期时间 base64url(JSON)。poll 端点把
不透明 state 原样传回，自包含即无需服务端会话存储；过期后拒绝继续轮询。
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from typing import Any

import httpx

from ...provider.base import (
    AuthFlow,
    AuthResult,
    AuthSession,
    UpstreamProtocolViolation,
)
from ...provider.proxy import build_client
from .client import APP_VERSION_FALLBACK, ZCODE_HOST, EP_OAUTH_INIT
from .credential import credential_from_login, decode_jwt_payload

AUTH_STATE_TTL_SECONDS = 15 * 60
PENDING_HTTP = frozenset({408, 429})


def _encode_state(flow_id: str, flow_secret: str, exp: int) -> str:
    raw = json.dumps({"fid": flow_id, "fsec": flow_secret, "exp": exp},
                     separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_state(state: str) -> dict[str, Any]:
    padded = state + "=" * ((4 - len(state) % 4) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError) as error:
        raise UpstreamProtocolViolation("unknown or consumed auth state") from error
    if not isinstance(data, dict) or not data.get("fid") or not data.get("fsec"):
        raise UpstreamProtocolViolation("unknown or consumed auth state")
    return data


class _StateStore:
    """cancel 端点要求 `store.cancel(state, username)`；state 自包含、无服务端
    会话可撤 —— 设备码流没有「撤销轮询」的上游语义，返回 True 让前端正常收尾。
    """

    def cancel(self, state: str, username: str) -> bool:
        try:
            _decode_state(state)
        except UpstreamProtocolViolation:
            return False
        return True


class ZcodeOAuth:
    """设备码登录（poll 轨道）。`client` 可注入（测试与 MockTransport）。"""

    def __init__(self, *, host: str = ZCODE_HOST,
                 client: httpx.AsyncClient | None = None,
                 ttl_seconds: int = AUTH_STATE_TTL_SECONDS,
                 proxy: str | None = None) -> None:
        self.host = host.rstrip("/")
        self.store = _StateStore()
        self._ttl = ttl_seconds
        self._client = client
        self._proxy = proxy

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = build_client(timeout=httpx.Timeout(30.0), proxy=self._proxy)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def start(self, username: str, force_login: bool = False) -> AuthSession:
        """发起设备授权流，拿到 authorize_url（state 自包含 flow_id + 会话密钥）。"""
        flow_secret = secrets.token_hex(32)
        headers = {
            "Authorization": f"Bearer {flow_secret}",
            "Content-Type": "application/json",
            "User-Agent": f"ZCode/{APP_VERSION_FALLBACK}",
            "HTTP-Referer": ZCODE_HOST,
            "X-ZCode-App-Version": APP_VERSION_FALLBACK,
            "X-Platform": "win32",
        }
        try:
            response = await self._http.post(
                f"{self.host}{EP_OAUTH_INIT}", headers=headers,
                json={"provider": "bigmodel"})
        except httpx.HTTPError as error:
            raise UpstreamProtocolViolation(
                "zcode authorize init failed",
                public="ZCode 授权初始化失败（网络错误），请稍后重试") from error
        if response.status_code >= 400:
            raise UpstreamProtocolViolation(
                f"zcode authorize init http {response.status_code}: {response.text[:200]}",
                public=f"ZCode 授权初始化失败（HTTP {response.status_code}），请稍后重试")
        try:
            parsed = response.json()
        except ValueError as error:
            raise UpstreamProtocolViolation("zcode authorize init non-JSON response") from error
        data = parsed.get("data") if isinstance(parsed, dict) else None
        data = data if isinstance(data, dict) else {}
        flow_id = data.get("flow_id")
        authorize_url = data.get("authorize_url")
        if not (isinstance(flow_id, str) and flow_id):
            raise UpstreamProtocolViolation("zcode authorize init missing flow_id")
        if not (isinstance(authorize_url, str) and authorize_url.startswith("https://")):
            raise UpstreamProtocolViolation("zcode authorize init missing authorize_url")
        raw_interval = data.get("poll_interval_sec")
        interval = raw_interval if isinstance(raw_interval, int) and raw_interval >= 1 else 2
        state = _encode_state(flow_id, flow_secret, int(time.time()) + self._ttl)
        return AuthSession(flow=AuthFlow.POLL, state=state, auth_url=authorize_url,
                           interval=interval, callback_url=None)

    async def poll(self, auth_state: str, username: str) -> AuthResult | None:
        """返回 None 表示仍在等待授权；成功后 state 即作废（一次性）。"""
        data = _decode_state(auth_state)
        if int(data.get("exp") or 0) < int(time.time()):
            raise UpstreamProtocolViolation(
                "zcode auth state expired",
                public="登录会话已过期，请重新发起登录")
        flow_id = str(data["fid"])
        flow_secret = str(data["fsec"])
        headers = {
            "Authorization": f"Bearer {flow_secret}",
            "User-Agent": f"ZCode/{APP_VERSION_FALLBACK}",
            "HTTP-Referer": ZCODE_HOST,
            "X-ZCode-App-Version": APP_VERSION_FALLBACK,
        }
        url = f"{self.host}/api/v1/oauth/cli/poll/{flow_id}"
        try:
            response = await self._http.get(url, headers=headers)
        except httpx.HTTPError:
            return None                                  # 网络抖动：继续轮询
        if response.status_code in PENDING_HTTP:
            return None
        if 400 <= response.status_code < 500:
            # 4xx（除 408/429）是终态失败；5xx 与网络错误继续重试
            raise UpstreamProtocolViolation(
                f"zcode login poll http {response.status_code}: {response.text[:160]}",
                public=f"ZCode 登录轮询被拒（HTTP {response.status_code}）")
        if response.status_code != 200:
            return None
        try:
            parsed = response.json()
        except ValueError:
            return None
        body_data = parsed.get("data") if isinstance(parsed, dict) else None
        if parsed.get("code") != 0 or not isinstance(body_data, dict):
            return None
        status = body_data.get("status")
        if status == "pending" or status is None:
            return None
        if status == "failed":
            raise UpstreamProtocolViolation(
                "zcode login failed",
                public="用户拒绝了授权或授权失败，请重新发起登录")
        if status != "ready":
            raise UpstreamProtocolViolation(
                f"zcode login poll unknown status {status!r}",
                public="ZCode 轮询响应状态无法识别，请重新发起登录")
        jwt = body_data.get("token")
        if not (isinstance(jwt, str) and jwt):
            raise UpstreamProtocolViolation("zcode login ready but missing token")
        user = body_data.get("user")
        user = user if isinstance(user, dict) else {}
        uid = (user.get("user_id") or user.get("id")) if isinstance(user, dict) else None
        if not (isinstance(uid, str) and uid):
            payload = decode_jwt_payload(jwt)
            uid = payload.get("user_id") or payload.get("sub") if payload else None
        if not (isinstance(uid, str) and uid):
            raise UpstreamProtocolViolation(
                "zcode login missing user id",
                public="ZCode 登录响应里没有账号标识，无法入库，请重新登录")
        display = user.get("name") or user.get("email")
        display = display if isinstance(display, str) and display else uid
        credential = credential_from_login(jwt, uid=uid, nickname=display)
        return AuthResult(credential_data=credential.to_dict(), nickname=credential.nickname)
