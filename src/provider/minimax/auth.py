"""MiniMax OAuth 设备码 + PKCE 登录（poll 轨道）。

移植自 free-llm-router `src/minimax-oauth.ts`。三个实测判据（写错会得到
「用户还没来得及点授权就报失败」）：

1. **`pending` 有两种形态**：HTTP 200 + `status:"pending"`（MiniMax 自有）
   与 非 200 + `error:"authorization_pending"`（OAuth 标准）—— 两种都要认；
2. PKCE 是 **S256**（`code_challenge` = base64url(SHA256(verifier))）；
3. `slow_down` 的退避是 **+5 秒且跨请求累积**（与 Cline 的 +1 不同）。

state 自包含（device_code + code_verifier + 过期时间 + 当前间隔），
无服务端会话存储；slow_down 的累积值存在进程内 dict（重启回落到服务端
基准间隔，可接受）。
"""

from __future__ import annotations

import base64
import hashlib
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
from .client import ACCOUNT_HOST, EP_TOKEN, AUDIENCE, CLIENT_ID, SCOPE
from .credential import credential_from_token_grant

AUTH_STATE_TTL_SECONDS = 15 * 60


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _encode_state(payload: dict[str, Any]) -> str:
    return _b64url(json.dumps(payload, separators=(",", ":")).encode())


def _decode_state(state: str) -> dict[str, Any]:
    padded = state + "=" * ((4 - len(state) % 4) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError) as error:
        raise UpstreamProtocolViolation("unknown or consumed auth state") from error
    if not isinstance(data, dict) or not data.get("dc") or not data.get("cv"):
        raise UpstreamProtocolViolation("unknown or consumed auth state")
    return data


class _StateStore:
    """cancel 端点要求；state 自包含、无上游可撤销语义。"""

    def __init__(self) -> None:
        self.intervals: dict[str, int] = {}

    def cancel(self, state: str, username: str) -> bool:
        try:
            _decode_state(state)
        except UpstreamProtocolViolation:
            return False
        self.intervals.pop(state, None)
        return True


class MinimaxOAuth:
    """设备码 + PKCE 登录。`client` 可注入（测试）。"""

    def __init__(self, *, account_host: str = ACCOUNT_HOST,
                 client: httpx.AsyncClient | None = None,
                 ttl_seconds: int = AUTH_STATE_TTL_SECONDS,
                 proxy: str | None = None) -> None:
        self.account_host = account_host.rstrip("/")
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
        """申请设备码（POST form-urlencoded 到账号域）。"""
        verifier_bytes = secrets.token_bytes(32)
        code_verifier = _b64url(verifier_bytes)
        code_challenge = _b64url(hashlib.sha256(code_verifier.encode()).digest())
        try:
            response = await self._http.post(
                f"{self.account_host}/oauth2/device/code",
                headers={"Accept": "application/json",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"client_id": CLIENT_ID, "scope": SCOPE, "audience": AUDIENCE,
                      "code_challenge": code_challenge, "code_challenge_method": "S256"})
        except httpx.HTTPError as error:
            raise UpstreamProtocolViolation(
                "minimax device code request failed",
                public="MiniMax 设备码申请失败（网络错误），请稍后重试") from error
        if response.status_code >= 400:
            raise UpstreamProtocolViolation(
                f"minimax device code http {response.status_code}",
                public=f"MiniMax 设备码申请失败（HTTP {response.status_code}）")
        try:
            payload = response.json()
        except ValueError as error:
            raise UpstreamProtocolViolation("minimax device code non-JSON") from error
        if not isinstance(payload, dict):
            raise UpstreamProtocolViolation("minimax device code non-object")

        def mstr(key: str) -> str:
            value = payload.get(key)
            return value if isinstance(value, str) and value else ""

        device_code = mstr("device_code")
        user_code = mstr("user_code")
        verification_uri = mstr("verification_uri") or mstr("verification_url")
        expires_in = payload.get("expires_in")
        if not device_code or not user_code or not verification_uri \
                or not (isinstance(expires_in, (int, float))
                        and not isinstance(expires_in, bool) and expires_in > 0):
            raise UpstreamProtocolViolation(
                "minimax device code missing fields",
                public="MiniMax 设备码响应缺少必要字段，请重试")
        interval = payload.get("interval")
        interval = interval if isinstance(interval, int) and interval > 0 else 5
        complete = mstr("verification_uri_complete") or verification_uri
        state = _encode_state({"dc": device_code, "cv": code_verifier,
                               "exp": int(time.time()) + self._ttl,
                               "iv": interval})
        return AuthSession(flow=AuthFlow.POLL, state=state, auth_url=complete,
                           interval=interval, callback_url=None)

    async def poll(self, auth_state: str, username: str) -> AuthResult | None:
        """轮询一次授权结果（pending 两种形态都认；slow_down +5s 累积）。"""
        data = _decode_state(auth_state)
        if int(data.get("exp") or 0) < int(time.time()):
            raise UpstreamProtocolViolation(
                "minimax auth state expired",
                public="登录会话已过期，请重新发起登录")
        device_code = str(data["dc"])
        code_verifier = str(data["cv"])
        base_interval = int(data.get("iv") or 5)
        interval = self.store.intervals.get(auth_state, base_interval)
        try:
            response = await self._http.post(
                f"{self.account_host}{EP_TOKEN}",
                headers={"Accept": "application/json",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                      "device_code": device_code, "client_id": CLIENT_ID,
                      "code_verifier": code_verifier})
        except httpx.HTTPError:
            return None                       # 网络层失败不判终态
        try:
            record = response.json()
        except ValueError:
            record = {}
        if not isinstance(record, dict):
            record = {}

        def mstr(key: str) -> str:
            value = record.get(key)
            return value if isinstance(value, str) and value else ""

        status = mstr("status")
        error = mstr("error")
        # 形态一：HTTP 200 + status；形态二：非 200 + error
        if response.status_code == 200 and status == "pending":
            return None
        if response.status_code == 200 and status == "slow_down":
            self.store.intervals[auth_state] = interval + 5
            return None
        if response.status_code == 200 and status in ("denied", "access_denied"):
            raise UpstreamProtocolViolation("minimax denied",
                                            public="用户拒绝了授权，请重新发起登录")
        if response.status_code == 200 and status in ("expired", "expired_token"):
            raise UpstreamProtocolViolation("minimax expired",
                                            public="设备码已过期，请重新发起登录")
        if error == "authorization_pending":
            return None
        if error == "slow_down":
            self.store.intervals[auth_state] = interval + 5
            return None
        if error == "access_denied":
            raise UpstreamProtocolViolation("minimax denied",
                                            public="用户拒绝了授权，请重新发起登录")
        if error == "expired_token":
            raise UpstreamProtocolViolation("minimax expired",
                                            public="设备码已过期，请重新发起登录")
        if response.status_code == 200 and mstr("access_token"):
            credential = credential_from_token_grant(record)
            if credential is None:
                raise UpstreamProtocolViolation(
                    "minimax token grant invalid",
                    public="MiniMax 令牌响应无法解析（缺 access_token / scope 不含 "
                           "agent.default）")
            self.store.intervals.pop(auth_state, None)
            return AuthResult(credential_data=credential.to_dict(),
                              nickname=credential.nickname)
        raise UpstreamProtocolViolation(
            f"minimax oauth failed: {error or response.status_code}",
            public=f"MiniMax 授权失败：{error or f'HTTP {response.status_code}'}")
