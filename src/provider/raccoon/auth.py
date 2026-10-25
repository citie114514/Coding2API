"""Raccoon 微信扫码登录（poll 轨道）。

移植自 free-llm-router `src/raccoon-oauth.ts`。核心事实：

- 扫码的 `qrcode_code` 由**客户端自造**（16 字节随机 hex），服务端只做轮询
  查询 —— 任意自造值都被接受，完全绕开那条收不到的
  `office-raccoon://auth/callback` 自定义协议。
- 二维码承载的是**公开登录页** URL：`{base}/login/mp?code=…&appname=商汤小浣熊官网`。
  用户微信扫码 → 微信内授权 → 服务端把该 code 置 success → 轮询取回 token。
- 二维码**不在服务端渲染**：AuthSession.auth_url 就是那个 URL，前端对
  raccoon 渠道用浏览器端 QR 库把 auth_url 画出来（并保留链接兜底）。
- 轮询的**任何异常都降级为 pending**（网络抖动/响应畸形/未知 status）：
  把未知状态误判成 success 会拿到空 token 卡死，误判成 canceled 会让
  用户正在扫的二维码被无故刷新。
- 成功后**必须**立即调 /user_info 补 user_id（扫码成功只下发 token，
  JWT sub 只是过渡 uid —— 见 credential.py 的口径注释）。
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
from .client import RaccoonClient
from .credential import API_BASE, AUTH_PREFIX, credential_from_raw

AUTH_STATE_TTL_SECONDS = 10 * 60
# 上游扫码状态（实测枚举）
STATUS_PENDING = "pending"
STATUS_LOGGING = "logging"
STATUS_CANCELED = "canceled"
STATUS_SUCCESS = "success"


def _encode_state(code: str, exp: int) -> str:
    return base64.urlsafe_b64encode(
        json.dumps({"code": code, "exp": exp}, separators=(",", ":")).encode()).decode().rstrip("=")


def _decode_state(state: str) -> dict[str, Any]:
    padded = state + "=" * ((4 - len(state) % 4) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError) as error:
        raise UpstreamProtocolViolation("unknown or consumed auth state") from error
    if not isinstance(data, dict) or not data.get("code"):
        raise UpstreamProtocolViolation("unknown or consumed auth state")
    return data


class _StateStore:
    def cancel(self, state: str, username: str) -> bool:
        try:
            _decode_state(state)
        except UpstreamProtocolViolation:
            return False
        return True


class RaccoonOAuth:
    """微信扫码登录。`client` 可注入（测试）。"""

    def __init__(self, *, client: RaccoonClient | None = None,
                 http: httpx.AsyncClient | None = None,
                 ttl_seconds: int = AUTH_STATE_TTL_SECONDS,
                 proxy: str | None = None) -> None:
        self.store = _StateStore()
        self._ttl = ttl_seconds
        self._login_client = http
        self._proxy = proxy
        self._client = client or RaccoonClient(proxy=proxy)

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._login_client is None:
            self._login_client = build_client(timeout=httpx.Timeout(30.0), proxy=self._proxy)
        return self._login_client

    async def aclose(self) -> None:
        if self._login_client is not None:
            await self._login_client.aclose()
        await self._client.aclose()

    async def start(self, username: str, force_login: bool = False) -> AuthSession:
        """生成扫码 code 与登录页 URL（纯本地计算，不碰网络）。"""
        code = secrets.token_hex(16)
        from urllib.parse import quote
        qr_url = (f"{API_BASE}/login/mp?code={code}"
                  f"&appname={quote('商汤小浣熊官网')}")
        state = _encode_state(code, int(time.time()) + self._ttl)
        return AuthSession(flow=AuthFlow.POLL, state=state, auth_url=qr_url,
                           interval=2, callback_url=None)

    async def poll(self, auth_state: str, username: str) -> AuthResult | None:
        """轮询一次扫码状态；成功后补 user_info 再入库。"""
        data = _decode_state(auth_state)
        if int(data.get("exp") or 0) < int(time.time()):
            raise UpstreamProtocolViolation(
                "raccoon auth state expired",
                public="登录二维码已过期，请重新发起登录")
        code = str(data["code"])
        status, payload = await self._poll_once(code)
        if status != STATUS_SUCCESS:
            return None
        token_payload = payload if isinstance(payload, dict) else {}
        access = token_payload.get("access_token")
        if not (isinstance(access, str) and access):
            return None                      # 缺 token 的 success 视为未完成
        credential = credential_from_raw({
            "access_token": access,
            "refresh_token": token_payload.get("refresh_token") if isinstance(
                token_payload.get("refresh_token"), str) else "",
            "office_identity": token_payload.get("office_identity") if isinstance(
                token_payload.get("office_identity"), str) else "",
        })
        credential = await self._client.enrich(credential)
        return AuthResult(credential_data=credential.to_dict(), nickname=credential.nickname)

    async def _poll_once(self, code: str) -> tuple[str, Any]:
        """POST login_with_qrcode_code → (status, data)。任何异常降级 pending。"""
        try:
            response = await self._http.post(
                f"{API_BASE}{AUTH_PREFIX}/login_with_qrcode_code",
                headers={"Content-Type": "application/json"},
                json={"qrcode_code": code})
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return STATUS_PENDING, None
        if not isinstance(payload, dict):
            return STATUS_PENDING, None
        envelope_code = payload.get("code")
        data = payload.get("data")
        if envelope_code != 0 or not isinstance(data, dict):
            return STATUS_PENDING, None
        status = data.get("status")
        if status == STATUS_CANCELED:
            raise UpstreamProtocolViolation(
                "raccoon qr canceled",
                public="扫码登录已被取消，请重新发起登录")
        if status == STATUS_LOGGING:
            return STATUS_LOGGING, None      # 已扫码，等待微信内确认
        if status == STATUS_SUCCESS:
            return STATUS_SUCCESS, data
        return STATUS_PENDING, None
