"""Loomy 短信验证码登录（三步交互流的辅助层）。

短信路径是纯 HTTP 三步（发码 → 用户输入 → 校验），不需要本地监听 ——
但它不是「start 后轮询」的轨道，而是**中途需要用户输入**。本模块只提供
无状态的两个动作（send_sms / verify），会话编排由管理台的专用端点承担：

1. `POST /api/auth/upstream/loomy/sms/send {phone}`
   → `{state, message}`（state 自包含 phone+msgid+过期）；
2. `POST /api/auth/upstream/loomy/sms/verify {state, code}`
   → 与 poll 轨道相同的 `{status:'success', credential_id}` 落库响应。

state 编码：base64url(JSON{phone, msgid, exp})，自包含、无服务端会话。
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from ...provider.base import AuthResult, UpstreamProtocolViolation
from .client import LoomyClient
from .credential import credential_from_login

AUTH_STATE_TTL_SECONDS = 10 * 60


def _encode_state(phone: str, msgid: str, exp: int) -> str:
    return base64.urlsafe_b64encode(
        json.dumps({"phone": phone, "msgid": msgid, "exp": exp},
                   separators=(",", ":")).encode()).decode().rstrip("=")


def _decode_state(state: str) -> dict[str, Any]:
    padded = state + "=" * ((4 - len(state) % 4) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError) as error:
        raise UpstreamProtocolViolation("unknown or consumed loomy sms state") from error
    if not isinstance(data, dict) or not data.get("phone") or not data.get("msgid"):
        raise UpstreamProtocolViolation("unknown or consumed loomy sms state")
    return data


class LoomySmsLogin:
    """短信登录动作集合。`client` 可注入（测试）。"""

    def __init__(self, client: LoomyClient, *, ttl_seconds: int = AUTH_STATE_TTL_SECONDS) -> None:
        self.client = client
        self._ttl = ttl_seconds

    async def send_sms(self, phone: str) -> tuple[str, str]:
        """发码 → (state, 提示语)。msgid 藏进 state，不落服务端会话。"""
        msgid = await self.client.send_sms_code(phone)
        state = _encode_state(phone, msgid, int(time.time()) + self._ttl)
        return state, "验证码已发送，请查收短信（5 分钟内有效）"

    async def verify(self, state: str, code: str) -> AuthResult:
        """校验验证码 → 凭据（一次性：state 里没有可消费的会话，重放只会因
        msgid 已被上游消费而失败）。"""
        data = _decode_state(state)
        if int(data.get("exp") or 0) < int(time.time()):
            raise UpstreamProtocolViolation(
                "loomy sms state expired",
                public="短信验证会话已过期，请重新发送验证码")
        credential = await self.client.login_by_sms(
            str(data["phone"]), code, str(data["msgid"]))
        return AuthResult(credential_data=credential.to_dict(),
                          nickname=credential.nickname)
