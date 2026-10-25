"""MiniMax Code（中国版）渠道：Anthropic 协议 + OAuth 设备码 + 积分签到。

移植自 free-llm-router 的 `src/providers/minimax.ts`（+ minimax-oauth /
minimax-messages / minimax-credits）。扩展分支新增渠道之一。
"""

from __future__ import annotations

from .auth import MinimaxOAuth
from .client import MinimaxClient, MinimaxProvider, UpstreamHTTPError
from .credential import MinimaxCredential, credential_from_raw

__all__ = [
    "MinimaxOAuth",
    "MinimaxClient",
    "MinimaxProvider",
    "MinimaxCredential",
    "credential_from_raw",
    "UpstreamHTTPError",
]
