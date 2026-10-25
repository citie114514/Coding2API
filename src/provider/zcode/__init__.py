"""ZCode（智谱）渠道：Anthropic 协议 + 静态 JWT + 设备码登录。

移植自 free-llm-router（XuexGao/free-llm-router）的 `src/providers/zcode.ts`，
协议事实与踩坑记录见各模块文件头。扩展分支新增渠道之一。
"""

from __future__ import annotations

from .auth import ZcodeOAuth
from .client import UpstreamHTTPError, ZcodeClient, ZcodeProvider
from .credential import ZcodeCredential, credential_from_raw, decode_jwt_payload

__all__ = [
    "ZcodeOAuth",
    "ZcodeClient",
    "ZcodeProvider",
    "ZcodeCredential",
    "credential_from_raw",
    "decode_jwt_payload",
    "UpstreamHTTPError",
]
