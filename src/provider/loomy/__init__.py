"""Loomy（讯飞）渠道：OpenAI 协议 + 短信验证码登录 + 每日额度。

移植自 free-llm-router 的 `src/providers/loomy.ts`。扩展分支新增渠道之一。
无 refresh 端点（session 14 天，过期只能重新登录，如实缺席 refresh）。
短信登录走管理台专用端点（send/verify 两步），不走通用 poll 轨道。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ...provider.base import CheckinResult, ErrKind, Event, Model, Quota
from .auth import LoomySmsLogin
from .client import LoomyClient, UpstreamHTTPError
from .credential import LoomyCredential, credential_from_raw, looks_like_loomy


@dataclass(slots=True)
class LoomyProvider:
    """Provider 协议实现（细接口）。无 refresh（上游无续期端点，如实缺席）。"""

    client: LoomyClient = field(default_factory=LoomyClient)
    pacer: Any | None = None

    id: str = "loomy"

    def import_credential(self, raw: dict) -> dict:
        return credential_from_raw(raw).to_dict()

    def _cred(self, data: dict) -> LoomyCredential:
        return LoomyCredential.from_dict(data)

    def classify(self, status: int, body: bytes) -> ErrKind:
        return UpstreamHTTPError(status, body).kind()

    async def probe_quota(self, credential_data: dict) -> Quota:
        return await self.client.probe_quota(self._cred(credential_data))

    async def list_models(self, credential_data: dict) -> list[Model]:
        return await self.client.fetch_models(self._cred(credential_data))

    async def stream_chat(self, credential_data: dict, payload: dict,
                          model: str) -> AsyncIterator[Event]:
        if self.pacer is not None:
            await self.pacer.wait_turn("loomy")
        try:
            async for event in self.client.stream_chat(
                    self._cred(credential_data), payload, model):
                yield event
        finally:
            if self.pacer is not None:
                self.pacer.release("loomy")

    async def checkin(self, credential_data: dict) -> CheckinResult:
        return await self.client.checkin(self._cred(credential_data))

    async def aclose(self) -> None:
        await self.client.aclose()


__all__ = [
    "LoomySmsLogin",
    "LoomyClient",
    "LoomyProvider",
    "LoomyCredential",
    "credential_from_raw",
    "looks_like_loomy",
    "UpstreamHTTPError",
]
