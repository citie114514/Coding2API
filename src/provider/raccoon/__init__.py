"""Raccoon（商汤小浣熊）渠道：OpenAI 协议 + 微信扫码登录 + 积分池。

移植自 free-llm-router 的 `src/providers/raccoon.ts`（+ raccoon-oauth /
raccoon-credits）。扩展分支新增渠道之一。无每日签到（上游产品形态：
每日积分由服务端自动发放），有一次性桌面端登录奖励。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ...provider.base import CheckinResult, ErrKind, Event, Model, Quota
from .auth import RaccoonOAuth
from .client import RaccoonClient, UpstreamHTTPError
from .credential import RaccoonCredential, credential_from_raw


@dataclass(slots=True)
class RaccoonProvider:
    """Provider 协议实现（细接口）。无 checkin（上游无签到端点，如实缺席）。"""

    client: RaccoonClient = field(default_factory=RaccoonClient)
    pacer: Any | None = None

    id: str = "raccoon"

    def import_credential(self, raw: dict) -> dict:
        return credential_from_raw(raw).to_dict()

    def _cred(self, data: dict) -> RaccoonCredential:
        return RaccoonCredential.from_dict(data)

    def classify(self, status: int, body: bytes) -> ErrKind:
        error = UpstreamHTTPError(status, body)
        return error.kind()

    async def probe_quota(self, credential_data: dict) -> Quota:
        return await self.client.probe_quota(self._cred(credential_data))

    async def list_models(self, credential_data: dict) -> list[Model]:
        return await self.client.fetch_models(self._cred(credential_data))

    async def stream_chat(self, credential_data: dict, payload: dict,
                          model: str) -> AsyncIterator[Event]:
        if self.pacer is not None:
            await self.pacer.wait_turn("raccoon")
        try:
            async for event in self.client.stream_chat(
                    self._cred(credential_data), payload, model):
                yield event
        finally:
            if self.pacer is not None:
                self.pacer.release("raccoon")

    async def refresh(self, credential_data: dict) -> dict:
        fresh = await self.client.refresh(self._cred(credential_data))
        return fresh.to_dict()

    async def aclose(self) -> None:
        await self.client.aclose()


__all__ = [
    "RaccoonOAuth",
    "RaccoonClient",
    "RaccoonProvider",
    "RaccoonCredential",
    "credential_from_raw",
    "UpstreamHTTPError",
]
