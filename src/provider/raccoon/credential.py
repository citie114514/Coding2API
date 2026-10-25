"""Raccoon（商汤小浣熊）凭据。

移植自 free-llm-router `src/providers/raccoon.ts`。口径要点：

- `uid` 必须对同一账号恒定：`user_id` → JWT `sub` → token 派生（MD5 过渡值）。
  **绝不用 access_token 本身**（每 3 小时续期就换一个，拿它当主键会让同一
  账号在池里反复新增）。
- 扫码登录成功后服务端**只下发 token**（无 user_id）⇒ 解析时派生 uid 不抛错，
  登录后由 OAuth 层立即调 `/user_info` 把稳定 id 补实。
- `expires_at` 缺失时回退 JWT `exp`（只读显式字段会让续期永不触发，
  凭据悄悄过期且无任何报错 —— 参考实现记录过的同型缺陷）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

API_BASE = "https://xiaohuanxiong.com"
AUTH_PREFIX = "/api/web/auth/v1"
LLM_PREFIX = "/api/web/llm/v2"
POINTS_PREFIX = "/api/web/points/v1"
DESKTOP_PREFIX = "/api/web/desktop/v1"
CLIENT_PLATFORM = "desktop-windows"
CLIENT_VERSION = "v1.0.35"


def decode_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) < 2 or not parts[1]:
        return None
    padded = parts[1].replace("-", "+").replace("_", "/")
    padded += "=" * ((4 - len(padded) % 4) % 4)
    try:
        parsed = json.loads(base64.b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def jwt_exp_epoch(token: str) -> int | None:
    """JWT `exp` → epoch 秒（只解码不验签；失败返回 None）。"""
    payload = decode_jwt_payload(token)
    if payload is None:
        return None
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and not isinstance(exp, bool) and exp > 0:
        return int(exp) if exp < 1e12 else int(exp / 1000)
    return None


def derive_uid(access_token: str, phone: str) -> str:
    """token 派生过渡 uid（取 JWT payload 段而非整条 token：签名段每续期都变）。"""
    parts = access_token.split(".")
    material = f"{phone}\x00{parts[1] if len(parts) >= 2 and parts[1] else access_token}"
    digest = hashlib.md5(material.encode()).hexdigest()
    return f"raccoon-{digest[:16]}"


@dataclass(slots=True)
class RaccoonCredential:
    access_token: str
    refresh_token: str = ""
    expires_at: int | None = None          # epoch 秒
    uid: str = ""
    nickname: str = ""
    org_code: str = ""                     # 个人账号为空串；请求头总是发送
    phone: str = ""
    user_id: str = ""
    device_id: str = ""

    @property
    def token_expires_at(self) -> int | None:
        return self.expires_at

    def needs_refresh(self, skew_seconds: int = 0) -> bool:
        if not self.refresh_token:
            return False
        if self.expires_at is None:
            return False
        return self.expires_at - skew_seconds <= int(time.time())

    def to_dict(self) -> dict[str, Any]:
        return {"access_token": self.access_token, "refresh_token": self.refresh_token,
                "expires_at": self.expires_at, "uid": self.uid, "nickname": self.nickname,
                "org_code": self.org_code, "phone": self.phone, "user_id": self.user_id,
                "device_id": self.device_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RaccoonCredential:
        expires = data.get("expires_at")
        return cls(
            access_token=str(data.get("access_token") or ""),
            refresh_token=str(data.get("refresh_token") or ""),
            expires_at=expires if isinstance(expires, int) else None,
            uid=str(data.get("uid") or ""),
            nickname=str(data.get("nickname") or ""),
            org_code=str(data.get("org_code") or ""),
            phone=str(data.get("phone") or ""),
            user_id=str(data.get("user_id") or ""),
            device_id=str(data.get("device_id") or ""),
        )

    def merge_refreshed(self, fresh: RaccoonCredential) -> RaccoonCredential:
        """续期合并：服务端可能只回新 access_token —— refresh_token 与附加字段保留旧值。"""
        self.access_token = fresh.access_token
        if fresh.refresh_token:
            self.refresh_token = fresh.refresh_token
        if fresh.expires_at is not None:
            self.expires_at = fresh.expires_at
        return self


def _pick(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def credential_from_raw(raw: Any) -> RaccoonCredential:
    """解析 Raccoon 凭据（严格判别：JWT 形态或有 refresh_token/user_id/phone）。"""
    if not isinstance(raw, dict):
        raise ValueError("Raccoon 凭据必须是一个 JSON 对象")
    source = dict(raw)
    for wrapper in ("credential", "credentials", "auth"):
        inner = source.get(wrapper)
        if isinstance(inner, dict):
            source = {**source, **inner}
            break
    access = _pick(source, "access_token", "accessToken", "token")
    if not access:
        raise ValueError("缺少访问令牌：请提供 Raccoon 的 `access_token` 字段")
    refresh = _pick(source, "refresh_token", "refreshToken")
    payload = decode_jwt_payload(access) or {}
    sub = payload.get("sub") if isinstance(payload.get("sub"), str) else ""
    user_id = _pick(source, "user_id", "userId")
    phone = _pick(source, "phone", "mobile")
    looks_like_jwt = len(access.split(".")) >= 3
    if not looks_like_jwt and not refresh and not user_id and not phone and not sub:
        raise ValueError(
            "这不像是 Raccoon 凭据：`access_token` 不是 JWT（三段 base64url），"
            "且没有 `refresh_token` / `user_id` / `phone` 可用于识别账号")
    uid = user_id or sub or derive_uid(access, phone)
    expires_raw = _pick(source, "expires_at", "expiresAt")
    expires_at: int | None = None
    if expires_raw:
        try:
            value = float(expires_raw)
            expires_at = int(value / 1000) if value >= 1e12 else int(value)
        except ValueError:
            expires_at = None
    if expires_at is None:
        expires_at = jwt_exp_epoch(access)
    nickname = _pick(source, "nickname", "name")
    if not nickname:
        nickname = f"Raccoon {phone}" if phone else f"Raccoon {uid}"
    return RaccoonCredential(
        access_token=access, refresh_token=refresh, expires_at=expires_at, uid=uid,
        nickname=nickname, org_code=_pick(source, "office_identity", "officeIdentity", "org_code"),
        phone=phone, user_id=user_id, device_id=_pick(source, "device_id", "deviceId"))
