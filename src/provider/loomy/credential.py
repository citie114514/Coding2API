"""Loomy（讯飞）凭据：讯飞 session（32 位 hex），14 天，无续期端点。

移植自 free-llm-router `src/providers/loomy.ts`。判别与 uid 口径：

- 严格判别：有 token 字段 **且**（32 位小写 hex session / 15–20 位数字
  userid / 11 位手机号）之一 —— 只看「有 access_token」会把 Raccoon 的
  JWT 凭据也吞下来。
- uid：有 userid 直接用（讯飞侧稳定）；否则从 token 确定性派生
  `loomy-<md5 前 16 hex>`。**绝不用手机号当唯一 uid**（同一手机号可对应
  多个讯飞账号，会合并成一条）；也不直接拿 token 当 uid（凭据本体进
  存储 key 会让日志/面板展示路径泄漏凭据）。
- session 由登录请求里的 `expire` 参数声明有效期（默认 14 天），响应
  **不带**到期时间 —— 到期时刻由本地推算。没有 refresh 端点，过期只能
  重新短信/微信登录。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

SESSION_TTL_SECONDS = 1_209_600            # 14 天
SMS_CODE_TTL_SECONDS = 300

USERID_HINT = re.compile(r"^\d{15,20}$")
SESSION_HINT = re.compile(r"^[0-9a-f]{32}$")
PHONE_HINT = re.compile(r"^1\d{10}$")


def derive_uid(access_token: str, phone: str) -> str:
    digest = hashlib.md5(f"{phone}\x00{access_token}".encode()).hexdigest()
    return f"loomy-{digest[:16]}"


@dataclass(slots=True)
class LoomyCredential:
    access_token: str                       # 讯飞 session（32 位小写 hex）
    userid: str = ""
    phone: str = ""
    nickname: str = ""
    expires_at: int | None = None           # epoch 秒；None = 未知

    @property
    def token_expires_at(self) -> int | None:
        return self.expires_at

    def needs_refresh(self, skew_seconds: int = 0) -> bool:
        """无续期端点：恒 False（过期由 DEAD 分类处置，提示重新登录）。"""
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"access_token": self.access_token, "userid": self.userid,
                "phone": self.phone, "nickname": self.nickname,
                "expires_at": self.expires_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoomyCredential:
        expires = data.get("expires_at")
        return cls(access_token=str(data.get("access_token") or ""),
                   userid=str(data.get("userid") or ""),
                   phone=str(data.get("phone") or ""),
                   nickname=str(data.get("nickname") or ""),
                   expires_at=expires if isinstance(expires, int) else None)

    def merge_refreshed(self, fresh: Any) -> Any:
        """无续期链路；仅为调用形态同形而保留。"""
        return self


def _pick(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    source = dict(raw)
    for wrapper in ("credential", "credentials", "auth"):
        inner = source.get(wrapper)
        if isinstance(inner, dict):
            source = {**source, **inner}
            break
    return source


def looks_like_loomy(raw: Any) -> bool:
    """判别特征（与 parse 共用一份，避免两处判据分叉）。"""
    if not isinstance(raw, dict):
        return False
    source = _flatten(raw)
    access = _pick(source, "access_token", "accessToken", "session", "token")
    if not access:
        return False
    userid = _pick(source, "userid", "user_id", "userId")
    phone = _pick(source, "phone", "mobile", "phone_number")
    return bool(SESSION_HINT.match(access) or USERID_HINT.match(userid)
                or PHONE_HINT.match(phone))


def credential_from_raw(raw: Any) -> LoomyCredential:
    if not isinstance(raw, dict):
        raise ValueError("Loomy 凭据必须是一个 JSON 对象")
    source = _flatten(raw)
    access = _pick(source, "access_token", "accessToken", "session", "token")
    if not access:
        raise ValueError("缺少访问令牌：请提供 Loomy 的 `access_token`（讯飞 session）字段")
    userid = _pick(source, "userid", "user_id", "userId")
    phone = _pick(source, "phone", "mobile", "phone_number")
    if not looks_like_loomy(source):
        raise ValueError(
            "这不像是 Loomy 凭据：`access_token` 不是 32 位十六进制的讯飞 session，"
            "且没有 15–20 位 `userid` 或 11 位手机号可用于识别账号")
    nickname = _pick(source, "nickname", "name", "user_name")
    expires_raw = _pick(source, "expires_at", "expiresAt")
    expires_at: int | None = None
    if expires_raw:
        try:
            value = float(expires_raw)
            expires_at = int(value / 1000) if value >= 1e12 else int(value)
        except ValueError:
            expires_at = None
    uid = userid if USERID_HINT.match(userid) else derive_uid(access, phone)
    if not nickname:
        nickname = f"Loomy {phone}" if phone else f"Loomy {userid or uid}"
    return LoomyCredential(access_token=access, userid=userid, phone=phone,
                           nickname=nickname, expires_at=expires_at)


def credential_from_login(session: str, userid: str, phone: str) -> LoomyCredential:
    """短信/微信登录成功 → 凭据。到期时刻由本地按 14 天推算（服务端不回传）。"""
    return LoomyCredential(access_token=session, userid=userid, phone=phone,
                           nickname=f"Loomy {phone}" if phone else f"Loomy {userid}",
                           expires_at=int(time.time() + SESSION_TTL_SECONDS))
