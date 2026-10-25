"""MiniMax Code（中国版）凭据。

移植自 free-llm-router `src/providers/minimax.ts`。协议事实：

- **access_token 不是 JWT**（前缀 `mmoat_`、60 字符、0 个点），从 JWT 解
  `exp` 恒失败 —— 过期时间用 `expires_in` 自算或显式字段，JWT 分支仅兜底。
- token 响应硬校验：`token_type == bearer`（大小写不敏感）+ `scope` 含
  `agent.default`，否则视为无效响应。
- `uid` 兜底链：account_id → JWT sub → 令牌派生稳定键（对同一 token 恒同值，
  重复轮询不会堆出重复账号）。
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any

CLIENT_ID = "mcode-public"
AUDIENCE = "agent-backend"
SCOPE = "agent.default"


def _stable_key(secret: str) -> str:
    """由令牌派生稳定本地主键（仅兜底，不参与鉴权）。"""
    a = 0x811C9DC5
    b = 0x01000193
    for ch in secret:
        code = ord(ch) & 0xFFFFFFFF
        a = ((a ^ code) * 0x01000193) & 0xFFFFFFFF
        b = ((b ^ code) * 0x811C9DC5) & 0xFFFFFFFF
    return f"minimax-{a:08x}{b:08x}"


def _parse_timestamp(value: Any) -> int | None:
    """时间字段 → epoch 秒（接受秒 / 毫秒 / ISO 字符串 / 纯数字串）。"""
    num: float | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
    elif isinstance(value, str) and value.strip():
        trimmed = value.strip()
        if trimmed.isdigit():
            num = float(trimmed)
        else:
            try:
                from datetime import datetime
                num = datetime.fromisoformat(trimmed.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
    if num is None or num <= 0:
        return None
    return int(num / 1000) if num >= 1e12 else int(num)


@dataclass(slots=True)
class MinimaxCredential:
    access_token: str
    refresh_token: str = ""
    expires_at: int | None = None          # epoch 秒；None = 未知
    uid: str = ""
    nickname: str = ""
    scope: str = ""
    timezone_id: str = ""                  # 签到上报时区（空 = Asia/Shanghai 默认）

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
                "scope": self.scope, "timezone_id": self.timezone_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MinimaxCredential:
        return cls(
            access_token=str(data.get("access_token") or ""),
            refresh_token=str(data.get("refresh_token") or ""),
            expires_at=data.get("expires_at") if isinstance(data.get("expires_at"), int) else None,
            uid=str(data.get("uid") or ""),
            nickname=str(data.get("nickname") or ""),
            scope=str(data.get("scope") or ""),
            timezone_id=str(data.get("timezone_id") or ""),
        )

    def merge_refreshed(self, fresh: MinimaxCredential) -> MinimaxCredential:
        """续期合并：新 refresh_token 缺失时**保留旧值**（丢掉会让下次续期必败）。"""
        self.access_token = fresh.access_token
        if fresh.refresh_token:
            self.refresh_token = fresh.refresh_token
        if fresh.expires_at is not None:
            self.expires_at = fresh.expires_at
        return self


def _first_str(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _nested(source: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    current: Any = source
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current if isinstance(current, dict) else None


def _decode_jwt_claims(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3 or not parts[1]:
        return None
    padded = parts[1].replace("-", "+").replace("_", "/")
    padded += "=" * ((4 - len(padded) % 4) % 4)
    try:
        parsed = json.loads(base64.b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def credential_from_raw(raw: Any) -> MinimaxCredential:
    """解析 MiniMax 凭据（官方 auth.json 蛇形 / 驼峰 / 嵌套 / 裸令牌）。"""
    if isinstance(raw, str):
        source: dict[str, Any] = {"access_token": raw.strip()}
    elif isinstance(raw, dict):
        source = dict(raw)
        for key in ("credential", "credentials", "data", "auth"):
            nested = source.get(key)
            if isinstance(nested, dict):
                source = nested
                break
    else:
        raise ValueError(
            "MiniMax 凭据必须是一个 JSON 对象或令牌字符串"
            "（可粘贴 `~/.minimax/auth/prod/cn/mcode-public/auth.json` 的内容）")

    access = (_first_str(source, ("access_token", "accessToken", "access", "token"))
              or (_nested(source, ("auth",)) and _first_str(_nested(source, ("auth",)),
                                                            ("access_token", "accessToken")))
              or (_nested(source, ("data",)) and _first_str(_nested(source, ("data",)),
                                                            ("access_token", "accessToken"))))
    if not access:
        raise ValueError(
            "未能从凭据里读到 MiniMax 访问令牌（`access_token` / `accessToken`）。"
            "访问令牌形如 `mmoat_…`，可从 MiniMax Code 客户端的登录态文件里取得")
    refresh = (_first_str(source, ("refresh_token", "refreshToken", "refresh"))
               or (_nested(source, ("auth",)) and _first_str(_nested(source, ("auth",)),
                                                             ("refresh_token", "refreshToken")))
               or "")
    account_id = (_first_str(source, ("account_id", "accountId", "user_id", "userId",
                                      "uid", "sub", "minimax_user_id", "mavis_user_id"))
                  or (_nested(source, ("account",)) and _first_str(_nested(source, ("account",)),
                                                                   ("uid", "id", "account_id")))
                  or "")
    nickname = (_first_str(source, ("nickname", "displayName", "name", "username"))
                or (_nested(source, ("account",)) and _first_str(_nested(source, ("account",)),
                                                                 ("nickname", "name")))
                or "")
    expires_raw = (_first_str(source, ("expires_at", "expiresAt", "expire_time", "expireTime"))
                   or (_nested(source, ("auth",)) and _first_str(_nested(source, ("auth",)),
                                                                 ("expires_at", "expiresAt"))))
    expires_at = _parse_timestamp(expires_raw) if expires_raw else None
    claims = _decode_jwt_claims(access)
    if expires_at is None and claims:
        exp = claims.get("exp")
        if isinstance(exp, (int, float)) and not isinstance(exp, bool):
            expires_at = int(exp) if exp < 1e12 else int(exp / 1000)
    jwt_sub = claims and _first_str(claims, ("sub",))
    scope = _first_str(source, ("scope",)) or ""
    timezone_id = _first_str(source, ("timezone_id", "timezoneId")) or ""
    uid = account_id or (jwt_sub or "") or _stable_key(access)
    return MinimaxCredential(access_token=access, refresh_token=refresh or "",
                             expires_at=expires_at, uid=uid,
                             nickname=nickname or uid, scope=scope,
                             timezone_id=timezone_id)


def credential_from_token_grant(record: dict[str, Any]) -> MinimaxCredential | None:
    """令牌响应 → 凭据。硬校验照抄参考实现，任一不符返回 None。"""
    access = record.get("access_token")
    if not (isinstance(access, str) and access):
        return None
    token_type = record.get("token_type")
    if not (isinstance(token_type, str) and token_type.lower() == "bearer"):
        return None
    scope = record.get("scope")
    scope = scope if isinstance(scope, str) else ""
    if SCOPE not in scope:
        return None
    expires_in = record.get("expires_in")
    if not (isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool)
            and expires_in > 0):
        return None
    refresh_token = record.get("refresh_token")
    refresh_token = refresh_token if isinstance(refresh_token, str) and refresh_token else ""
    account_id = record.get("account_id") or record.get("accountId")
    account_id = account_id if isinstance(account_id, str) and account_id else ""
    uid = account_id or _stable_key(access)
    return MinimaxCredential(access_token=access, refresh_token=refresh_token,
                             expires_at=int(time.time() + expires_in),
                             uid=uid, nickname=uid, scope=scope)
