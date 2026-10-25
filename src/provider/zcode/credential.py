"""ZCode（智谱）凭据：静态 JWT，无续期。

ZCode 的凭据是**静态的**：JWT payload 实测只有 `{user_id, token_version,
sub, iat}` —— **没有 `exp`**，也**没有任何续期端点**（free-llm-router
`src/providers/zcode.ts` 的「刻意不实现 refresh()」一节有完整依据）。
因此本 dataclass 的 `needs_refresh` 恒 False，`refresh` 链路整体缺席；
真失效时上游回 401 / 业务码 1002，由错误分类（DEAD）处置。

`device_mid` 的语义：**不是账号标识**。同一账号每次重新登录都会得到新值
（实测其值不被服务端绑定校验），但它请求头 `X-Device-Mid` 的**必需值**
—— 缺了回 `400 {"code":3001}`。故随凭证持久化，绝不每次现生成
（那会让设备身份每次请求都变）。
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field
from typing import Any


def decode_jwt_payload(token: str) -> dict[str, Any] | None:
    """取 JWT payload 段字段（**不验签**，只用于账号标识与展示）。"""
    parts = token.split(".")
    if len(parts) < 2:
        return None
    padded = parts[1].replace("-", "+").replace("_", "/")
    padded += "=" * ((4 - len(padded) % 4) % 4)
    try:
        parsed = json.loads(base64.b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass(slots=True)
class ZcodeCredential:
    jwt: str
    uid: str
    device_mid: str = ""
    app_version: str = "3.14.3"
    nickname: str = ""

    @property
    def token_expires_at(self) -> int | None:
        """JWT 无 `exp`（也没有续期端点）⇒ 恒 None（面板不显示过期倒计时）。"""
        return None

    def needs_refresh(self, skew_seconds: int = 0) -> bool:
        """静态凭据：永不需要刷新。"""
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"jwt": self.jwt, "uid": self.uid, "device_mid": self.device_mid,
                "app_version": self.app_version, "nickname": self.nickname}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZcodeCredential:
        return cls(
            jwt=str(data.get("jwt") or ""),
            uid=str(data.get("uid") or ""),
            device_mid=str(data.get("device_mid") or ""),
            app_version=str(data.get("app_version") or "3.14.3"),
            nickname=str(data.get("nickname") or ""),
        )

    def merge_refreshed(self, fresh: ZcodeCredential) -> ZcodeCredential:
        """无续期链路；仅为与其它渠道的调用形态同形而保留。"""
        return self


def _first_str(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def credential_from_raw(raw: Any) -> ZcodeCredential:
    """解析 ZCode 凭据（JSON 导入）。

    接受的形态：
    1. `{"zcode_jwt": …}` / `{"jwt": …}` / `{"access_token": …}` 等键；
    2. 上面任一种再套一层 `{"credential": {…}}` / `{"data": {…}}`；
    3. JWT 单字符串（包一层 `{"jwt": "<token>"}` 也可）。

    判据是上游真正需要的两个字段：JWT 与账号标识 `user_id`（后者优先从
    显式字段取，回落到 JWT payload）。`device_mid` 缺省随机生成（实测
    该值不被服务端绑定校验，与官方登录行为一致）。
    """
    source: dict[str, Any]
    if isinstance(raw, str):
        source = {"jwt": raw.strip()}
    elif isinstance(raw, dict):
        source = dict(raw)
        for key in ("credential", "credentials", "data", "zcode"):
            nested = source.get(key)
            if isinstance(nested, dict):
                source = nested
                break
    else:
        raise ValueError("ZCode 凭据必须是一个 JWT 字符串，或含 `zcode_jwt` 字段的 JSON 对象")

    jwt = _first_str(source, ("zcode_jwt", "zcodeJwt", "jwt", "access_token",
                              "accessToken", "token"))
    if not jwt:
        raise ValueError(
            "ZCode 凭据缺少 JWT（`zcode_jwt` / `jwt`）。"
            "请用管理台的设备码登录，或把 JWT 粘进 JSON 的 `zcode_jwt` 字段")

    payload = decode_jwt_payload(jwt)
    uid = (_first_str(source, ("user_id", "userId", "uid"))
           or (payload and _first_str(payload, ("user_id", "userId", "sub", "uid"))) or None)
    if not uid:
        raise ValueError(
            "ZCode 凭据缺少账号标识（`user_id`），且 JWT 里也解不出 —— "
            "这个值是去重与面板展示的唯一稳定标识")
    device_mid = (_first_str(source, ("device_mid", "deviceMid", "mid"))
                  or str(uuid.uuid4()))
    nickname = (_first_str(source, ("account_name", "accountName", "displayName",
                                    "name", "nickname", "phone"))
                or (payload and _first_str(payload, ("name", "displayName")))
                or uid)
    app_version = _first_str(source, ("app_version", "appVersion")) or "3.14.3"
    return ZcodeCredential(jwt=jwt, uid=uid, device_mid=device_mid,
                           app_version=app_version, nickname=nickname)


def credential_from_login(jwt: str, uid: str, nickname: str | None,
                          app_version: str = "3.14.3") -> ZcodeCredential:
    """设备码登录成功 → 凭据。`device_mid` 随机生成并随凭证持久化。"""
    return ZcodeCredential(jwt=jwt, uid=uid,
                           device_mid=str(uuid.uuid4()),
                           app_version=app_version,
                           nickname=nickname or uid)
