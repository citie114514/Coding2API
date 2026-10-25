"""Loomy（讯飞）上游客户端：OpenAI 协议 + CAccount HMAC-SHA1 签名。

移植自 free-llm-router `src/providers/loomy.ts`。协议要点：

- **两套认证**：业务端点（/models、/points/*）只认小写 `token` 头；
  chat 端点只认 `Authorization: Bearer`，但官方客户端两个都发（照抄），
  `Bearer ` 前缀必需（实测无前缀回 `100002 缺少 token`）。
- **业务失败恒返回 HTTP 200**，成败只能读 body 的 `code`（'000000' 为 ok）。
  chat 路径对 `application/json` 响应 peek 一次（只在 content-type 明确是
  JSON 时读 —— 绝不对 SSE 流做整包读取）。
- 倍率**没有独立字段**，写在模型 `name` 末尾括号里（三种括号风格混用）。
- 「签到」= 触发每日额度重置（不是 +积分）；幂等判据 `alreadyProcessed`；
  `dailyQuota` 只在 first-login 响应里有，不硬编码 5000。
- **无 refresh 端点**（session 14 天，过期只能重新登录）—— 刻意不实现。
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ...engine.sse import iter_frames
from ...provider import base
from ...provider.base import CheckinResult, ErrKind, Event, Model, Quota
from ...provider.proxy import build_client
from .credential import LoomyCredential

API_BASE = "https://loomyad.xunfei.cn/api/v1"
ACCOUNT_BASE = "https://account.xfinfr.com"
OK_CODE = "000000"
AUTH_ERROR_CODE = "100002"

ACCESS_KEY_ID = "2thryby66wxi53sk"
ACCESS_KEY_SECRET = "zsak6eadrbawz683wf5r3m2snrwj868r"
APP_ID = "GM3LOOMY"

STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
SHORT_TIMEOUT = httpx.Timeout(30.0)

def _classify_status(status: int, body: bytes) -> ErrKind:
    text = body.decode("utf-8", errors="replace")
    if AUTH_ERROR_CODE in text or status in (401, 403):
        return ErrKind.DEAD
    if status == 402:
        return ErrKind.CREDIT
    if status == 429:
        return ErrKind.SOFT
    if status >= 500:
        return ErrKind.SOFT
    if status == 400:
        return ErrKind.INVALID
    return ErrKind.OTHER


class UpstreamHTTPError(base.UpstreamHTTPError):
    """Loomy 上游错误。code='100002'（登录失效）→ DEAD。"""

    classify_status = staticmethod(_classify_status)


def business_headers(token: str, *, json_body: bool = False) -> dict[str, str]:
    """业务端点头：只认小写 `token` 头，**刻意不加** Authorization。"""
    headers: dict[str, str] = {"Accept": "application/json", "token": token}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def chat_headers(token: str) -> dict[str, str]:
    """chat 端点头：两个认证头都发（官方客户端 session 模式同款）。"""
    return {"Accept": "text/event-stream", "Content-Type": "application/json",
            "Authorization": f"Bearer {token}", "token": token}


# ── CAccount HMAC-SHA1 签名（讯飞账号端点） ─────────────────────────

def _escape_rfc3986(value: str) -> str:
    """RFC3986 转义：unreserved 只留 A-Za-z0-9 - _ . ~。

    等价于 TS 的 `encodeURIComponent` 再补转 `! ' ( ) *`（sign.js 同口径）；
    Python 的 `quote(value, safe="")` 恰好就是这个结果。
    """
    from urllib.parse import quote
    return quote(value, safe="")


def _escaped_path(raw_path: str) -> str:
    """补前导 /、剥末尾 /（长度 > 1 时）、按 / 切段逐段转义再拼回。"""
    clean = raw_path if raw_path.startswith("/") else f"/{raw_path}"
    if len(clean) > 1 and clean.endswith("/"):
        clean = clean[:-1]
    return "/".join(_escape_rfc3986(seg) for seg in clean.split("/"))


def _content_md5(body: str) -> str:
    """空 body 返回**空串**（不是空串的 md5），与客户端 sign.js 一致。"""
    if not body:
        return ""
    return base64.b64encode(hashlib.md5(body.encode()).digest()).decode()


def auth_headers(path: str, body: str, *, method: str = "POST",
                 content_type: str = "application/json") -> dict[str, str]:
    """讯飞账号端点签名头。

    ⚠️ 9 段待签名字符串以**两个换行结尾**（SignedHeaders 与
    CanonicalizedHeaders 恒为空串）—— 去掉尾随换行签名即失配。
    ⚠️ 认证头前缀是 `account {ak}:{sig}`，不是 Bearer。
    ⚠️ body 字符串必须与实际发送的**完全一致**（二次序列化会改字节）。
    """
    date = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
    nonce = uuid.uuid4().hex
    string_to_sign = "\n".join([
        method.upper(), _escaped_path(path), "", _content_md5(body),
        content_type, date, nonce, "", "",
    ])
    signature = base64.b64encode(
        hmac.new(ACCESS_KEY_SECRET.encode(), string_to_sign.encode(),
                 hashlib.sha1).digest()).decode()
    headers = {"Authorization": f"account {ACCESS_KEY_ID}:{signature}",
               "Date": date, "Nonce": nonce, "Content-Type": content_type}
    md5 = _content_md5(body)
    if md5:
        headers["Content-MD5"] = md5
    return headers


def account_body(param: dict[str, Any]) -> dict[str, Any]:
    """讯飞账号端点信封 `{base, param}`。ua 硬编码 macOS（官方 Windows 也发这个）。"""
    return {"base": {"appid": APP_ID, "modelid": "Web", "version": "1.0.0",
                     "devid": "web", "ua": "Loomy|Desktop|Electron|macOS",
                     "traceid": uuid.uuid4().hex},
            "param": param}


def parse_envelope(payload: Any) -> tuple[bool, str, str, Any]:
    """业务信封 → (ok, code, message, data)。业务失败恒 HTTP 200，只认 code。"""
    if not isinstance(payload, dict):
        return False, "", "响应不是 JSON 对象", None
    code = payload.get("code")
    code = code if isinstance(code, str) else ""
    message = payload.get("desc") if isinstance(payload.get("desc"), str) \
        and payload.get("desc") else (payload.get("message")
                                      if isinstance(payload.get("message"), str) else "")
    if code != OK_CODE:
        return False, code, message or f"业务错误 {code or '(缺少 code)'}", None
    return True, code, message, payload.get("data")


def split_rate(raw_name: str) -> tuple[str, str]:
    """从展示名抽倍率（三种括号风格混用 + 已规范化形态，幂等）。"""
    import re as _re
    original = raw_name.strip() if isinstance(raw_name, str) else ""
    if not original:
        return "", ""
    bracketed = _re.match(r"^(.*?)\s*[（(]\s*(x\s*[\d.]+)\s*[)）]\s*$", original, _re.I)
    if bracketed:
        name, rate = bracketed.group(1).strip(), _re.sub(r"\s+", "", bracketed.group(2)).lower()
        return (name, rate) if name else (original, "")
    normalized = _re.match(r"^(.*?)\s*·\s*(x\s*[\d.]+)\s*$", original, _re.I)
    if normalized:
        name, rate = normalized.group(1).strip(), _re.sub(r"\s+", "", normalized.group(2)).lower()
        if name:
            return name, rate
    return original, ""


def display_name(raw_name: str) -> str:
    name, rate = split_rate(raw_name)
    return f"{name} · {rate}" if rate else name


def parse_models_payload(payload: Any) -> list[Model]:
    """`GET /models` → Model 列表。必须认**三种**形态：业务信封 / 裸数组 /
    裸对象 + `data[]`（实测 `/models` 真实形态是第三种，没有 code 字段）。"""
    if isinstance(payload, dict):
        ok, _, _, data = parse_envelope(payload)
        if ok:
            entries = data if isinstance(data, list) else []
        elif isinstance(payload.get("data"), list):
            entries = payload["data"]          # OpenAI 风格列表信封（无 code）
        else:
            entries = []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    out: list[Model] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not (isinstance(model_id, str) and model_id.strip()) or item.get("type") != "chat":
            continue                           # 判据是 type==='chat'，不是 input_modalities
        model_id = model_id.strip()
        raw_name = item.get("name") if isinstance(item.get("name"), str) and item.get("name") \
            else model_id
        context = item.get("context_length")
        capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        modalities = capabilities.get("input_modalities")
        modalities = [m.lower() for m in modalities if isinstance(m, str)] \
            if isinstance(modalities, list) else []
        out.append(Model(
            id=model_id, name=display_name(raw_name),
            max_input_tokens=context if isinstance(context, int) and context > 0 else None,
            max_output_tokens=None,            # 目录无输出上限字段，如实未知
            supports_images="image" in modalities,
            credit_rate=None,                  # Loomy 只有倍率没有免费概念
        ))
    return out


def _cleanup_tool_pairing(messages: list[Any]) -> list[Any]:
    """孤儿 tool 消息与空名 tool_calls 清理（与 raccoon 同实现，OpenAI 规范修复）。"""
    from ..raccoon.client import _cleanup_tool_pairing as _cleanup
    return _cleanup(messages)


def prepare_body(payload: dict[str, Any], model: str) -> dict[str, Any]:
    body = dict(payload)
    body["messages"] = copy.deepcopy(payload.get("messages"))
    body["model"] = model
    body["stream"] = True
    if isinstance(body.get("messages"), list):
        body["messages"] = _cleanup_tool_pairing(body["messages"])
    return body


def _read_error_body(text: str) -> str | None:
    """非 SSE 响应体里的业务错误（OpenAI 形状或 Loomy 信封）。"""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
        return error["message"]
    code = parsed.get("code")
    if isinstance(code, str) and code != OK_CODE:
        desc = parsed.get("desc") if isinstance(parsed.get("desc"), str) else ""
        if code == AUTH_ERROR_CODE:
            return f"登录已失效（{code} {desc}）：Loomy 没有续期端点，请重新登录".strip()
        return f"上游业务错误 {code}：{desc}" if desc else f"上游业务错误 {code}"
    return None


@dataclass(slots=True)
class LoomyClient:
    api_base: str = API_BASE
    account_base: str = ACCOUNT_BASE
    _stream_client: httpx.AsyncClient | None = None
    _short_client: httpx.AsyncClient | None = None
    proxy: str | None = None

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")
        self.account_base = self.account_base.rstrip("/")

    def _stream(self) -> httpx.AsyncClient:
        if self._stream_client is None:
            self._stream_client = build_client(timeout=STREAM_TIMEOUT, proxy=self.proxy)
        return self._stream_client

    def _short(self) -> httpx.AsyncClient:
        if self._short_client is None:
            self._short_client = build_client(timeout=SHORT_TIMEOUT, proxy=self.proxy)
        return self._short_client

    async def aclose(self) -> None:
        for client in (self._stream_client, self._short_client):
            if client is not None:
                await client.aclose()

    async def stream_chat(self, credential: LoomyCredential, payload: dict[str, Any],
                          model: str) -> AsyncIterator[Event]:
        """chat/completions。失败恒 HTTP 200 → 对 JSON 响应 peek 一次业务错误。"""
        from ..qoder.events import parse_inner_chunk          # OpenAI 帧 → 中立事件（共用）
        body = prepare_body(payload, model)
        async with self._stream().stream(
            "POST", f"{self.api_base}/chat/completions",
            headers=chat_headers(credential.access_token), json=body,
        ) as response:
            if response.status_code >= 400:
                raw = await base.read_body_bounded(response)
                raise UpstreamHTTPError(response.status_code, raw,
                                        **base.retry_hints(response.headers))
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                # 非流式 JSON：可安全整读。200 + 业务错误必须显式抛，
                # 否则表现为「干净地停止、没有任何报错」（登录已失效的典型形态）。
                raw = await base.read_body_bounded(response)
                detail = _read_error_body(raw.decode("utf-8", errors="replace"))
                if detail is not None:
                    raise UpstreamHTTPError(200, raw)
                yield Event(kind=EventKind.CONTENT, content="")   # 不会到达；占位防 fallthrough
                return
            async for frame in iter_frames(response.aiter_bytes()):
                if not frame.data:
                    continue
                for event in parse_inner_chunk(frame.data):
                    yield event

    async def _business(self, credential: LoomyCredential, path: str, *,
                        method: str = "GET", json_body: dict[str, Any] | None = None) -> Any:
        """业务请求 + 拆信封；code != '000000' 抛 UpstreamHTTPError（body 带码）。"""
        headers = business_headers(credential.access_token, json_body=json_body is not None)
        try:
            response = await self._short().request(
                method, f"{self.api_base}{path}", headers=headers, json=json_body)
        except httpx.HTTPError as error:
            raise base.UpstreamTransportError(str(error)) from error
        try:
            parsed = response.json()
        except ValueError as error:
            raise UpstreamHTTPError(response.status_code,
                                    response.content[:2048]) from error
        ok, code, message, data = parse_envelope(parsed)
        if not ok:
            raise UpstreamHTTPError(200, json.dumps(
                {"code": code, "desc": message}, ensure_ascii=False).encode())
        return data

    async def fetch_models(self, credential: LoomyCredential) -> list[Model]:
        """用 raw（不拆信封）拉目录：解析器要同时认信封与裸数组两种形状。"""
        try:
            response = await self._short().get(
                f"{self.api_base}/models",
                headers=business_headers(credential.access_token))
        except httpx.HTTPError as error:
            raise base.UpstreamTransportError(str(error)) from error
        if response.status_code in (401, 403):
            raise UpstreamHTTPError(response.status_code, response.content)
        try:
            payload = response.json()
        except ValueError as error:
            raise UpstreamHTTPError(response.status_code,
                                    response.content[:2048]) from error
        ok, code, message, _ = parse_envelope(payload)
        if not ok and code:
            # 信封层失败必须报出来（100002 等），否则被解析成空目录、
            # 最终报成「目录为空」——与真实原因毫无关系
            raise UpstreamHTTPError(200, json.dumps(
                {"code": code, "desc": message}, ensure_ascii=False).encode())
        models = parse_models_payload(payload)
        if not models:
            raise UpstreamHTTPError(200, b'{"error":"loomy model catalog empty"}')
        return models

    async def probe_quota(self, credential: LoomyCredential) -> Quota:
        """两个积分池分开报告（每日池当天作废；合并会让用户看不出哪部分会失效）。

        ⚠️ 用 points/records（只读），绝不调 first-login（写端点）当查询用。
        """
        probed_at = int(time.time())
        data = await self._business(
            credential, "/points/records?pageNo=1&pageSize=1&recordType=all")

        def num(key: str) -> float | None:
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return None

        permanent = num("balance")
        if permanent is None:
            # 核心字段缺失 ⇒ 形状不对；返回 0 会把「查不到」伪装成「余额为 0」
            raise UpstreamHTTPError(200, b'{"error":"loomy balance missing balance field"}')
        daily = num("dailyBalance") or 0.0
        total = num("availableBalance") or (permanent + daily)
        packages = [
            {"name": "永久积分", "total": permanent, "used": 0.0, "end": None},
            {"name": "每日赠送（当天有效，消耗后不回补）", "total": daily, "used": 0.0,
             "end": None},
        ]
        return Quota(remaining=total, total=total, cycle_end=None, packages=packages,
                     probed_at=probed_at)

    async def checkin(self, credential: LoomyCredential) -> CheckinResult:
        """触发每日赠送额度（不是 +积分）。幂等判据 alreadyProcessed。"""
        data = await self._business(credential, "/points/first-login",
                                    method="POST", json_body={})
        if not isinstance(data, dict):
            data = {}

        def num(key: str) -> float | None:
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return None

        daily_quota = num("dailyQuota")
        daily_consumed = num("dailyConsumed")
        daily_balance = num("dailyBalance")
        if data.get("alreadyProcessed") is True:
            detail = (f"今日每日额度已初始化：{daily_balance:g}/{daily_quota:g}"
                      "（消耗后不回补）" if daily_quota is not None
                      else "今日每日额度已初始化（幂等命中，未重复发放）")
            return CheckinResult(ok=True, already_checked_in=True, credit=0,
                                 message=detail)
        gained = (max(0.0, daily_quota - (daily_consumed or 0.0))
                  if daily_quota is not None and daily_consumed is not None
                  else (daily_quota or 0.0))
        detail = (f"已触发今日每日赠送额度：可用 {gained:g}/{daily_quota:g}"
                  "（消耗后不回补，次日重置）" if daily_quota is not None
                  else "已触发今日每日赠送额度（服务端未回报 quota，具体数额未知）")
        return CheckinResult(ok=True, credit=gained, message=detail)

    # ── 短信登录（讯飞账号端点，HMAC-SHA1 签名） ────────────────────

    async def _post_account(self, path: str, param: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(account_body(param), separators=(",", ":"),
                          ensure_ascii=False)
        headers = auth_headers(path, body)
        try:
            response = await self._short().post(
                f"{self.account_base}{path}", headers=headers, content=body.encode())
        except httpx.HTTPError as error:
            raise base.UpstreamTransportError(str(error)) from error
        try:
            parsed = response.json()
        except ValueError as error:
            raise UpstreamHTTPError(response.status_code,
                                    response.content[:2048]) from error
        ok, _, message, data = parse_envelope(parsed)
        if not ok:
            # 服务端文案透传（上层据它区分「验证码错误」与「手机号格式不正确」）
            raise UpstreamHTTPError(response.status_code, (message or "讯飞账号请求失败").encode())
        return data if isinstance(data, dict) else {}

    async def send_sms_code(self, phone: str) -> str:
        """下发短信验证码 → msgid（提交验证码时必须原样带回）。"""
        data = await self._post_account(
            "/login/phone/sendMsgCode",
            {"ccode": "86", "phone": phone, "expire": 300})
        msgid = data.get("msgid")
        if not (isinstance(msgid, str) and msgid):
            # 不返回空串：上层会拿它去登录，报的将是与真实原因无关的错误
            raise UpstreamHTTPError(200, b'{"error":"loomy sms response missing msgid"}')
        return msgid

    async def login_by_sms(self, phone: str, code: str, msgid: str) -> LoomyCredential:
        """短信验证码登录。"""
        data = await self._post_account(
            "/login/phone/checkCode",
            {"ccode": "86", "phone": phone, "mcode": code, "msgid": msgid,
             "expire": 1209600})
        session = data.get("session")
        userid = data.get("userid")
        if not (isinstance(session, str) and session):
            raise UpstreamHTTPError(200, b'{"error":"loomy login response missing session"}')
        if not (isinstance(userid, str) and userid):
            raise UpstreamHTTPError(200, b'{"error":"loomy login response missing userid"}')
        from .credential import credential_from_login
        return credential_from_login(session, userid, phone)
