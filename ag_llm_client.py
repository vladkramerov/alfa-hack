"""Клиент к OpenAI-совместимому API (Qwen организаторов).

Как в baseline.py: пакет openai, настройки из OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL.
Если openai не установлен, запрос уходит напрямую через httpx. Если API не настроен,
недоступен или кончился бюджет токенов, available() возвращает False и агент работает на фолбэке."""
from __future__ import annotations

import json
import logging
import math
import re
import time

from ag_config import Config
from ag_io_contract import LLMSettings

log = logging.getLogger("agent.llm")

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None
try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


class LLMClient:
    def __init__(self, settings: LLMSettings, cfg: Config):
        self.s = settings
        self.cfg = cfg
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0
        self.failures_in_row = 0
        self.seconds = 0.0          # сколько ждали ответов (потолок — cfg.llm_time_share лимита)
        self.disabled_reason: str | None = None
        if not cfg.llm_enabled:
            self.disabled_reason = "LLM выключена (LLM_DISABLED=1)"
        elif not all((settings.base_url, settings.api_key, settings.model)):
            self.disabled_reason = "не заданы OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL"
        elif OpenAI is None and httpx is None:
            self.disabled_reason = "нет ни openai, ни httpx"
        self._sdk = None
        self._hints = True          # слать ли параметры отключения рассуждений (см. _extra)
        self._prompt_tokens_per_byte = 0.5  # conservative first request; calibrate from actual usage
        self.last: dict = {}        # служебное о последнем ответе: finish_reason, токены (для local/llm_check.py)
        if not self.disabled_reason and OpenAI is not None:
            self._sdk = OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                               timeout=cfg.llm_timeout, max_retries=0)

    # ---- состояние ----
    def available(self) -> bool:
        if self.disabled_reason:
            return False
        if self.tokens_in + self.tokens_out >= self.cfg.llm_max_tokens_total:
            self.disabled_reason = "исчерпан бюджет токенов"
            return False
        if self.seconds >= self.cfg.llm_time_share * self.cfg.time_limit:
            self.disabled_reason = f"на ответы LLM ушло {self.seconds:.0f} с — дальше без неё, время нужнее моделям"
            return False
        return True

    def stats(self) -> dict:
        return {"calls": self.calls, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "seconds": round(self.seconds), "disabled_reason": self.disabled_reason}

    # ---- запросы ----
    def chat(self, system: str, user: str, max_tokens: int = 1500, temperature: float | None = None,
             timeout: float | None = None) -> str | None:
        if not self.available():
            return None
        if self.cfg.llm_no_think:
            user = user + "\n/no_think"  # Qwen3: отключить длинные рассуждения, они съедают токены и время
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        temp = self.cfg.llm_temperature if temperature is None else temperature
        budget = max_tokens
        for attempt in range(3):   # обычный повтор, повтор с запасом токенов, повтор без параметров
            if not self.available():
                break
            # Reserve prompt room before sending it: the previous implementation
            # could send a 14k-token prompt with only 6k left in the entire budget.
            prompt_reserve = math.ceil(len((system + user).encode("utf-8")) * self._prompt_tokens_per_byte) + 128
            room = self.cfg.llm_max_tokens_total - self.tokens_in - self.tokens_out - prompt_reserve
            if room < 64:
                self.disabled_reason = "недостаточно бюджета для контекста и ответа LLM"
                break
            budget = min(budget, room)
            wait_left = self.cfg.llm_time_share * self.cfg.time_limit - self.seconds
            effective_timeout = min(timeout or self.cfg.llm_timeout, wait_left)
            if effective_timeout <= 0:
                break
            try:
                self.calls += 1
                t0 = time.time()
                try:
                    text, p_tok, c_tok, finish = self._request(messages, budget, temp, effective_timeout)
                finally:
                    self.seconds += time.time() - t0
                if p_tok:
                    self._prompt_tokens_per_byte = min(0.5, max(0.1, 1.2 * p_tok / max(1, len((system + user).encode("utf-8")))))
                self.tokens_in += p_tok or len(system + user) // 3
                self.tokens_out += c_tok or len(text) // 3
                text = _strip_thinking(text)
                self.last = {"finish_reason": finish, "completion_tokens": c_tok, "max_tokens": budget,
                             "thinking_hints": self._hints}
                if text:
                    self.failures_in_row = 0
                    return text
                # пустой ответ: модель потратила весь лимит на рассуждения (у Qwen3 они идут в отдельное
                # поле или в <think> без конца) — один повтор с запасом токенов, дальше считаем неудачей
                log.warning("LLM вернула пустой ответ (finish_reason=%s, %s токенов из %s)", finish, c_tok, budget)
                if finish != "length" or budget >= 4 * max_tokens:
                    break
                budget = min(4 * max_tokens, 8000)
            except Exception as e:  # noqa: BLE001
                if self._hints and _rejects_params(e):
                    # прокси не знает параметров отключения рассуждений — больше их не шлём
                    log.warning("LLM не принимает параметры отключения рассуждений: %s", type(e).__name__)
                    self._hints = False
                    continue
                log.warning("LLM ошибка (попытка %d): %s", attempt + 1, type(e).__name__)
                time.sleep(2 * (attempt + 1))
        self.failures_in_row += 1
        if self.failures_in_row >= 3:
            self.disabled_reason = "3 неудачных запроса подряд"
        return None

    def _extra(self) -> dict:
        """Просьба не рассуждать для серверов, которые это понимают (vLLM/SGLang — chat_template_kwargs,
        OpenRouter-подобные — reasoning). Если сервер отвергнет параметры, chat() перестанет их слать."""
        if not (self._hints and self.cfg.llm_no_think):
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}, "reasoning": {"enabled": False}}

    def _request(self, messages: list, max_tokens: int, temperature: float,
                 timeout: float) -> tuple[str, int, int, str | None]:
        extra = self._extra()
        if self._sdk is not None:
            r = self._sdk.chat.completions.create(model=self.s.model, messages=messages, max_tokens=max_tokens,
                                                  temperature=temperature, timeout=timeout,
                                                  **({"extra_body": extra} if extra else {}))
            u = r.usage
            return (r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0) or 0,
                    getattr(u, "completion_tokens", 0) or 0, r.choices[0].finish_reason)
        if not self.s.base_url:
            raise ValueError("OPENAI_BASE_URL is required")
        base = self.s.base_url.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.s.api_key or 'EMPTY'}"}
        payload = {"model": self.s.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                   **extra}
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code in (400, 422):
            raise ValueError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
        data = resp.json()
        u = data.get("usage") or {}
        ch = data["choices"][0]
        return (ch["message"].get("content") or "", int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)),
                ch.get("finish_reason"))

    def chat_json(self, system: str, user: str, max_tokens: int = 1500, temperature: float | None = None) -> dict | None:
        text = self.chat(system, user + "\n\nОтветь одним JSON-объектом без пояснений.", max_tokens, temperature)
        if text is None:
            return None
        obj = extract_json(text)
        if obj is None:
            log.warning("LLM вернула не-JSON: %d символов", len(text))
        return obj


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:            # рассуждение оборвано лимитом токенов: ответа после него нет
        text = text[:text.index("<think>")]
    return text.strip()


def _rejects_params(e: Exception) -> bool:
    """Ошибка 400/422 из-за незнакомых параметров запроса (а не из-за ключа, бюджета или сети)."""
    msg = str(e).lower()
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    bad_request = status in (400, 422) or "http 400" in msg or "http 422" in msg
    return bad_request and any(w in msg for w in ("chat_template_kwargs", "reasoning", "extra", "unrecognized",
                                                  "unknown", "unsupported", "not permitted", "invalid"))


def extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # первый сбалансированный {...}
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
