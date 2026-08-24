"""Provider adapters. Common async interface so the same workload runs against any
OpenAI-compatible endpoint. Each provider streams one turn and returns metrics +
the provider-reported usage (crucially: cached_tokens, to verify cache hit/miss).

Add a provider by adding an entry to the PROVIDERS registry (or via env vars); every
OpenAI-compatible endpoint uses the OpenAICompat client.
"""
import os
import json
import time

REASONING_EFFORT = os.environ.get("INFERBENCH_REASONING_EFFORT") or None
DEFAULT_SESSION_HEADER = "X-Session-Id"
TOKEN_PARAMS = ("both", "max_tokens", "max_completion_tokens")


def token_budget_fields(n, token_param="both"):
    """Fields that cap decode. TokenHub and others renamed max_tokens.

    Default sends both so a stack that honors only one still gets the sampled
    cap. Strict APIs that 400: pass max_tokens or max_completion_tokens alone.
    """
    n = int(n)
    if token_param == "both":
        return {"max_tokens": n, "max_completion_tokens": n}
    if token_param in ("max_tokens", "max_completion_tokens"):
        return {token_param: n}
    raise ValueError(f"token_param must be one of {TOKEN_PARAMS}, got {token_param!r}")


def resolve_token_param(*choices):
    for choice in choices:
        if not choice:
            continue
        if choice not in TOKEN_PARAMS:
            raise ValueError(f"token_param must be one of {TOKEN_PARAMS}, got {choice!r}")
        return choice
    return "both"


def _env():
    """Process env, plus optional INFERBENCH_ENV file (real env vars win)."""
    d = dict(os.environ)
    path = os.environ.get("INFERBENCH_ENV")
    if path and os.path.exists(path):
        for ln in open(path, errors="ignore"):
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.rstrip("\n").split("=", 1)
                d.setdefault(k, v.strip().strip('"').strip("'"))
    return d


class OpenAICompat:
    """OpenAI /v1/chat/completions streaming. base_url should include /v1."""

    def __init__(self, name, base_url, model, headers, extra_body=None, session_header=None,
                 token_param="both"):
        self.name = name
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json", **headers}
        self.extra_body = extra_body or {}
        self.session_header = session_header
        self.token_param = resolve_token_param(token_param)

    async def run_turn(self, client, messages, max_tokens, session_id=None):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **token_budget_fields(max_tokens, self.token_param),
            **({"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}),
            **self.extra_body,
        }
        headers = dict(self.headers)
        if self.session_header and session_id:
            headers[self.session_header] = session_id
        t0 = time.monotonic()
        ttft = None
        first_chunk = None
        usage = {}
        try:
            async with client.stream("POST", self.url, headers=headers, json=body) as r:
                if r.status_code != 200:
                    await r.aread()
                    return {"ok": False, "code": r.status_code}
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    d = line[6:]
                    if d == "[DONE]":
                        break
                    try:
                        j = json.loads(d)
                    except Exception:
                        continue
                    if first_chunk is None:
                        first_chunk = time.monotonic() - t0
                    delta = (j.get("choices") or [{}])[0].get("delta", {}) or {}
                    if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                        ttft = time.monotonic() - t0
                    if j.get("usage"):
                        usage = j["usage"]
            e2e = time.monotonic() - t0
            if ttft is None:
                ttft = first_chunk
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            return {
                "ok": True, "ttft": ttft, "e2e": e2e,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": cached,
                "completion_tokens": usage.get("completion_tokens"),
            }
        except Exception as e:
            return {"ok": False, "code": type(e).__name__}


# Or skip the table: set <NAME>_BASE_URL, <NAME>_MODEL, <NAME>_API_KEY.
PROVIDERS = {
    "local":  {"base_url": "http://localhost:30000/v1", "model": "your-model", "key_env": "LOCAL_API_KEY"},
    "hosted": {"base_url_env": "HOSTED_BASE_URL", "model": "your-model", "key_env": "HOSTED_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "your-model", "key_env": "OPENAI_API_KEY",
               "token_param": "max_completion_tokens", "extra_body": {"reasoning_effort": "medium"}},
}


def _v1(url):
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def _require_env(e, key, what):
    val = e.get(key)
    if not val:
        raise ValueError(f"missing {what}: set environment variable {key}")
    return val


def build_provider(name, token_param=None):
    """Build a provider from the PROVIDERS registry, or from <NAME>_* env vars if unknown."""
    e = _env()
    cfg = PROVIDERS.get(name)
    p = name.upper().replace("-", "_")
    if cfg is None:
        if f"{p}_BASE_URL" in e and f"{p}_MODEL" in e:
            cfg = {"base_url": e[f"{p}_BASE_URL"], "model": e[f"{p}_MODEL"],
                   "key_env": f"{p}_API_KEY"}
        else:
            raise ValueError(
                f"unknown provider '{name}'. Known: {', '.join(sorted(PROVIDERS))}. "
                f"Or set {p}_BASE_URL, {p}_MODEL, {p}_API_KEY in the environment.")

    if "base_url_env" in cfg:
        base = _v1(_require_env(e, cfg["base_url_env"], "base URL"))
    else:
        base = _v1(cfg["base_url"])

    if "headers_env" in cfg:
        headers = {h: _require_env(e, env, f"header {h}") for h, env in cfg["headers_env"].items()}
    else:
        headers = {"Authorization": f"Bearer {_require_env(e, cfg['key_env'], 'API key')}"}

    session_header = (
        cfg.get("session_header")
        or e.get(f"{p}_SESSION_HEADER")
        or e.get("INFERBENCH_SESSION_HEADER")
    )
    model = e.get(f"{p}_MODEL", cfg["model"])
    token_param = resolve_token_param(
        token_param,
        e.get(f"{p}_TOKEN_PARAM"),
        e.get("INFERBENCH_TOKEN_PARAM"),
        cfg.get("token_param"),
        "both",
    )
    return OpenAICompat(
        name, base, model, headers,
        extra_body=cfg.get("extra_body"),
        session_header=session_header,
        token_param=token_param,
    )


def ensure_affinity(provider, enabled):
    """If affinity is requested, ensure a session header is set (default X-Session-Id).

    Returns True when session ids will be sent. Without this, --affinity was a silent no-op
    for env-defined providers that never set session_header.
    """
    if not enabled:
        return False
    if not provider.session_header:
        provider.session_header = os.environ.get("INFERBENCH_SESSION_HEADER", DEFAULT_SESSION_HEADER)
        print(
            f"note: --affinity using header {provider.session_header!r} "
            f"(set INFERBENCH_SESSION_HEADER or <NAME>_SESSION_HEADER / provider "
            f"session_header to match your router)",
            flush=True,
        )
    return True
