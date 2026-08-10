"""Provider adapters. Common async interface so the same workload runs against any
OpenAI-compatible endpoint. Each provider streams one turn and returns metrics +
the provider-reported usage (crucially: cached_tokens, to verify cache hit/miss).

Add a provider by adding an entry to the PROVIDERS registry (or via env vars); every
OpenAI-compatible endpoint uses the OpenAICompat client.
"""
import os
import json
import time
import httpx

# Reasoning-effort pin. None = send nothing -> the endpoint's own default. Pin a string ONLY for a like-for-like CAPACITY comparison
# across endpoints. Reasoning models typically expose distinct paths, e.g.
#   "none"/"minimal" -> ~0 reasoning tokens ; "low"/"medium"/"high" ; "xhigh"/"max".
# Overridable at runtime: export INFERBENCH_REASONING_EFFORT=max
REASONING_EFFORT = os.environ.get("INFERBENCH_REASONING_EFFORT") or None

# Default sticky-routing header when --affinity is used and the provider did not set one.
DEFAULT_SESSION_HEADER = "X-Session-Id"


def _env():
    """Credentials from the environment, with an optional .env overlay.

    Set the keys your providers need directly in the environment (e.g. an API key per
    provider, plus any base-URL vars), or point the INFERBENCH_ENV variable at a .env
    file. Real environment variables win; the file only fills gaps.
    """
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
                 token_param="max_tokens"):
        self.name = name
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json", **headers}
        self.extra_body = extra_body or {}
        self.session_header = session_header   # sticky-routing header for cache affinity, if any
        # Output-cap field name. Default "max_tokens": honored by SGLang/vLLM and most
        # OpenAI-compatible servers. OpenAI/gpt-5.x reasoning models require
        # "max_completion_tokens". CRITICAL GOTCHA: some OpenAI-compatible endpoints SILENTLY
        # IGNORE max_completion_tokens -> no output cap -> a reasoning model runs to the server
        # default (often 65,536 = 2^16) -> multi-minute E2E that looks like slow serving but is
        # runaway generation length. If you see OutMax == 65536, switch that provider to max_tokens.
        self.token_param = token_param

    async def run_turn(self, client, messages, max_tokens, session_id=None):
        body = {
            "model": self.model,
            "messages": messages,
            self.token_param: max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **({"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}),
            **self.extra_body,
        }
        headers = dict(self.headers)
        if self.session_header and session_id:
            headers[self.session_header] = session_id
        t0 = time.monotonic()
        ttft = None
        first_chunk = None   # time to first streamed chunk of any kind (prefill-done fallback)
        usage = {}
        finish_reason = None
        request_id = None
        try:
            async with client.stream("POST", self.url, headers=headers, json=body) as r:
                if r.status_code != 200:
                    body_text = (await r.aread()).decode(errors="replace")[:200]
                    # A 429 is a quota signal, not a generic failure: Retry-After and the body
                    # name WHICH limit fired (tokens/min vs requests/min vs concurrency), which
                    # is the difference between "endpoint is slow" and "you are being metered".
                    return {"ok": False, "code": r.status_code, "http_status": r.status_code,
                            "retry_after": r.headers.get("retry-after"),
                            "request_id": r.headers.get("x-request-id"),
                            "error_body": body_text}
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
                    if request_id is None and j.get("id"):
                        request_id = j["id"]
                    choice = (j.get("choices") or [{}])[0]
                    delta = choice.get("delta", {}) or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    # first GENERATED token: content OR reasoning_content (reasoning models)
                    if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                        ttft = time.monotonic() - t0
                    if j.get("usage"):
                        usage = j["usage"]
            e2e = time.monotonic() - t0
            if ttft is None:
                ttft = first_chunk   # no content-bearing delta seen -> use first-chunk time
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            return {
                "ok": True, "ttft": ttft, "e2e": e2e,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": cached,
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                # finish_reason "length" means the output cap bound, not that the model stopped.
                # An HTTP 200 carrying no usage and no finish_reason is a STALLED STREAM, not a
                # success: it contributes no tokens but does contribute to the latency tail.
                "finish_reason": finish_reason,
                "http_status": 200,
                "request_id": request_id,
            }
        except Exception as e:
            return {"ok": False, "code": type(e).__name__}


# --- Provider registry -------------------------------------------------------
# Add a provider = add ONE entry. Minimum is base_url + model + key_env (Bearer auth):
#   "myprovider": {"base_url": "https://.../v1", "model": "the-model", "key_env": "MY_API_KEY"}
# Optional keys for the rare cases:
#   base_url_env   read the base URL from an env var instead of hardcoding it
#   headers_env    custom auth headers {HeaderName: ENV_VAR} (instead of "Authorization: Bearer")
#   token_param    "max_completion_tokens" for OpenAI/gpt-5.x; default "max_tokens" everywhere else
#   extra_body     extra JSON fields sent on every request (e.g. {"reasoning_effort": "medium"})
#   session_header sticky-routing header for prefix-cache affinity
#
# The entries below are EXAMPLES showing each feature — replace them with your own. Or skip this
# table entirely: pass any name and set <NAME>_BASE_URL, <NAME>_MODEL, <NAME>_API_KEY in the
# environment (see the fallback in build_provider).
PROVIDERS = {
    # self-hosted SGLang/vLLM (Bearer auth, honors max_tokens):
    "local":  {"base_url": "http://localhost:30000/v1", "model": "your-model", "key_env": "LOCAL_API_KEY"},
    # hosted OpenAI-compatible provider (Bearer auth); URL from an env var:
    "hosted": {"base_url_env": "HOSTED_BASE_URL", "model": "your-model", "key_env": "HOSTED_API_KEY"},
    # OpenAI/gpt-5.x reasoning model (needs max_completion_tokens + a reasoning_effort):
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


def build_provider(name):
    """Build a provider from the PROVIDERS registry, or from <NAME>_* env vars if unknown."""
    e = _env()
    cfg = PROVIDERS.get(name)
    p = name.upper().replace("-", "_")
    if cfg is None:
        # Zero-code fallback: define it purely via environment variables.
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

    # session_header: registry > <NAME>_SESSION_HEADER > INFERBENCH_SESSION_HEADER > None
    # (--affinity then fills a default via ensure_affinity)
    session_header = (
        cfg.get("session_header")
        or e.get(f"{p}_SESSION_HEADER")
        or e.get("INFERBENCH_SESSION_HEADER")
    )
    model = e.get(f"{p}_MODEL", cfg["model"])
    return OpenAICompat(
        name, base, model, headers,
        extra_body=cfg.get("extra_body"),
        session_header=session_header,
        token_param=cfg.get("token_param", "max_tokens"),
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
