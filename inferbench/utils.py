"""Shared helpers: percentiles, arrivals, prompt text, table formatting."""
import math
import os
import random

MAX_TURNS = 8

VOCAB = (
    "agent ticket repository state tool plan policy commit diff branch file module "
    "function request handler cache token context window prefix suffix latency queue "
    "replica container prompt system message reason review deploy config schema route "
    "worker session affinity throughput prefill decode buffer scale limit error retry "
    "customer order invoice search fetch write read update delete query index vector"
).split()


def run_salt():
    """Per-run salt so seeded text is new to the endpoint. Pin with SALT=..."""
    return os.environ.get("SALT") or os.urandom(6).hex()


def pct(xs, p):
    """Linear-interpolated percentile. None values dropped; empty → None."""
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def next_interval(qps, arrival, rng):
    """Seconds to the next arrival. uniform = even spacing; poisson = exponential."""
    if arrival == "uniform":
        return 1.0 / qps
    return -math.log(1.0 - rng.random()) / qps


def seeded_text(n_tokens, seed, salt):
    """Deterministic ~n_tokens words from (salt, seed). Same seed → same bytes."""
    rng = random.Random(str((salt, seed)))
    n = max(1, int(n_tokens))
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def error_str(errors, empty="0"):
    return ",".join(f"{k}:{v}" for k, v in errors.items()) or empty


def fmt_ms(x):
    return f"{x * 1000:.0f}" if x is not None else "-"


def fmt_k(x):
    return f"{x / 1000:.0f}k" if x is not None else "-"


def fmt_num(x, digits=0):
    return f"{x:.{digits}f}" if x is not None else "-"
