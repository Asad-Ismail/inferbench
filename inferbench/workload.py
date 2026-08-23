"""Long-context, multi-turn size distribution and prefix-cache composition.

Replace ROLE_PROFILES (quantile, tokens) points to match your traffic.
  total  = full context per turn
  output = completion tokens per turn
"""
import hashlib
import math
import os
import random

try:
    from inferbench.utils import VOCAB
except ImportError:
    from utils import VOCAB

ROLE_PROFILES = {
    "heavy": {
        "total":   [(0.0, 2000), (0.05, 13000), (0.25, 32000), (0.5, 80000), (0.75, 160000),
                    (0.90, 270000), (0.95, 350000), (0.99, 500000), (1.0, 780000)],
        "output":  [(0.0, 20), (0.5, 230), (0.95, 2300), (0.99, 5600), (1.0, 35000)],
        "break_rate": 0.12,
    },
    "light": {
        "total":   [(0.0, 1000), (0.5, 68000), (0.9, 156000), (0.99, 250000), (1.0, 400000)],
        "output":  [(0.0, 20), (0.5, 630), (0.95, 2000), (1.0, 6000)],
        "break_rate": 0.02,
    },
}

SIZE_SCALE = 1.0  # set < 1 for cheap dry runs
MAX_CTX = 0       # clamp sampled totals; 0 = no clamp
FIXED_PREFIX_TOKENS = 25000
# Designed warm-turn cache fraction. A stable-prefix-only model hits ~95% (too high).
CACHE_FRACTION = float(os.environ.get("INFERBENCH_CACHE_FRACTION", "0.59"))

_FIXED_PREFIX_WORDS = None


def clamp_ctx(tokens):
    if MAX_CTX and MAX_CTX > 0:
        return min(float(tokens), float(MAX_CTX))
    return float(tokens)


def quantile_at(points, u):
    """Log-interpolated value at quantile u in [0, 1]."""
    for i in range(1, len(points)):
        q0, v0 = points[i - 1]
        q1, v1 = points[i]
        if u <= q1:
            f = (u - q0) / (q1 - q0) if q1 > q0 else 0.0
            lo, hi = math.log(max(v0, 1)), math.log(max(v1, 1))
            return math.exp(lo + f * (hi - lo))
    return float(points[-1][1])


def sample_pct(points, rng):
    return quantile_at(points, rng.random())


def stratified_totals(role, n):
    """n sizes spanning the role distribution, ordered so every prefix has ~the mean.

    A short level still sees InMean ≈ the full distribution, so levels compare load, not sample weight.
    """
    vals = [clamp_ctx(quantile_at(ROLE_PROFILES[role]["total"], (i + 0.5) / n)) for i in range(n)]
    target = sum(vals) / len(vals)
    remaining, ordered, s = list(vals), [], 0.0
    for k in range(len(vals)):
        best = min(remaining, key=lambda v: abs((s + v) / (k + 1) - target))
        ordered.append(best)
        remaining.remove(best)
        s += best
    return ordered


def diverse_text(approx_tokens, rng):
    """Non-repeating text so two conversations do not share cacheable blocks."""
    n = max(1, int(approx_tokens * SIZE_SCALE))
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def _fixed_words():
    global _FIXED_PREFIX_WORDS
    if _FIXED_PREFIX_WORDS is None:
        r = random.Random(424242)
        _FIXED_PREFIX_WORDS = [r.choice(VOCAB) for _ in range(FIXED_PREFIX_TOKENS)]
    return _FIXED_PREFIX_WORDS


def fixed_prefix(n_tokens=FIXED_PREFIX_TOKENS):
    """Leading slice of the global system prefix (so small requests still share a prefix)."""
    words = _fixed_words()
    k = max(0, min(len(words), int(n_tokens * SIZE_SCALE)))
    return "[SYSTEM PROMPT + TOOL SCHEMA] " + " ".join(words[:k])


class Conversation:
    """Stable prefix + per-turn fresh block. Cache-break prepends a nonce to the system prompt."""

    def __init__(self, role, rng, total=None):
        assert role in ROLE_PROFILES
        self.role = role
        self.rng = rng
        prof = ROLE_PROFILES[role]
        self.total = clamp_ctx(sample_pct(prof["total"], rng) if total is None else total)
        cache_budget = int(CACHE_FRACTION * self.total)
        self.fixed_tokens = min(FIXED_PREFIX_TOKENS, cache_budget)
        self.stable_tokens = max(0, cache_budget - self.fixed_tokens)
        self.fresh_tokens = max(1, int((1.0 - CACHE_FRACTION) * self.total))
        self.stable_block = diverse_text(self.stable_tokens, rng) if self.stable_tokens else ""
        self.turn = 0
        self.nonce = ""

    def next_turn(self, break_cache=False):
        self.turn += 1
        if break_cache:
            self.nonce = hashlib.md5(f"{self.turn}:{self.rng.random()}".encode()).hexdigest()
        system = (self.nonce + "\n" if self.nonce else "") + fixed_prefix(self.fixed_tokens)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "[context] " + self.stable_block},
            {"role": "assistant", "content": "(ack — continuing)"},
            {"role": "user", "content": f"[turn {self.turn}] " + diverse_text(self.fresh_tokens, self.rng)},
        ]
        max_out = int(sample_pct(ROLE_PROFILES[self.role]["output"], self.rng))
        return messages, max_out


def should_break(role, rng, mode):
    """fixed = never, broken = always, real = per-role rate."""
    if mode == "fixed":
        return False
    if mode == "broken":
        return True
    return rng.random() < ROLE_PROFILES[role]["break_rate"]
