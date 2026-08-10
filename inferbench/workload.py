"""Workload generator for a long-context, multi-turn agent chat load.

Import this and drive it against any OpenAI-compatible endpoint (see providers.py). It reproduces:
  1. the per-role request SIZE distribution (heavy-tailed), and
  2. the multi-turn CACHING pattern: a stable system prefix that an endpoint's
     prefix-cache can reuse across turns, plus controllable cache-BREAKS (a nonce
     injected into the prefix forces a full re-prefill).

Provenance: the percentiles below are a synthetic long-context agent-style profile.
Replace them with your own (quantile, tokens) points to match a specific workload.
  total  = full context per turn (tokens)
  output = completion tokens per turn
"""
import os
import math
import hashlib
import random

# (quantile, tokens) points; sampler interpolates in log-space between them.
# Named size profiles (quantile → tokens). Edit or add entries to match your traffic.
ROLE_PROFILES = {
    "heavy": {  # long-context, heavy-tailed input
        "total":   [(0.0, 2000), (0.05, 13000), (0.25, 32000), (0.5, 80000), (0.75, 160000),
                    (0.90, 270000), (0.95, 350000), (0.99, 500000), (1.0, 780000)],
        "output":  [(0.0, 20), (0.5, 230), (0.95, 2300), (0.99, 5600), (1.0, 35000)],
        "break_rate": 0.12,
    },
    "light": {  # shorter / lighter input distribution
        "total":   [(0.0, 1000), (0.5, 68000), (0.9, 156000), (0.99, 250000), (1.0, 400000)],
        "output":  [(0.0, 20), (0.5, 630), (0.95, 2000), (1.0, 6000)],
        "break_rate": 0.02,
    },
}

# Multiply all token sizes by this to run cheap dry-runs (1.0 = real weight).
SIZE_SCALE = 1.0


def quantile_at(points, u):
    """Value at quantile u in [0,1] of an empirical (quantile, value) dist, log-interpolated."""
    for i in range(1, len(points)):
        q0, v0 = points[i - 1]
        q1, v1 = points[i]
        if u <= q1:
            f = (u - q0) / (q1 - q0) if q1 > q0 else 0.0
            lo, hi = math.log(max(v0, 1)), math.log(max(v1, 1))
            return math.exp(lo + f * (hi - lo))
    return float(points[-1][1])


def sample_pct(points, rng):
    """Random draw from the (quantile, value) distribution."""
    return quantile_at(points, rng.random())


def ladder_totals(rungs, n, seed=0):
    """A FIXED, deterministic list of n request sizes dealt from `rungs`, where every
    consecutive block of len(rungs) entries contains each rung exactly once, shuffled within
    the block so the order is not a repeating pattern. n is rounded up to a whole number of
    blocks, so cycling the list keeps the mix exactly balanced for any request count.

    Differs from stratified_totals, which samples evenly-spaced QUANTILES and therefore
    inherits the distribution's shape -- the tail stays as rare as it is in production (~4%
    above 400k on the default profile). A ladder gives every rung an EQUAL share instead.
    Use it when the largest requests dominate the SLO and the natural tail is too thin to
    measure: it fixes both the "each level rolled its own dice" ambiguity and the "the
    slowest traffic is the least sampled" problem in one step.

    Cost: the size mix is deliberately NOT representative of production traffic, so pooled
    medians from a ladder run describe the ladder. Compare per-rung, not pooled.
    """
    rungs = [float(r) for r in rungs]
    if not rungs:
        raise ValueError("ladder_totals needs at least one rung")
    k = len(rungs)
    blocks = max(1, -(-int(n) // k))
    out = []
    for b in range(blocks):
        deck = list(rungs)
        random.Random(f"ladder:{seed}:{b}").shuffle(deck)
        out += deck
    return out


def stratified_totals(role, n):
    """A FIXED, deterministic list of n request sizes spanning the distribution (evenly-spaced
    quantiles, tail included), reordered so that EVERY PREFIX already averages ~the distribution
    mean. Greedy mean-stabilization: at each step pick the remaining size that keeps the running
    mean closest to target. Consequence: a level that completes only N requests still sees mean
    input ~= the full distribution mean, for ANY N -- so InMean is ~constant across levels
    regardless of how many each completes, and level comparisons reflect load, not sample weight.
    The full pass is still the exact stratified distribution; the ordering just balances big vs
    small early."""
    vals = [quantile_at(ROLE_PROFILES[role]["total"], (i + 0.5) / n) for i in range(n)]
    target = sum(vals) / len(vals)
    remaining, ordered, s = list(vals), [], 0.0
    for k in range(len(vals)):
        best = min(remaining, key=lambda v: abs((s + v) / (k + 1) - target))
        ordered.append(best); remaining.remove(best); s += best
    return ordered


_VOCAB = ("agent ticket repository state tool plan policy commit diff branch file module "
          "function request handler cache token context window prefix suffix latency queue "
          "replica container prompt system message reason review deploy config schema route "
          "worker session affinity throughput prefill decode buffer scale limit error retry "
          "customer order invoice search fetch write read update delete query index vector").split()


def diverse_text(approx_tokens, rng):
    """~approx_tokens tokens of NON-repeating text from an rng, so different conversations
    do NOT share cacheable blocks (the repeated-lorem version caused fake cache hits)."""
    n = max(1, int(approx_tokens * SIZE_SCALE))
    return " ".join(rng.choice(_VOCAB) for _ in range(n))


# Shared system/tool-schema prefix: identical across conversations (fixed seed) so the
# endpoint can cache it once globally. Main cross-request cache lever.
FIXED_PREFIX_TOKENS = 25000

# CALIBRATION: fraction of each request served FROM CACHE. We size a stable per-conv block to
# CACHE_FRACTION and a NEW fresh block to (1-CACHE_FRACTION) each turn, so the endpoint's
# prefix cache naturally measures ~this hit rate. A pure stable-prefix + tiny-increment model
# gives ~95% hit (over-optimistic). Set to match your traffic.
CACHE_FRACTION = float(os.environ.get("INFERBENCH_CACHE_FRACTION", "0.59"))


_FIXED_PREFIX_WORDS = None


def _fixed_words():
    global _FIXED_PREFIX_WORDS
    if _FIXED_PREFIX_WORDS is None:
        r = random.Random(424242)
        _FIXED_PREFIX_WORDS = [r.choice(_VOCAB) for _ in range(FIXED_PREFIX_TOKENS)]
    return _FIXED_PREFIX_WORDS


def fixed_prefix(n_tokens=FIXED_PREFIX_TOKENS):
    """First n_tokens of the canonical global system prefix. Truncating to a leading slice
    (for small requests whose cache budget < FIXED_PREFIX_TOKENS) keeps it a prefix of the
    same string, so the endpoint's prefix cache still hits across requests."""
    words = _fixed_words()
    k = max(0, min(len(words), int(n_tokens * SIZE_SCALE)))
    return "[SYSTEM PROMPT + TOOL SCHEMA] " + " ".join(words[:k])


class Conversation:
    """One multi-turn conversation.

    A big STABLE system prefix (the cacheable bulk) + small per-turn appends. Each
    next_turn() returns the full message list for that turn plus a max_tokens budget.
    On a cache-break, a nonce is prepended to the system prompt so the endpoint's
    prefix cache misses and it must re-prefill the whole context.
    """

    def __init__(self, role, rng, total=None):
        assert role in ROLE_PROFILES
        self.role = role
        self.rng = rng
        prof = ROLE_PROFILES[role]
        # total: if given (fixed deterministic schedule) use it; else draw randomly. Content is
        # always unique (per-conv rng), so cache-hit stays the designed CACHE_FRACTION either way.
        total = sample_pct(prof["total"], rng) if total is None else float(total)
        self.total = total
        # Each turn = [FIXED prefix (global cache)] + [per-conv STABLE block] + [FRESH block].
        # FIXED + STABLE are byte-identical across this conv's turns (cacheable with affinity).
        # FRESH is new every turn = the (1-CACHE_FRACTION) the GPU must prefill. Context stays
        # ~total (old fresh dropped) instead of growing unbounded. Fixed prefix shrinks when the
        # cache budget is smaller than FIXED_PREFIX_TOKENS.
        cache_budget = int(CACHE_FRACTION * total)
        self.fixed_tokens = min(FIXED_PREFIX_TOKENS, cache_budget)   # shrinks for small requests
        self.stable_tokens = max(0, cache_budget - self.fixed_tokens)
        self.fresh_tokens = max(1, int((1.0 - CACHE_FRACTION) * total))
        self.stable_block = diverse_text(self.stable_tokens, rng) if self.stable_tokens else ""
        self.turn = 0
        self.nonce = ""

    def next_turn(self, break_cache=False):
        self.turn += 1
        if break_cache:
            # nonce in the fixed prefix -> whole cache misses
            self.nonce = hashlib.md5(f"{self.turn}:{self.rng.random()}".encode()).hexdigest()
        system = (self.nonce + "\n" if self.nonce else "") + fixed_prefix(self.fixed_tokens)   # SHARED prefix (<=25k)
        # STABLE context: identical every turn of this conv -> cache hit on turns 2+ (via affinity).
        stable_user = "[context] " + self.stable_block
        # FRESH content: new every turn -> always uncached.
        fresh_user = f"[turn {self.turn}] " + diverse_text(self.fresh_tokens, self.rng)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": stable_user},
            {"role": "assistant", "content": "(ack — continuing)"},
            {"role": "user", "content": fresh_user},
        ]
        max_out = int(sample_pct(ROLE_PROFILES[self.role]["output"], self.rng))
        return messages, max_out


def should_break(role, rng, mode):
    """mode: 'fixed' (never break -> best-case cache reuse),
             'broken' (always break -> worst-case full re-prefill),
             'real'  (break at the per-role rate)."""
    if mode == "fixed":
        return False
    if mode == "broken":
        return True
    return rng.random() < ROLE_PROFILES[role]["break_rate"]
