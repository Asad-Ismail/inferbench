"""Reproducible scaling probe: empirically measure how prefill latency (TTFT) and
throughput depend on (1) input context length and (2) prefix-cache hit rate.

This isolates the PREFILL cost from decode by sending max_tokens=8 (enough for a first
token on reasoning models; decode is negligible vs prefill), so TTFT ~= prefill time.
Two sweeps, each fully parameterized + seeded so a reviewer gets the same curve:

  context  : hold cache-hit fixed (default 0 = all-fresh), sweep input length.
             -> fits TTFT(L) = a*L + b*L^2, reports the empirical linear->quadratic
                crossover L* = a/b and compares it to the theory value 12*d_model.
  cachehit : hold total context fixed, sweep the cached fraction.
             -> shows fresh-prefill work (and TTFT) scales with (1 - hit), i.e. QPS ~ 1/(1-hit).

Design choices that make it reproducible:
  * All prompt text is generated from explicit seeds -> identical bytes run-to-run,
    so the endpoint's RadixAttention cache behaves identically.
  * A "cached" block is byte-stable and sent under a fixed session id (affinity) with a
    warmup call first, so the endpoint has it in cache before we measure.
  * A "fresh" block is unique per measurement -> guaranteed cache miss.
  * We report BOTH the intended cache-hit and the endpoint-measured cached/prompt ratio,
    so you can see the workload actually hit the intended composition.

Usage:
  python3 scaling_probe.py --provider myapi --sweep context \
      --lengths 4000,8000,16000,32000,64000,128000,256000 --repeats 3 --out ctx.csv
  python3 scaling_probe.py --provider myapi --sweep cachehit \
      --context 200000 --hits 0,0.25,0.5,0.59,0.75,0.9,0.95 --repeats 3 --out hit.csv

Requires: providers.py (build_provider). Optional: numpy for the quadratic fit.
"""
import argparse, asyncio, random, statistics, csv, sys, os
import httpx

try:
    from inferbench.providers import build_provider, ensure_affinity
except ImportError:
    from providers import build_provider, ensure_affinity

# Per-RUN salt: mixed into ALL generated text so every run sends content the endpoint has
# NEVER seen -> a true cache-miss baseline (deterministic seeds alone would hit a prior run's
# cache and silently inflate cache-hit). Sizes stay seed-deterministic; what reproduces is the
# scaling RELATIONSHIP + crossover, not literal bytes/latencies. Override with SALT=... to pin.
RUN_SALT = os.environ.get("SALT") or os.urandom(6).hex()

# Synthetic vocab: one word ~= one token for sizing; verify against endpoint prompt_tokens.
_VOCAB = ("agent ticket repository state tool plan policy commit diff branch file module "
          "function request handler cache token context window prefix suffix latency queue "
          "replica container prompt system message reason review deploy config schema route "
          "worker session affinity throughput prefill decode buffer scale limit error retry "
          "customer order invoice search fetch write read update delete query index vector").split()

# Optional theory crossover line (12 * d_model). Set to your model's hidden size.
D_MODEL = int(os.environ.get("INFERBENCH_D_MODEL", "4096"))
THEORY_CROSSOVER = 12 * D_MODEL


def block(approx_tokens, seed):
    """Deterministic ~approx_tokens-token text from `seed` (same seed -> identical bytes).
    seed may be any hashable (tuple/int/str); stringified + run-salted so random.Random accepts
    it AND every run produces globally-unique bytes (true cache-miss baseline)."""
    rng = random.Random(str((RUN_SALT, seed)))
    n = max(1, int(approx_tokens))
    return " ".join(rng.choice(_VOCAB) for _ in range(n))


def make_messages(total_tokens, cache_fraction, stable_seed, fresh_seed):
    """Prompt = [stable cached prefix (cache_fraction)] + [fresh suffix (1-cache_fraction)].
    stable_seed is constant for a probe point (so it caches under affinity); fresh_seed is
    unique per call (so it always misses)."""
    cached = int(round(cache_fraction * total_tokens))
    fresh = max(1, total_tokens - cached)
    msgs = []
    if cached > 0:
        msgs.append({"role": "system", "content": "[STABLE] " + block(cached, stable_seed)})
    msgs.append({"role": "user", "content": "[FRESH] " + block(fresh, fresh_seed)})
    return msgs, cached, fresh


async def measure(client, prov, total, hit, sid, stable_seed, repeats, warmup, call_ctr):
    """Warm the cached prefix, then take `repeats` prefill-latency samples. Returns dict of
    medians + the endpoint-measured cache-hit (to prove the intended composition landed)."""
    # max_tokens=8: enough for a reasoning model to stream a first token so TTFT is captured;
    # decode of a few tokens is negligible vs prefill, so TTFT ~= prefill time.
    MT = 8
    # warmup: same stable prefix, throwaway fresh part -> loads cached block into the cache.
    for _ in range(warmup):
        call_ctr[0] += 1
        m, _, _ = make_messages(total, hit, stable_seed, ("warm", call_ctr[0]))
        await prov.run_turn(client, m, MT, session_id=sid)
    ttfts, measured_hits, ptoks = [], [], []
    for _ in range(repeats):
        call_ctr[0] += 1
        m, cached_intended, fresh = make_messages(total, hit, stable_seed, ("meas", call_ctr[0]))
        r = await prov.run_turn(client, m, MT, session_id=sid)
        if not r.get("ok"):
            print(f"    ! error total={total} hit={hit}: {r.get('code')}", flush=True)
            continue
        if r.get("ttft") is None:
            continue
        ttfts.append(r.get("ttft"))
        pt, ct = r.get("prompt_tokens"), r.get("cached_tokens")
        if pt:
            ptoks.append(pt)
            if ct is not None:
                measured_hits.append(ct / pt)
    if not ttfts:
        return None
    return {
        "total_tokens": total,
        "intended_hit": hit,
        "measured_hit": statistics.median(measured_hits) if measured_hits else None,
        "prompt_tokens": statistics.median(ptoks) if ptoks else None,
        "fresh_tokens_est": None,  # filled by caller from measured prompt_tokens
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_min_s": min(ttfts),
        "ttft_max_s": max(ttfts),
        "n": len(ttfts),
    }


def quad_fit(xs, ys):
    """Fit y = c + a*x + b*x^2 (c = fixed network/queue overhead). Returns (c,a,b,cross,r2).
    Crossover L* = a/b is where the linear and quadratic prefill terms are equal."""
    import numpy as np  # required for the 3-param fit; install numpy to reproduce
    X = np.array([[1.0, x, x * x] for x in xs], dtype=float)
    Y = np.array(ys, dtype=float)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    c, a, b = float(coef[0]), float(coef[1]), float(coef[2])
    pred = X @ coef
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    cross = a / b if b > 0 else float("inf")
    return c, a, b, cross, r2


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--sweep", choices=["context", "cachehit"], required=True)
    ap.add_argument("--lengths", default="4000,8000,16000,32000,64000,128000,256000",
                    help="context-sweep input lengths (tokens)")
    ap.add_argument("--cache-hit", type=float, default=0.0, help="fixed cache-hit for context sweep")
    ap.add_argument("--context", type=int, default=200000, help="fixed total context for cachehit sweep")
    ap.add_argument("--hits", default="0,0.25,0.5,0.59,0.75,0.9,0.95", help="cachehit-sweep hit fractions")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None, help="CSV output path")
    a = ap.parse_args()

    try:
        prov = build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(2)
    ensure_affinity(prov, True)  # cachehit / warm prefix needs sticky routing
    call_ctr = [0]
    rng = random.Random(a.seed)
    rows = []
    print(f"SCALING PROBE | provider={a.provider} sweep={a.sweep} repeats={a.repeats} "
          f"warmup={a.warmup} seed={a.seed} | max_tokens=8 (prefill-isolated)\n", flush=True)

    async with httpx.AsyncClient(timeout=900, limits=httpx.Limits(max_connections=8)) as client:
        if a.sweep == "context":
            lengths = [int(x) for x in a.lengths.split(",")]
            print(f"context sweep @ fixed cache-hit={a.cache_hit}; theory crossover 12*d_model={THEORY_CROSSOVER}")
            print(f"{'ctx_tok':>10} | {'prompt_tok':>10} | {'meas_hit':>8} | {'TTFT_p50_s':>10} | "
                  f"{'min':>6} | {'max':>6} | n")
            print("-" * 78)
            for L in lengths:
                sid = f"probe-ctx-{L}-{rng.randint(0,10**9)}"
                res = await measure(client, prov, L, a.cache_hit, sid, ("ctx", L), a.repeats, a.warmup, call_ctr)
                if res:
                    rows.append(res)
                    print(f"{L:>10} | {str(res['prompt_tokens']):>10} | "
                          f"{(res['measured_hit'] or 0):>8.3f} | {res['ttft_p50_s']:>10.3f} | "
                          f"{res['ttft_min_s']:>6.2f} | {res['ttft_max_s']:>6.2f} | {res['n']}", flush=True)
            # fit against endpoint-measured FRESH tokens (prompt_tokens*(1-measured_hit))
            xs = [r["prompt_tokens"] * (1 - (r["measured_hit"] or 0)) for r in rows if r["prompt_tokens"]]
            ys = [r["ttft_p50_s"] for r in rows if r["prompt_tokens"]]
            if len(xs) >= 4:
                c_, a_, b_, cross, r2 = quad_fit(xs, ys)
                print(f"\nfit TTFT = c + a*Lfresh + b*Lfresh^2 : c={c_:.3f}s (fixed overhead)  "
                      f"a={a_:.3e}s/tok  b={b_:.3e}s/tok^2")
                print(f"  empirical crossover L* = a/b = {cross:,.0f} fresh tokens   (theory 12*d_model={THEORY_CROSSOVER:,})")
                print(f"  R^2 = {r2:.4f}")
                print(f"  interpretation: below L* prefill (and 1/QPS) is ~LINEAR in tokens; above L* the "
                      f"b*L^2 attention term dominates -> ~QUADRATIC.")
        else:
            hits = [float(x) for x in a.hits.split(",")]
            print(f"cachehit sweep @ fixed context={a.context} tokens")
            print(f"{'intended_hit':>12} | {'meas_hit':>8} | {'prompt_tok':>10} | {'fresh_tok':>9} | "
                  f"{'TTFT_p50_s':>10} | n")
            print("-" * 74)
            for h in hits:
                sid = f"probe-hit-{h}-{rng.randint(0,10**9)}"
                res = await measure(client, prov, a.context, h, sid, ("hit", a.context), a.repeats, a.warmup, call_ctr)
                if res:
                    fresh_meas = (res["prompt_tokens"] or 0) * (1 - (res["measured_hit"] or 0))
                    res["fresh_tokens_est"] = fresh_meas
                    rows.append(res)
                    print(f"{h:>12.2f} | {(res['measured_hit'] or 0):>8.3f} | "
                          f"{str(res['prompt_tokens']):>10} | {fresh_meas:>9.0f} | "
                          f"{res['ttft_p50_s']:>10.3f} | {res['n']}", flush=True)
            # show TTFT vs (1-hit): expect ~proportional
            print(f"\nexpect TTFT ~ proportional to fresh tokens = (1-hit)*context "
                  f"-> QPS ~ 1/(1-hit).")
            if len(rows) >= 2:
                ref = min(rows, key=lambda r: r["intended_hit"])  # the lowest-hit (most fresh) point
                print(f"{'intended_hit':>12} | {'fresh_ratio_vs_h0':>17} | {'ttft_ratio_vs_h0':>16}")
                print("-" * 52)
                for r in rows:
                    fr = (r["fresh_tokens_est"] or 1) / (ref["fresh_tokens_est"] or 1)
                    tr = r["ttft_p50_s"] / ref["ttft_p50_s"]
                    print(f"{r['intended_hit']:>12.2f} | {fr:>17.3f} | {tr:>16.3f}", flush=True)

    if a.out and rows:
        keys = ["total_tokens", "intended_hit", "measured_hit", "prompt_tokens",
                "fresh_tokens_est", "ttft_p50_s", "ttft_min_s", "ttft_max_s", "n"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\nwrote {len(rows)} rows -> {a.out}", flush=True)
    print("\n---DONE---", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
