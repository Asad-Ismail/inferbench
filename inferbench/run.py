"""Closed-loop runner: concurrency ramp of multi-turn conversation lanes.

Measures achieved QPS/latency/cache-hit at fixed concurrency N (each lane waits for
a reply before the next turn) — useful for cache mode checks and "N concurrent chats"
UX, not for arrival-rate capacity or the goodput knee (use run_openloop.py for that).

Each lane advances ONE conversation sequentially (so the endpoint's prefix cache
can engage across turns). Ramps the number of concurrent lanes and reports the
measured cache-hit rate.

Usage:
  python run.py --provider myapi --role heavy --mode fixed  --levels 2,4,8,16
  python run.py --provider myapi --role heavy --mode broken --levels 2,4,8,16
  python run.py --provider myapi --role light --mode real --scale 0.1   # cheap dry-run

  --mode fixed  = stable prefix (best-case cache reuse)
  --mode broken = prefix mutated every turn (worst case: always miss)
  --mode real   = break at the per-role rate
  --scale       = multiply all token sizes (0.1 for a cheap smoke test; 1.0 = real weight)
"""
import argparse
import asyncio
import sys
import time
import random

import httpx

try:
    from inferbench import workload as W
    from inferbench.providers import build_provider, ensure_affinity
except ImportError:
    import workload as W
    from providers import build_provider, ensure_affinity

MAX_TURNS_PER_CONV = 8


def pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def lane(client, provider, role, mode, deadline, seed, stats, use_affinity):
    rng = random.Random(seed)
    conv = W.Conversation(role, rng)
    sid = f"conv-{seed}-0"
    conv_n = 0
    turns = 0
    while time.monotonic() < deadline:
        if turns >= MAX_TURNS_PER_CONV:
            conv_n += 1
            conv = W.Conversation(role, rng)
            sid = f"conv-{seed}-{conv_n}"
            turns = 0
        brk = W.should_break(role, rng, mode)
        messages, max_out = conv.next_turn(break_cache=brk)
        r = await provider.run_turn(client, messages, max_out,
                                    session_id=sid if use_affinity else None)
        turns += 1
        if not r.get("ok"):
            stats["errs"][r.get("code")] = stats["errs"].get(r.get("code"), 0) + 1
            continue
        stats["ok"] += 1
        stats["ttft"].append(r.get("ttft"))
        stats["e2e"].append(r.get("e2e"))
        pt, ct = r.get("prompt_tokens"), r.get("cached_tokens")
        if pt:
            stats["ptok"].append(pt)
            if ct is not None:
                stats["hit"].append(ct / pt)


async def run_level(provider, role, mode, lanes, dur, seed, use_affinity):
    stats = {"ok": 0, "errs": {}, "ttft": [], "e2e": [], "ptok": [], "hit": []}
    limits = httpx.Limits(max_connections=lanes + 8, max_keepalive_connections=lanes + 8)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        t0 = time.monotonic()
        deadline = t0 + dur
        await asyncio.gather(*[
            lane(client, provider, role, mode, deadline, seed + i, stats, use_affinity)
            for i in range(lanes)
        ])
        wall = time.monotonic() - t0
    qps = stats["ok"] / wall
    err = sum(stats["errs"].values())
    errstr = ",".join(f"{k}:{v}" for k, v in stats["errs"].items()) or "-"
    hit = pct(stats["hit"], 0.5)

    def f(x, d=1):
        return f"{x:.{d}f}" if x is not None else "-"
    print(f"lanes={lanes:3d} | qps={qps:6.2f} | ok={stats['ok']:4d} err={err:3d} [{errstr}] "
          f"| TTFT p50={f(pct(stats['ttft'],.5),2)}s p95={f(pct(stats['ttft'],.95),2)}s "
          f"| e2e p50={f(pct(stats['e2e'],.5),1)}s p95={f(pct(stats['e2e'],.95),1)}s "
          f"| cache_hit={f(hit*100,0) if hit is not None else '-'}% "
          f"| prompt_tok p50={f(pct(stats['ptok'],.5),0)}", flush=True)
    return qps


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--role", choices=sorted(W.ROLE_PROFILES), default="heavy")
    ap.add_argument("--mode", choices=["fixed", "broken", "real"], default="real")
    ap.add_argument("--levels", default="2,4,8,16,24,32")
    ap.add_argument("--dur", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--affinity", action="store_true",
                    help="send a session id (sticky-routing header) for cache affinity")
    a = ap.parse_args()
    W.SIZE_SCALE = a.scale
    try:
        prov = build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    use_affinity = ensure_affinity(prov, a.affinity)
    levels = [int(x) for x in a.levels.split(",")]
    print(f"inferbench | provider={a.provider} role={a.role} mode={a.mode} "
          f"affinity={use_affinity} scale={a.scale} | {a.dur}s/level | lanes=multi-turn conversations\n",
          flush=True)
    for lv in levels:
        await run_level(prov, a.role, a.mode, lv, a.dur, a.seed, use_affinity)
    print("\n---DONE---", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
