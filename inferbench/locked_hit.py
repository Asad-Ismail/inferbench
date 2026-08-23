"""Ramp at a pinned cache-hit.

Pre-primes a conversation pool so every request is [cached prefix] + [fresh suffix]
at the target hit. Fresh text is per-run salted, so it never caches.

This is not a capacity / knee finder. If the pool is busy it waits (closed-loop).
ok/wall includes drain.

  python3 locked_hit.py --provider myapi --hit 0.59 --max-ctx 128000 --levels 0.05,0.1,0.15,0.2 --dur 90
"""
import argparse
import asyncio
import math
import random
import sys
import time

import httpx

try:
    from inferbench import workload as W
    from inferbench.providers import build_provider, ensure_affinity
    from inferbench.utils import (
        error_str, fmt_ms, fmt_num, next_interval, pct, run_salt, seeded_text,
    )
except ImportError:
    import workload as W
    from providers import build_provider, ensure_affinity
    from utils import error_str, fmt_ms, fmt_num, next_interval, pct, run_salt, seeded_text

SALT = run_salt()


class PConv:
    """Pre-primed conversation with a cached prefix of hit * ctx tokens."""

    def __init__(self, i, ctx, hit):
        self.i = i
        self.ctx = int(ctx)
        self.cached = max(1, int(round(hit * self.ctx)))
        self.fresh = max(1, self.ctx - self.cached)
        self.stable = "[STABLE] " + seeded_text(self.cached, ("stable", i), SALT)
        self.sid = f"lock-{i}"
        self.k = 0
        self.out_rng = random.Random(str(("out", i)))

    def prime_messages(self):
        return [
            {"role": "system", "content": self.stable},
            {"role": "user", "content": "[FRESH] " + seeded_text(min(self.fresh, 400), ("prime", self.i), SALT)},
        ]

    def turn_messages(self):
        self.k += 1
        return [
            {"role": "system", "content": self.stable},
            {"role": "user", "content": "[FRESH] " + seeded_text(self.fresh, ("f", self.i, self.k), SALT)},
        ]

    def max_out(self):
        return int(W.sample_pct(W.ROLE_PROFILES["heavy"]["output"], self.out_rng))


async def prime_pool(client, prov, pool, conc):
    sem = asyncio.Semaphore(conc)
    failed = []

    async def one(c):
        async with sem:
            r = await prov.run_turn(client, c.prime_messages(), 8, session_id=c.sid)
            if not r.get("ok"):
                failed.append((c.sid, r.get("code")))

    await asyncio.gather(*[one(c) for c in pool])
    if failed:
        codes = {}
        for _, code in failed:
            codes[code] = codes.get(code, 0) + 1
        print(f"error: priming failed for {len(failed)}/{len(pool)} conversations: {codes}",
              file=sys.stderr)
        sys.exit(2)


async def run_level(client, prov, pool, qps, dur, rng, arrival, drain_grace=300.0):
    st = {"ok": 0, "errs": {}, "ttft": [], "e2e": [], "hit": [], "samples": [], "tps": [], "otok": []}
    t0 = time.monotonic()
    free = list(pool)

    async def one(c):
        try:
            r = await prov.run_turn(client, c.turn_messages(), c.max_out(), session_id=c.sid)
            if r.get("ok"):
                st["ok"] += 1
                st["ttft"].append(r.get("ttft"))
                st["e2e"].append(r.get("e2e"))
                st["samples"].append((time.monotonic() - t0, r.get("e2e")))
                pt, ct = r.get("prompt_tokens"), r.get("cached_tokens")
                if pt and ct is not None:
                    st["hit"].append(ct / pt)
                ttft, e2e, ot = r.get("ttft"), r.get("e2e"), r.get("completion_tokens")
                if ot:
                    st["otok"].append(ot)
                if ttft is not None and e2e is not None and ot and e2e > ttft and ot > 1:
                    st["tps"].append(ot / (e2e - ttft))
            else:
                st["errs"][r.get("code")] = st["errs"].get(r.get("code"), 0) + 1
        finally:
            free.append(c)

    if qps <= 0:
        raise ValueError(f"qps must be > 0, got {qps}")
    tasks = []
    deadline = t0 + dur
    nxt = 0.0
    while time.monotonic() < deadline:
        nxt += next_interval(qps, arrival, rng)
        target = t0 + nxt
        now = time.monotonic()
        if target > now:
            await asyncio.sleep(target - now)
        while not free and time.monotonic() < deadline:
            await asyncio.sleep(0.001)
        if not free:
            break
        c = free.pop(rng.randrange(len(free)))
        tasks.append(asyncio.create_task(one(c)))
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=drain_grace)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        st["cancelled"] = len(pending)
    wall = time.monotonic() - t0
    sustained = st["ok"] / wall
    err = sum(st["errs"].values())
    backlog = st.get("cancelled", 0)
    blk = f" backlog_cancelled={backlog}" if backlog else ""
    print(f"{qps:>10.2f} | {sustained:>9.3f} | {fmt_ms(pct(st['ttft'],.5)):>7s} | {fmt_ms(pct(st['ttft'],.95)):>7s} | "
          f"{fmt_ms(pct(st['ttft'],.99)):>7s} | {fmt_ms(pct(st['e2e'],.5)):>7s} | {fmt_ms(pct(st['e2e'],.95)):>7s} | "
          f"{fmt_ms(pct(st['e2e'],.99)):>7s} | {fmt_num(pct(st['tps'],.5)):>6s} | {fmt_num(pct(st['otok'],.5)):>6s} | "
          f"{(pct(st['hit'],.5) or 0):>6.3f} | {error_str(st['errs']):>6s}{blk}", flush=True)
    drift = None
    if wall >= 300 and len(st["samples"]) >= 12:
        third = wall / 3.0
        buckets = [[e for (tc, e) in st["samples"] if lo <= tc < lo + third] for lo in (0, third, 2 * third)]
        p95s = [pct(b, .95) for b in buckets if b]
        cnts = [len(b) for b in buckets]
        if len(p95s) == 3:
            verdict = "STEADY" if p95s[2] <= 1.5 * p95s[0] else "GROWING"
            print(f"           drift E2E p95 by third (n={cnts}): "
                  f"{p95s[0]:.1f}s -> {p95s[1]:.1f}s -> {p95s[2]:.1f}s  => {verdict}", flush=True)
            drift = {"p95_thirds": p95s, "verdict": verdict}
    return {"qps": qps, "sustained": sustained, "e2e_p95": pct(st["e2e"], .95),
            "hit": pct(st["hit"], .5), "err": err, "n": st["ok"], "drift": drift}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--hit", type=float, required=True)
    ap.add_argument("--levels", default="0.05,0.1,0.15,0.2,0.25,0.3")
    ap.add_argument("--dur", type=int, default=90)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arrival", default="uniform", choices=["uniform", "poisson"])
    ap.add_argument("--uniform-ctx", action="store_true",
                    help="every request = distribution mean (no tail)")
    ap.add_argument("--max-ctx", "--max-cap", dest="max_ctx", type=int, default=0,
                    help="clamp every request ctx to this max (model context window)")
    ap.add_argument("--target-mean", type=int, default=0,
                    help="with --max-ctx, scale the sub-cap dist so the mean hits this")
    ap.add_argument("--timeout", type=float, default=1800,
                    help="per-request HTTP timeout in seconds; 0 disables")
    ap.add_argument("--drain-grace", type=float, default=300.0)
    a = ap.parse_args()
    try:
        prov = build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    ensure_affinity(prov, True)
    rng = random.Random(a.seed)
    pts = W.ROLE_PROFILES["heavy"]["total"]
    qat = W.quantile_at
    if a.uniform_ctx:
        mean_ctx = int(sum(qat(pts, (i + 0.5) / 400) for i in range(400)) / 400)
        if a.max_ctx:
            mean_ctx = min(mean_ctx, a.max_ctx)
        pool = [PConv(i, mean_ctx, a.hit) for i in range(a.pool)]
        mode = f"UNIFORM-ctx={mean_ctx:,}"
    elif a.max_ctx:
        cap = a.max_ctx
        s = 1.0
        if a.target_mean:
            lo, hi = 0.1, 20.0
            for _ in range(60):
                mid = (lo + hi) / 2
                m = sum(min(cap, mid * qat(pts, (i + 0.5) / 2000)) for i in range(2000)) / 2000
                if m < a.target_mean:
                    lo = mid
                else:
                    hi = mid
            s = (lo + hi) / 2
        pool = [PConv(i, min(cap, s * qat(pts, (i + 0.5) / a.pool)), a.hit) for i in range(a.pool)]
        mode = f"CAPPED max={cap:,} scale={s:.3f}"
    else:
        pool = [PConv(i, qat(pts, (i + 0.5) / a.pool), a.hit) for i in range(a.pool)]
        mode = "stratified"
    ctxs = sorted(c.ctx for c in pool)
    print(f"LOCKED HIT={a.hit} | provider={a.provider} pool={a.pool} {mode} "
          f"arrival={a.arrival} timeout={a.timeout or 'off'} | {a.dur}s/level | salt={SALT}")
    print(f"pool ctx tokens: mean={sum(ctxs)//len(ctxs):,} p50={pct(ctxs,.5):.0f} "
          f"p95={pct(ctxs,.95):.0f} min={ctxs[0]:.0f} max={ctxs[-1]:.0f}")
    print("note: busy pool waits (does not drop); ok/wall includes drain — not DoneQPS")
    if not a.max_ctx:
        print("note: no --max-ctx; heavy draws up to 780k tokens")
    print()
    levels = [float(x) for x in a.levels.split(",")]
    timeout = None if a.timeout == 0 else a.timeout
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=2000)) as client:
        print("priming pool...", flush=True)
        await prime_pool(client, prov, pool, a.conc)
        print("primed.\n", flush=True)
        print(f"{'OfferedQPS':>10s} | {'ok/wall':>9s} | {'TTFTp50':>7s} | {'TTFTp95':>7s} | {'TTFTp99':>7s} | "
              f"{'E2Ep50':>7s} | {'E2Ep95':>7s} | {'E2Ep99':>7s} | {'tps50':>6s} | {'out50':>6s} | {'Hit':>6s} | {'Err':>6s}")
        print("-" * 125)
        for lv in levels:
            await run_level(client, prov, pool, lv, a.dur, rng, a.arrival, drain_grace=a.drain_grace)
    print("\n---DONE---", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
