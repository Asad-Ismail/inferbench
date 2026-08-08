"""Open-loop QPS knee at a CONTROLLED, LOCKED cache-hit.

Unlike run_openloop.py (whose blended cache-hit drifts with load because the warm/cold turn
ratio and request-size mix shift), this pins every request to the SAME target hit:

  * A fixed pool of conversations is PRE-PRIMED before the ramp: each conv has a stable prefix
    of round(hit * ctx) tokens that we warm into the endpoint cache (via affinity) up front.
  * During the ramp, every request = [stable prefix (cached -> hit)] + [fresh novel (1-hit)].
    Fresh text is per-run-salted + per-turn unique, so it never caches -> the measured hit is
    exactly `hit` for every request, at every offered QPS. No drift.

This makes cache-hit an independent variable: run at your target hit (e.g. 0.59) for the
real knee, or at other values to see how capacity scales with cache efficiency.

Usage:
  python3 knee_at_hit.py --provider myapi --hit 0.59 --levels 0.05,0.1,0.15,0.2,0.25,0.3 --dur 90
  python3 knee_at_hit.py --provider myapi --hit 0.90 --levels 0.05,0.1,0.15,0.2,0.25,0.3 --dur 90
"""
import argparse, asyncio, time, random, math, os, statistics, sys
import httpx

try:
    from inferbench import workload as W
    from inferbench.providers import build_provider, ensure_affinity
except ImportError:
    import workload as W
    from providers import build_provider, ensure_affinity

RUN_SALT = os.environ.get("SALT") or os.urandom(6).hex()


def block(approx_tokens, seed):
    rng = random.Random(str((RUN_SALT, seed)))
    return " ".join(rng.choice(W._VOCAB) for _ in range(max(1, int(approx_tokens))))


def quantile_at(points, u):
    """Evaluate the empirical (quantile,value) function at u in [0,1], log-interpolated.
    Used for STRATIFIED pool sampling: a small pool whose mean+tail match the full
    distribution while still fitting in the KV cache."""
    for i in range(1, len(points)):
        q0, v0 = points[i - 1]; q1, v1 = points[i]
        if u <= q1:
            f = (u - q0) / (q1 - q0) if q1 > q0 else 0.0
            lo, hi = math.log(max(v0, 1)), math.log(max(v1, 1))
            return math.exp(lo + f * (hi - lo))
    return float(points[-1][1])


def pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    if not xs: return None
    k = (len(xs) - 1) * p; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class PConv:
    """A pre-primed conversation with a fixed cached prefix sized to hit*ctx."""
    def __init__(self, i, ctx, hit):
        self.i = i
        self.ctx = int(ctx)
        self.cached = max(1, int(round(hit * self.ctx)))
        self.fresh = max(1, self.ctx - self.cached)
        self.stable = "[STABLE] " + block(self.cached, ("stable", i))
        self.sid = f"lock-{i}"
        self.k = 0
        self.out_rng = random.Random(str(("out", i)))

    def prime_messages(self):
        # warm the stable prefix into cache; fresh part is tiny/cheap here.
        return [{"role": "system", "content": self.stable},
                {"role": "user", "content": "[FRESH] " + block(min(self.fresh, 400), ("prime", self.i))}]

    def turn_messages(self):
        self.k += 1
        # full fresh block, novel every turn -> guaranteed miss -> measured hit == target.
        return [{"role": "system", "content": self.stable},
                {"role": "user", "content": "[FRESH] " + block(self.fresh, ("f", self.i, self.k))}]

    def max_out(self):
        return int(W.sample_pct(W.ROLE_PROFILES["heavy"]["output"], self.out_rng))


async def prime_pool(client, prov, pool, conc):
    sem = asyncio.Semaphore(conc)
    async def one(c):
        async with sem:
            await prov.run_turn(client, c.prime_messages(), 8, session_id=c.sid)
    await asyncio.gather(*[one(c) for c in pool])


async def run_level(client, prov, pool, qps, dur, rng, arrival, drain_grace=300.0):
    st = {"ok": 0, "errs": {}, "ttft": [], "e2e": [], "hit": [], "ptok": [], "samples": [], "tps": [], "otok": []}
    t0 = time.monotonic()
    free = list(pool)  # one in-flight request per primed conv (avoids overlapping same sid)
    async def one(c):
        try:
            r = await prov.run_turn(client, c.turn_messages(), c.max_out(), session_id=c.sid)
            if r.get("ok"):
                st["ok"] += 1; st["ttft"].append(r.get("ttft")); st["e2e"].append(r.get("e2e"))
                st["samples"].append((time.monotonic() - t0, r.get("e2e")))  # completion time, e2e
                pt, ct = r.get("prompt_tokens"), r.get("cached_tokens")
                if pt: st["ptok"].append(pt)
                if pt and ct is not None: st["hit"].append(ct / pt)
                # PER-REQUEST decode rate: output / (E2E - TTFT). The correct way (not percentile math).
                ttft, e2e, ot = r.get("ttft"), r.get("e2e"), r.get("completion_tokens")
                if ot: st["otok"].append(ot)
                if ttft is not None and e2e is not None and ot and e2e > ttft and ot > 1:
                    st["tps"].append(ot / (e2e - ttft))
            else:
                st["errs"][r.get("code")] = st["errs"].get(r.get("code"), 0) + 1
        finally:
            free.append(c)
    if qps <= 0:
        raise ValueError(f"qps must be > 0, got {qps}")
    tasks = []; deadline = t0 + dur; nxt = 0.0
    while time.monotonic() < deadline:
        nxt += (1.0 / qps) if arrival == "uniform" else (-math.log(1 - rng.random()) / qps)
        target = t0 + nxt; now = time.monotonic()
        if target > now: await asyncio.sleep(target - now)
        while not free and time.monotonic() < deadline:
            await asyncio.sleep(0.001)  # all pool convs busy; wait rather than overlap sids
        if not free:
            break
        c = free.pop(rng.randrange(len(free)))
        tasks.append(asyncio.create_task(one(c)))
    # bounded drain: at overloaded QPS the backlog would take many minutes to drain after the
    # arrival window, so let in-flight finish for up to `grace`, then cancel the rest.
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
    errstr = ",".join(f"{k}:{v}" for k, v in st["errs"].items()) or "0"
    def ms(x): return f"{x*1000:.0f}" if x is not None else "-"
    backlog = st.get("cancelled", 0)
    blk = f" backlog_cancelled={backlog}" if backlog else ""
    def n0(x): return f"{x:.0f}" if x is not None else "-"
    print(f"{qps:>10.2f} | {sustained:>9.3f} | {ms(pct(st['ttft'],.5)):>7s} | {ms(pct(st['ttft'],.95)):>7s} | "
          f"{ms(pct(st['ttft'],.99)):>7s} | {ms(pct(st['e2e'],.5)):>7s} | {ms(pct(st['e2e'],.95)):>7s} | "
          f"{ms(pct(st['e2e'],.99)):>7s} | {n0(pct(st['tps'],.5)):>6s} | {n0(pct(st['otok'],.5)):>6s} | "
          f"{(pct(st['hit'],.5) or 0):>6.3f} | {errstr:>6s}{blk}", flush=True)
    # DRIFT / steady-state check: split the level into thirds by completion time; if E2E p95
    # climbs across thirds, the queue is GROWING -> not truly sustained at this QPS.
    drift = None
    if wall >= 300 and len(st["samples"]) >= 12:
        third = wall / 3.0
        buckets = [[e for (tc, e) in st["samples"] if lo <= tc < lo + third] for lo in (0, third, 2 * third)]
        p95s = [pct(b, .95) for b in buckets if b]
        cnts = [len(b) for b in buckets]
        if len(p95s) == 3:
            verdict = "STEADY" if p95s[2] <= 1.5 * p95s[0] else "GROWING (queue building -> NOT sustained)"
            print(f"           drift E2E p95 by third (n={cnts}): "
                  f"{p95s[0]:.1f}s -> {p95s[1]:.1f}s -> {p95s[2]:.1f}s  => {verdict}", flush=True)
            drift = {"p95_thirds": p95s, "verdict": verdict}
    return {"qps": qps, "sustained": sustained, "e2e_p95": pct(st['e2e'], .95),
            "hit": pct(st['hit'], .5), "err": err, "n": st["ok"], "drift": drift}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--hit", type=float, required=True, help="locked cache-hit for EVERY request")
    ap.add_argument("--levels", default="0.05,0.1,0.15,0.2,0.25,0.3")
    ap.add_argument("--dur", type=int, default=90)
    ap.add_argument("--pool", type=int, default=30, help="pre-primed conversation pool size (stratified)")
    ap.add_argument("--conc", type=int, default=6, help="priming concurrency")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arrival", default="uniform", choices=["uniform", "poisson"])
    ap.add_argument("--uniform-ctx", action="store_true",
                    help="every request = the distribution MEAN ctx (tail removed) -> isolates the "
                         "throughput ceiling from heavy-tail head-of-line blocking")
    ap.add_argument("--max-cap", type=int, default=0,
                    help="clamp every request's ctx to this max (simulate context-capping/condensation)")
    ap.add_argument("--target-mean", type=int, default=0,
                    help="with --max-cap, scale the sub-cap distribution so the mean hits this "
                         "(hold avg load constant while removing the >cap tail)")
    ap.add_argument("--drain-grace", type=float, default=300.0,
                    help="seconds to wait for in-flight requests after the arrival window (then cancel)")
    a = ap.parse_args()
    try:
        prov = build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(2)
    ensure_affinity(prov, True)  # locked-hit needs sticky routing for the primed prefixes
    rng = random.Random(a.seed)
    pts = W.ROLE_PROFILES["heavy"]["total"]
    if a.uniform_ctx:
        # all convs = distribution mean (compute via a fine stratified quadrature): AVERAGE-input case.
        mean_ctx = int(sum(quantile_at(pts, (i + 0.5) / 400) for i in range(400)) / 400)
        pool = [PConv(i, mean_ctx, a.hit) for i in range(a.pool)]
        mode = f"UNIFORM-ctx={mean_ctx:,} (avg input, no tail)"
    elif a.max_cap:
        cap = a.max_cap
        if a.target_mean:  # scale sub-cap dist so mean == target (hold avg load, drop >cap tail)
            lo, hi = 0.1, 20.0
            for _ in range(60):
                mid = (lo + hi) / 2
                m = sum(min(cap, mid * quantile_at(pts, (i + 0.5) / 2000)) for i in range(2000)) / 2000
                if m < a.target_mean: lo = mid
                else: hi = mid
            s = (lo + hi) / 2
        else:
            s = 1.0
        pool = [PConv(i, min(cap, s * quantile_at(pts, (i + 0.5) / a.pool)), a.hit) for i in range(a.pool)]
        mode = f"CAPPED max={cap:,} scale={s:.3f} (tail clamped; mean held)"
    else:
        # STRATIFIED pool: ctx at evenly-spaced quantiles -> small pool matches the full
        # role input distribution's mean+tail while fitting the KV cache (no eviction).
        pool = [PConv(i, quantile_at(pts, (i + 0.5) / a.pool), a.hit) for i in range(a.pool)]
        mode = "stratified (full dist, mean+tail)"
    ctxs = sorted(c.ctx for c in pool)
    print(f"KNEE @ LOCKED HIT={a.hit} | provider={a.provider} pool={a.pool} {mode} "
          f"arrival={a.arrival} | {a.dur}s/level | 1 replica | salt={RUN_SALT}")
    print(f"pool ctx tokens: mean={sum(ctxs)//len(ctxs):,} p50={pct(ctxs,.5):.0f} "
          f"p95={pct(ctxs,.95):.0f} min={ctxs[0]:.0f} max={ctxs[-1]:.0f}\n")
    levels = [float(x) for x in a.levels.split(",")]
    async with httpx.AsyncClient(timeout=900, limits=httpx.Limits(max_connections=2000)) as client:
        print("priming pool (warming stable prefixes into cache)...", flush=True)
        await prime_pool(client, prov, pool, a.conc)
        print("primed.\n", flush=True)
        print(f"{'QPS Target':>10s} | {'Sustained':>9s} | {'TTFTp50':>7s} | {'TTFTp95':>7s} | {'TTFTp99':>7s} | "
              f"{'E2Ep50':>7s} | {'E2Ep95':>7s} | {'E2Ep99':>7s} | {'tps50':>6s} | {'out50':>6s} | {'Hit':>6s} | {'Err':>6s}   (ms; tps=tok/s)")
        print("-" * 125)
        rows = []
        for lv in levels:
            rows.append(await run_level(client, prov, pool, lv, a.dur, rng, a.arrival,
                                        drain_grace=a.drain_grace))
    good = [r for r in rows if r["err"] == 0 and r["sustained"] >= 0.85 * r["qps"]]
    if good:
        knee = max(good, key=lambda r: r["qps"]); s = knee["sustained"]
        mh = statistics.median([r["hit"] for r in rows if r["hit"] is not None])
        print(f"\nknee (keeps up, 0 err): offered {knee['qps']} -> {s:.3f} QPS/replica "
              f"| measured hit held at ~{mh:.3f} (target {a.hit})")
        GPUS_PER_REPLICA = 4   # GPUs per model replica — set to your deployment
        for tgt in [1, 5, 10, 20]:
            reps = math.ceil(tgt / s) if s > 0 else 0
            print(f"  {tgt:>5} QPS -> {reps:>4} replicas / {reps*GPUS_PER_REPLICA:>5} GPUs (linear; add imbalance headroom)")
    else:
        print("\nno level kept up at 0.85*offered with 0 err — knee below smallest offered level.")
    print("\n---DONE---", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
