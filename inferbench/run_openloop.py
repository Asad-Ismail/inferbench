"""Open-loop QPS ramp.

Poisson (or uniform) arrivals at fixed offered-QPS targets, multi-turn conversations with
optional session affinity, measuring sustained QPS / TTFT / E2E / cache-hit / errors per level.
Reuses workload.py + providers.py (streams usage incl. cached_tokens).

Usage:
  python run_openloop.py --provider myapi --role heavy --mode real --affinity \
    --levels 0.5,1,2,3,4,6 --dur 40 --scale 1.0
"""
import argparse, asyncio, time, random, math, sys
import httpx

try:
    from inferbench import workload as W
    from inferbench.providers import build_provider, ensure_affinity
except ImportError:
    import workload as W
    from providers import build_provider, ensure_affinity

MAX_TURNS = 8

def pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    if not xs: return None
    k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

class Conv:
    _n = 0
    def __init__(self, role, rng, total=None):
        self.conv = W.Conversation(role, rng, total=total)
        Conv._n += 1
        self.sid = f"conv-{Conv._n}"
        self.turns = 0

async def run_level(client, provider, role, mode, qps, dur, use_affinity, seed, arrival,
                    fixed_dist=False, pool=40, warmup=0.0, drain_grace=300.0):
    # Two independent, per-level RNG streams. A single shared stream fed BOTH the Poisson
    # arrivals AND the conversation seeds, but the conversation draw only fires when `free`
    # empties (latency-dependent) -> a faster endpoint consumed a different number of draws and
    # got a different arrival schedule AND different request sizes, so two endpoints were never
    # given the same test. Re-seeding per level also makes the Nth cold conversation identical
    # regardless of level order (heavy conversations no longer pile into whichever level runs
    # first -> fixes the spurious "low QPS is slow" artifact).
    arr_rng  = random.Random(f"arr:{seed}:{qps}")
    conv_rng = random.Random(f"conv:{seed}:{qps}")
    # --fixed-dist: every level walks the SAME deterministic stratified size schedule in order,
    # so all levels see the identical input distribution (no per-level "surprise" heavy sample).
    sizes = W.stratified_totals(role, pool) if fixed_dist else None
    cold_n = 0
    st = {"ok":0,"errs":{},"ttft":[],"e2e":[],"hit":[],"ptok":[],"otok":[],"done_t":[],"cancelled":0}
    free = []
    async def one(cv, fire_rel):
        brk = W.should_break(role, cv.conv.rng, mode)
        messages, max_out = cv.conv.next_turn(break_cache=brk)
        r = await provider.run_turn(client, messages, max_out, session_id=cv.sid if use_affinity else None)
        cv.turns += 1
        counted = fire_rel >= warmup   # drop the warm-up cohort from ALL metrics (steady-state only)
        if r.get("ok"):
            if counted:
                st["ok"]+=1; st["ttft"].append(r.get("ttft")); st["e2e"].append(r.get("e2e"))
                st["done_t"].append(time.monotonic()-t0)
                pt,ct = r.get("prompt_tokens"), r.get("cached_tokens")
                if pt: st["ptok"].append(pt)
                if pt and ct is not None: st["hit"].append(ct/pt)
                ot = r.get("completion_tokens")
                if ot: st["otok"].append(ot)   # watch OutMax: 65536 here == output cap not honored
        elif counted:
            st["errs"][r.get("code")] = st["errs"].get(r.get("code"),0)+1
        if cv.turns < MAX_TURNS: free.append(cv)   # recycle warm conversation
    if qps <= 0:
        raise ValueError(f"qps must be > 0, got {qps}")
    tasks=[]; t0=time.monotonic(); deadline=t0+dur; nxt=0.0
    while time.monotonic() < deadline:
        nxt += (1.0/qps) if arrival=="uniform" else (-math.log(1-arr_rng.random())/qps)  # even or Poisson
        target=t0+nxt; now=time.monotonic()
        if target>now: await asyncio.sleep(target-now)
        if free:
            cv = free.pop()
        else:
            tot = sizes[cold_n % len(sizes)] if sizes else None  # fixed schedule (deterministic) or random
            cv = Conv(role, random.Random(conv_rng.random()), total=tot); cold_n += 1
        tasks.append(asyncio.create_task(one(cv, time.monotonic()-t0)))
    # Bounded drain: at overload, waiting for the full backlog can take many minutes.
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=drain_grace)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            st["cancelled"] = len(pending)
    wall=time.monotonic()-t0
    # sustained = counted completions / the post-warm-up measurement window (incl. its drain).
    meas_win = (max(st["done_t"]) - warmup) if st["done_t"] else max(1e-9, wall - warmup)
    sustained = st["ok"]/meas_win if meas_win > 0 else 0.0
    err=sum(st["errs"].values())
    errstr=",".join(f"{k}:{v}" for k,v in st["errs"].items()) or "0"
    def ms(x): return f"{x*1000:.0f}" if x is not None else "-"
    def kt(x): return f"{x/1000:.0f}k" if x is not None else "-"
    out_p50 = pct(st["otok"], .5); out_max = max(st["otok"]) if st["otok"] else None
    in_p50 = pct(st["ptok"], .5)
    in_mean = (sum(st["ptok"])/len(st["ptok"])) if st["ptok"] else None
    blk = f" backlog_cancelled={st['cancelled']}" if st["cancelled"] else ""
    print(f"{qps:>10.2f} | {sustained:>9.3f} | {kt(in_p50):>6s} | {kt(in_mean):>6s} | "
          f"{ms(pct(st['ttft'],.5)):>8s} | {ms(pct(st['ttft'],.95)):>8s} | {ms(pct(st['ttft'],.99)):>8s} | "
          f"{ms(pct(st['e2e'],.5)):>8s} | {ms(pct(st['e2e'],.95)):>9s} | {ms(pct(st['e2e'],.99)):>9s} | "
          f"{(out_p50 or 0):>7.0f} | {(pct(st['hit'],.5) or 0):>9.3f} | {st['ok']:>4d} | {errstr:>6s}{blk}", flush=True)
    return {"qps":qps,"sustained":sustained,"e2e_p95":pct(st['e2e'],.95),"err":err,"out_max":out_max,
            "in_mean":in_mean,"cache":pct(st['hit'],.5),"ttft_p50":pct(st['ttft'],.5),"n":st['ok'],
            "cancelled":st["cancelled"]}

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--provider",required=True)
    ap.add_argument("--role",default="heavy",choices=sorted(W.ROLE_PROFILES))
    ap.add_argument("--mode",default="real",choices=["fixed","broken","real"])
    ap.add_argument("--levels",default="0.1,0.25,0.5,0.75,1,1.5,2,3,4,5")
    ap.add_argument("--dur",type=int,default=90); ap.add_argument("--scale",type=float,default=1.0)
    ap.add_argument("--affinity",action="store_true"); ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--arrival",default="uniform",choices=["uniform","poisson"])
    ap.add_argument("--fixed-dist",dest="fixed_dist",action="store_true",
                    help="every level walks the same deterministic stratified size schedule (fair level comparison)")
    ap.add_argument("--pool",type=int,default=40,help="size of the fixed stratified schedule")
    ap.add_argument("--warmup",type=float,default=0.0,
                    help="seconds of warm-up to discard from all metrics (measure steady state only)")
    ap.add_argument("--drain-grace",dest="drain_grace",type=float,default=300.0,
                    help="seconds to wait for in-flight requests after the arrival window (then cancel)")
    ap.add_argument("--stop-on-explode",dest="stop_on_explode",action="store_true",
                    help="halt the ramp after the first level that truly congests (errors, or E2E p95 blowup)")
    ap.add_argument("--explode-e2e-ms",dest="explode_e2e_ms",type=float,default=45000,
                    help="E2E p95 (ms) at/above which a level counts as exploded (a lone heavy-tail straggler "
                         "does not move p95 across hundreds of requests, so this fires only on real congestion)")
    ap.add_argument("--explode-errs",dest="explode_errs",type=int,default=3,
                    help="error count at/above which a level counts as exploded")
    a=ap.parse_args()
    W.SIZE_SCALE=a.scale
    try:
        prov=build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(2)
    use_affinity = ensure_affinity(prov, a.affinity)
    levels=[float(x) for x in a.levels.split(",")]
    print(f"OPEN-LOOP ramp | provider={a.provider} role={a.role} mode={a.mode} "
          f"affinity={use_affinity} scale={a.scale} arrival={a.arrival} | {a.dur}s/level | 1 replica\n")
    print(f"{'QPS Target':>10s} | {'Sustained':>9s} | {'InP50':>6s} | {'InMean':>6s} | {'TTFTp50':>8s} | {'TTFTp95':>8s} | {'TTFTp99':>8s} | "
          f"{'E2Ep50':>8s} | {'E2Ep95':>9s} | {'E2Ep99':>9s} | {'OutP50':>7s} | {'CacheHit':>9s} | {'N':>4s} | {'Errors':>6s}   (in=tok, lat=ms)")
    print("-"*145)
    rows=[]
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=2000)) as client:
        for lv in levels:
            row = await run_level(client, prov, a.role, a.mode, lv, a.dur, use_affinity, a.seed, a.arrival,
                                  fixed_dist=a.fixed_dist, pool=a.pool, warmup=a.warmup,
                                  drain_grace=a.drain_grace)
            rows.append(row)
            if a.stop_on_explode:
                e95 = row.get("e2e_p95")
                e95_ms = (e95 * 1000) if e95 is not None else 0.0
                if row["err"] >= a.explode_errs or e95_ms >= a.explode_e2e_ms:
                    why = (f"errors={row['err']} (>= {a.explode_errs})" if row["err"] >= a.explode_errs
                           else f"E2E p95={e95_ms:.0f}ms (>= {a.explode_e2e_ms:.0f}ms)")
                    print(f"\n*** EXPLODE at offered {lv} QPS: {why}. Halting ramp -- higher levels skipped. ***",
                          flush=True)
                    break
    # goodput knee + sizing. Heavy-tail stragglers can be slow even at low load, so we don't
    # gate on E2E p95. Knee = highest offered QPS the single replica still KEEPS UP with
    # (sustained >= 0.85*offered) and returns 0 errors.
    good=[r for r in rows if r["err"]==0 and r["sustained"]>=0.85*r["qps"]]
    if good:
        knee=max(good,key=lambda r:r["qps"])
        s=knee["sustained"]
        print(f"\ngoodput knee (keeps up: sustained>=0.85*offered, 0 err): offered {knee['qps']} -> sustained {s:.3f} QPS/replica")
        GPUS_PER_REPLICA = 4   # GPUs per model replica — set to your deployment
        for tgt in [1, 5, 10, 20]:
            reps=math.ceil(tgt/s) if s>0 else 0
            print(f"  {tgt:>6.1f} QPS -> {reps:>4} replicas / {reps*GPUS_PER_REPLICA:>5} GPUs (linear; add imbalance headroom)")
    print("\n---DONE---")

if __name__ == "__main__":
    asyncio.run(main())
