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

def pending_tolerance(offered, max_pending_frac):
    """Pending requests allowed after grace; zero is strict."""
    return math.ceil(max_pending_frac * offered)

def level_is_clean(row, rate_key, max_pending_frac):
    """Whether a level is eligible for an automatic knee."""
    return (row["err"] == 0
            and row["generator_dropped"] == 0
            and row["pending_after_grace"] <= pending_tolerance(row["offered"], max_pending_frac)
            and row["backlog_growth"] <= max(2, math.ceil(0.05 * row["offered"]))
            and (row[rate_key] or 0) >= 0.85 * row["qps"])

class Conv:
    _n = 0
    def __init__(self, role, rng, total=None):
        self.conv = W.Conversation(role, rng, total=total)
        Conv._n += 1
        self.sid = f"conv-{Conv._n}"
        self.turns = 0

async def run_level(client, provider, role, mode, qps, dur, use_affinity, seed, arrival,
                    fixed_dist=False, pool=40, warmup=0.0, drain_grace=300.0,
                    slo_e2e_ms=None, max_arrival_lag_ms=50.0, max_arrival_lag_intervals=1.0):
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
    lag_threshold_ms = max(max_arrival_lag_ms, max_arrival_lag_intervals * 1000.0 / qps)
    st = {
        "ttft":[], "e2e":[], "hit":[], "ptok":[], "otok":[],
        "events":[], "all_events":[], "fire_t":[], "all_fire_t":[],
        "arrival_lag":[], "scheduled":0, "generator_dropped":0,
    }
    free = []
    async def one(cv, fire_rel):
        brk = W.should_break(role, cv.conv.rng, mode)
        messages, max_out = cv.conv.next_turn(break_cache=brk)
        r = await provider.run_turn(client, messages, max_out, session_id=cv.sid if use_affinity else None)
        cv.turns += 1
        done_rel = time.monotonic() - t0
        counted = fire_rel >= warmup
        event = {"fire_t": fire_rel, "done_t": done_rel, "ok": bool(r.get("ok")),
                 "code": r.get("code"), "e2e": r.get("e2e")}
        st["all_events"].append(event)
        if counted:
            st["events"].append(event)
            if r.get("ok"):
                st["ttft"].append(r.get("ttft")); st["e2e"].append(r.get("e2e"))
                pt,ct = r.get("prompt_tokens"), r.get("cached_tokens")
                if pt: st["ptok"].append(pt)
                if pt and ct is not None: st["hit"].append(ct/pt)
                ot = r.get("completion_tokens")
                if ot: st["otok"].append(ot)
        if cv.turns < MAX_TURNS: free.append(cv)   # recycle warm conversation
    if qps <= 0:
        raise ValueError(f"qps must be > 0, got {qps}")
    tasks=[]; t0=time.monotonic(); deadline=t0+dur; nxt=0.0
    while time.monotonic() < deadline:
        nxt += (1.0/qps) if arrival=="uniform" else (-math.log(1-arr_rng.random())/qps)  # even or Poisson
        target=t0+nxt; now=time.monotonic()
        if target >= deadline:
            break
        if target>now: await asyncio.sleep(target-now)
        now = time.monotonic()
        if now >= deadline:
            break
        scheduled_rel = nxt
        counted = scheduled_rel >= warmup
        lag = max(0.0, now - target)
        if counted:
            st["scheduled"] += 1
            st["arrival_lag"].append(lag)
        if lag * 1000.0 > lag_threshold_ms:
            if counted:
                st["generator_dropped"] += 1
            continue
        if free:
            cv = free.pop()
        else:
            tot = sizes[cold_n % len(sizes)] if sizes else None  # fixed schedule (deterministic) or random
            cv = Conv(role, random.Random(conv_rng.random()), total=tot); cold_n += 1
        fire_rel = time.monotonic()-t0
        st["all_fire_t"].append(fire_rel)
        if fire_rel >= warmup:
            st["fire_t"].append(fire_rel)
        tasks.append(asyncio.create_task(one(cv, fire_rel)))
    # Bounded drain is diagnostic; it never extends the throughput denominator.
    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=drain_grace)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    else:
        pending = set()
    meas_win = max(1e-9, dur-warmup)
    events = st["events"]
    offered = len(st["fire_t"])
    terminal_at_end = [e for e in events if e["done_t"] <= dur]
    ok_at_end = [e for e in terminal_at_end if e["ok"]]
    completion_qps = len(ok_at_end) / meas_win
    backlog_end = len(st["all_fire_t"]) - sum(1 for e in st["all_events"] if e["done_t"] <= dur)
    checkpoints = [warmup + meas_win/3, warmup + 2*meas_win/3, dur]
    def backlog_at(t):
        return (sum(1 for f in st["all_fire_t"] if f <= t)
                - sum(1 for e in st["all_events"] if e["done_t"] <= t))
    backlog_growth = backlog_at(checkpoints[-1]) - backlog_at(checkpoints[0])
    drain_seconds = max(0.0, time.monotonic() - (t0+dur))
    errors = {}
    for e in events:
        if not e["ok"]:
            errors[e["code"]] = errors.get(e["code"], 0) + 1
    err = len([e for e in events if not e["ok"]])
    errstr=",".join(f"{k}:{v}" for k,v in errors.items()) or "0"
    slo_s = slo_e2e_ms/1000.0 if slo_e2e_ms is not None else None
    goodput_qps = (sum(1 for e in events if e["ok"] and e["e2e"] is not None and e["e2e"] <= slo_s) / meas_win
                   if slo_s is not None else None)
    def ms(x): return f"{x*1000:.0f}" if x is not None else "-"
    def kt(x): return f"{x/1000:.0f}k" if x is not None else "-"
    out_p50 = pct(st["otok"], .5); out_max = max(st["otok"]) if st["otok"] else None
    in_p50 = pct(st["ptok"], .5)
    in_mean = (sum(st["ptok"])/len(st["ptok"])) if st["ptok"] else None
    slo_text = f"{goodput_qps:.3f}" if goodput_qps is not None else "-"
    lag_p95 = pct(st["arrival_lag"],.95); lag_max=max(st["arrival_lag"]) if st["arrival_lag"] else None
    print(f"{qps:>6.2f} | {st['scheduled']:>5d} | {offered:>5d} | {st['generator_dropped']:>5d} | "
          f"{ms(lag_p95):>7s} | {ms(lag_max):>7s} | {len(ok_at_end):>6d} | {completion_qps:>7.3f} | "
          f"{backlog_end:>4d} | {backlog_growth:>3d} | {drain_seconds:>5.1f} | {len(pending):>9d} | "
          f"{slo_text:>7s} | {kt(in_p50):>6s} | {kt(in_mean):>6s} | "
          f"{ms(pct(st['ttft'],.5)):>8s} | {ms(pct(st['ttft'],.95)):>8s} | {ms(pct(st['ttft'],.99)):>8s} | "
          f"{ms(pct(st['e2e'],.5)):>8s} | {ms(pct(st['e2e'],.95)):>9s} | {ms(pct(st['e2e'],.99)):>9s} | "
          f"{(out_p50 or 0):>7.0f} | {(pct(st['hit'],.5) or 0):>9.3f} | {errstr:>6s}", flush=True)
    return {"qps":qps,"completion_qps":completion_qps,"goodput_qps":goodput_qps,
            "backlog_end":backlog_end,"backlog_growth":backlog_growth,
            "drain_seconds":drain_seconds,"pending_after_grace":len(pending),
            "scheduled":st["scheduled"],"generator_dropped":st["generator_dropped"],
            "e2e_p95":pct(st["e2e"],.95),"err":err,"out_max":out_max,
            "in_mean":in_mean,"cache":pct(st["hit"],.5),"ttft_p50":pct(st["ttft"],.5),
            "offered":offered,"completed_by_end":len(ok_at_end)}

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
                    help="bounded seconds to wait after arrivals stop; reported separately from throughput")
    ap.add_argument("--slo-e2e-ms",type=float,default=None,
                    help="optional per-request E2E SLO; reports SLO goodput (requests/s)")
    ap.add_argument("--max-arrival-lag-ms",type=float,default=50.0,
                    help="absolute floor (ms) for arrival-lag budget; overdue arrivals are dropped")
    ap.add_argument("--max-arrival-lag-intervals",type=float,default=1.0,
                    help="arrival-lag budget in mean inter-arrivals; effective=max(floor, intervals*1000/qps)")
    ap.add_argument("--max-pending-frac",type=float,default=0.0,
                    help="strict default: any pending request halts; positive value is diagnostic only")
    ap.add_argument("--stop-on-explode",dest="stop_on_explode",action="store_true",
                    help="halt the ramp after the first level that truly congests (errors, or E2E p95 blowup)")
    ap.add_argument("--explode-e2e-ms",dest="explode_e2e_ms",type=float,default=45000,
                    help="E2E p95 (ms) at/above which a level counts as exploded (a lone heavy-tail straggler "
                         "does not move p95 across hundreds of requests, so this fires only on real congestion)")
    ap.add_argument("--explode-errs",dest="explode_errs",type=int,default=3,
                    help="error count at/above which a level counts as exploded")
    a=ap.parse_args()
    if a.slo_e2e_ms is not None and a.drain_grace < a.slo_e2e_ms / 1000.0:
        ap.error("--drain-grace must be at least --slo-e2e-ms / 1000")
    if not 0 <= a.warmup < a.dur:
        ap.error("--warmup must be greater than or equal to 0 and smaller than --dur")
    if a.max_arrival_lag_ms <= 0 or a.max_arrival_lag_intervals <= 0:
        ap.error("arrival-lag limits must be positive")
    if a.max_pending_frac < 0:
        ap.error("--max-pending-frac must be non-negative")
    if a.max_pending_frac > 0:
        print(f"NOTE: --max-pending-frac={a.max_pending_frac:g} is DIAGNOSTIC ONLY; fleet sizing is suppressed.\n")
    W.SIZE_SCALE=a.scale
    try:
        prov=build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(2)
    use_affinity = ensure_affinity(prov, a.affinity)
    levels=[float(x) for x in a.levels.split(",")]
    print(f"OPEN-LOOP ramp | provider={a.provider} role={a.role} mode={a.mode} "
          f"affinity={use_affinity} scale={a.scale} arrival={a.arrival} | {a.dur}s/level | 1 replica\n")
    print(f"{'Offer':>6s} | {'Sched':>5s} | {'N':>5s} | {'GDrop':>5s} | {'LagP95':>7s} | {'LagMax':>7s} | "
          f"{'Done@T':>6s} | {'CompQPS':>7s} | {'B@T':>4s} | {'dB':>3s} | {'Drain':>5s} | {'Pending@G':>9s} | "
          f"{'SLOQPS':>7s} | {'InP50':>6s} | {'InMean':>6s} | {'TTFTp50':>8s} | {'TTFTp95':>8s} | {'TTFTp99':>8s} | "
          f"{'E2Ep50':>8s} | {'E2Ep95':>9s} | {'E2Ep99':>9s} | {'OutP50':>7s} | {'CacheHit':>9s} | {'Errors':>6s}   (in=tok, lat=ms)")
    print("-"*215)
    rows=[]
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=2000)) as client:
        for lv in levels:
            row = await run_level(client, prov, a.role, a.mode, lv, a.dur, use_affinity, a.seed, a.arrival,
                                  fixed_dist=a.fixed_dist, pool=a.pool, warmup=a.warmup,
                                  drain_grace=a.drain_grace, slo_e2e_ms=a.slo_e2e_ms,
                                  max_arrival_lag_ms=a.max_arrival_lag_ms,
                                  max_arrival_lag_intervals=a.max_arrival_lag_intervals)
            rows.append(row)
            pend_tol = pending_tolerance(row["offered"], a.max_pending_frac)
            if row["pending_after_grace"] > pend_tol:
                print(f"\n*** UNRESOLVED at offered {lv} QPS: {row['pending_after_grace']} request(s) "
                      f"still pending after {a.drain_grace:g}s grace. Halting ramp. ***", flush=True)
                break
            if row["generator_dropped"] > 0:
                print(f"\n*** LOAD GENERATOR SATURATED at offered {lv} QPS: "
                      f"{row['generator_dropped']} overdue arrivals dropped. Halting ramp. ***", flush=True)
                break
            if a.stop_on_explode:
                e95 = row.get("e2e_p95")
                e95_ms = (e95 * 1000) if e95 is not None else 0.0
                if row["err"] >= a.explode_errs or e95_ms >= a.explode_e2e_ms:
                    why = (f"errors={row['err']} (>= {a.explode_errs})" if row["err"] >= a.explode_errs
                           else f"E2E p95={e95_ms:.0f}ms (>= {a.explode_e2e_ms:.0f}ms)")
                    print(f"\n*** EXPLODE at offered {lv} QPS: {why}. Halting ramp -- higher levels skipped. ***",
                          flush=True)
                    break
    rate_key = "goodput_qps" if a.slo_e2e_ms is not None else "completion_qps"
    good=[r for r in rows if level_is_clean(r, rate_key, a.max_pending_frac)]
    if good:
        knee=max(good,key=lambda r:r["qps"])
        s=knee[rate_key] or 0.0
        print(f"\nfixed-window knee: offered {knee['qps']} -> {rate_key} {s:.3f} QPS")
        if a.max_pending_frac > 0:
            print("  Diagnostic pending tolerance enabled: fleet sizing suppressed.")
        else:
            GPUS_PER_REPLICA = 4
            for tgt in [1, 5, 10, 20]:
                reps=math.ceil(tgt/s) if s>0 else 0
                print(f"  {tgt:>6.1f} QPS -> {reps:>4} replicas / {reps*GPUS_PER_REPLICA:>5} GPUs (linear; add imbalance headroom)")
    print("\n---DONE---")

if __name__ == "__main__":
    asyncio.run(main())
