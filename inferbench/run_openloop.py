"""Open-loop QPS ramp.

Poisson (or uniform) arrivals at fixed offered-QPS targets, multi-turn conversations with
optional session affinity, measuring fixed-window completion QPS, bounded drain,
optional SLO goodput, TTFT/E2E, cache-hit, and errors per level.
Reuses workload.py + providers.py (streams usage incl. cached_tokens).

Usage:
  python run_openloop.py --provider myapi --role heavy --mode real --affinity \
    --levels 0.5,1,2,3,4,6 --dur 40 --scale 1.0
"""
import argparse, asyncio, csv, json, time, random, math, sys
from pathlib import Path
import httpx

try:
    from inferbench import workload as W
    from inferbench.providers import build_provider, ensure_affinity
except ImportError:
    import workload as W
    from providers import build_provider, ensure_affinity

# Effective sample size is CONVERSATIONS, not requests: input size is constant within a
# conversation, so its 2nd..Nth turns are near-duplicates that cost tokens and buy no precision.
# A new conversation is only created when `free` empties, so conversations accrue at roughly
# arrival_rate / MAX_TURNS. Lowering this trades duplicate turns for independent size draws at
# the SAME request count and token spend. Measured on a 900s run: 0.508 conversations per request
# at 2 vs 0.184 at 8, and the 95% CI on TTFT p50 narrows from 8.66s (7 conversations) to 1.34s
# (129). Raising it back changes the cache profile too -- turn 1 is always cold -- so runs at
# different --max-turns are not comparable on cache_hit or absolute TTFT, only size-controlled.
MAX_TURNS = 2

def pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    if not xs: return None
    k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

def pending_tolerance(offered, max_pending_frac):
    """Pending requests allowed after grace; zero is strict."""
    return math.ceil(max_pending_frac * offered)

PLAN_FIELDS = ["offered_qps", "kind", "idx", "value"]

def build_plan(role, qps, dur, seed, arrival, fixed_dist=False, pool=40):
    """Arrival times and cold-conversation sizes a level is ABOUT to produce.

    Uses throwaway Random instances seeded identically to the live streams, so this predicts
    the schedule rather than replacing it -- the live arr_rng/conv_rng are untouched and the
    run is byte-identical with or without a plan.

    Written before the level runs, it separates "the endpoint was slow" from "the harness
    offered a different workload than you asked for". Arrivals are fully determined by
    (seed, qps): a mismatch means the generator fell behind, not that the RNG drifted.
    Conversation sizes are determined only up to ORDER -- the Nth cold conversation always has
    the same size, but HOW MANY a level creates depends on latency, because a slow endpoint
    starves the recycle pool and starts more cold (more expensive) conversations. Comparing
    the cold count reached against the plan is the direct measure of that drift.
    """
    plan_arr  = random.Random(f"arr:{seed}:{qps}")
    plan_conv = random.Random(f"conv:{seed}:{qps}")
    arrivals, t = [], 0.0
    while True:
        t += (1.0/qps) if arrival == "uniform" else (-math.log(1-plan_arr.random())/qps)
        if t >= dur: break
        arrivals.append(t)
    # worst case every arrival is cold, so plan that many sizes (+ headroom)
    n = len(arrivals) + 8
    if fixed_dist:
        sizes = W.stratified_totals(role, pool)
        sizes = [sizes[i % len(sizes)] for i in range(n)]
    else:
        prof = W.ROLE_PROFILES[role]["total"]
        sizes = [W.sample_pct(prof, random.Random(plan_conv.random())) for _ in range(n)]
    return arrivals, sizes

def write_rows(path, rows, append=False):
    """Write one structured result record per offered QPS level."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".json", ".jsonl"}:
        raise ValueError("--out must end in .csv, .json, or .jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        fields = list(dict.fromkeys(key for row in rows for key in row))
        exists = append and path.exists() and path.stat().st_size > 0
        with path.open("a" if append else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                                 for k, v in row.items()})
    elif suffix == ".jsonl":
        with path.open("a" if append else "w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    else:
        existing = json.loads(path.read_text()) if append and path.exists() else []
        path.write_text(json.dumps([*existing, *rows], indent=2, sort_keys=True) + "\n")

# Per-request columns. Units follow the structured-output convention: _s for durations,
# _ms for arrival lag, _tokens for counts. Every row carries request_id and conv_id so a
# flagged request traces back to the endpoint's own logs and to the conversation that made it.
REQUEST_FIELDS = ["offered_qps", "conv_id", "turn", "cold", "counted",
                  "fire_t_s", "done_t_s", "arrival_lag_ms", "inflight",
                  "ok", "code", "http_status", "retry_after", "request_id",
                  "ttft_s", "e2e_s",
                  "prompt_tokens", "cached_tokens", "cached_ratio", "completion_tokens",
                  "reasoning_tokens", "finish_reason", "error_body"]

def open_request_log(path):
    """Per-request CSV, flushed per row so an interrupted level keeps everything finished."""
    if not path: return None
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=REQUEST_FIELDS, extrasaction="ignore")
    w.writeheader(); f.flush()
    return (f, w)

def write_request(handle, row):
    if not handle: return
    f, w = handle
    w.writerow(row); f.flush()

class Conv:
    _n = 0
    def __init__(self, role, rng, total=None):
        self.conv = W.Conversation(role, rng, total=total)
        Conv._n += 1
        self.sid = f"conv-{Conv._n}"
        self.turns = 0

async def run_level(client, provider, role, mode, qps, dur, use_affinity, seed, arrival,
                    fixed_dist=False, pool=40, warmup=0.0, drain_grace=300.0,
                    slo_e2e_ms=None, max_arrival_lag_ms=50.0, max_arrival_lag_intervals=1.0,
                    max_turns=None, request_log=None, plan_log=None):
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
    turn_cap = MAX_TURNS if max_turns is None else max_turns
    cold_n = 0
    lag_threshold_ms = max(max_arrival_lag_ms, max_arrival_lag_intervals * 1000.0 / qps)
    st = {
        "ttft":[], "e2e":[], "hit":[], "ptok":[], "otok":[],
        "events":[], "all_events":[], "fire_t":[], "all_fire_t":[],
        "arrival_lag":[], "scheduled":0, "generator_dropped":0,
        "inflight":0, "inflight_peak":0, "cold_counted":0, "stalled":0,
    }
    if plan_log:
        arrivals, plan_sizes = build_plan(role, qps, dur, seed, arrival, fixed_dist, pool)
        write_rows(plan_log, [{"offered_qps": qps, "kind": "arrival", "idx": i, "value": v}
                              for i, v in enumerate(arrivals)]
                           + [{"offered_qps": qps, "kind": "conv_total", "idx": i, "value": v}
                              for i, v in enumerate(plan_sizes)], append=True)
    free = []
    async def one(cv, fire_rel, cold, lag_ms):
        brk = W.should_break(role, cv.conv.rng, mode)
        messages, max_out = cv.conv.next_turn(break_cache=brk)
        st["inflight"] += 1
        st["inflight_peak"] = max(st["inflight_peak"], st["inflight"])
        inflight_at_dispatch = st["inflight"]
        try:
            r = await provider.run_turn(client, messages, max_out, session_id=cv.sid if use_affinity else None)
        finally:
            st["inflight"] -= 1
        cv.turns += 1
        done_rel = time.monotonic() - t0
        counted = fire_rel >= warmup
        event = {"fire_t": fire_rel, "done_t": done_rel, "ok": bool(r.get("ok")),
                 "code": r.get("code"), "e2e": r.get("e2e")}
        st["all_events"].append(event)
        pt, ct = r.get("prompt_tokens"), r.get("cached_tokens")
        if counted:
            st["events"].append(event)
            if cold: st["cold_counted"] += 1
            if r.get("ok"):
                # HTTP 200 with no usage is a stalled stream, not a success: it reports no
                # tokens, but its E2E still lands in the tail. Counted separately so it cannot
                # masquerade as a fast empty response.
                if not pt: st["stalled"] += 1
                st["ttft"].append(r.get("ttft")); st["e2e"].append(r.get("e2e"))
                if pt: st["ptok"].append(pt)
                if pt and ct is not None: st["hit"].append(ct/pt)
                ot = r.get("completion_tokens")
                if ot: st["otok"].append(ot)
        write_request(request_log, {
            "offered_qps": qps, "conv_id": cv.sid, "turn": cv.turns, "cold": int(cold),
            "counted": int(counted), "fire_t_s": round(fire_rel, 4), "done_t_s": round(done_rel, 4),
            "arrival_lag_ms": round(lag_ms, 3), "inflight": inflight_at_dispatch,
            "ok": int(bool(r.get("ok"))), "code": r.get("code"),
            "http_status": r.get("http_status"), "retry_after": r.get("retry_after"),
            "request_id": r.get("request_id"),
            "ttft_s": r.get("ttft"), "e2e_s": r.get("e2e"),
            "prompt_tokens": pt, "cached_tokens": ct,
            "cached_ratio": round(ct/pt, 4) if pt and ct is not None else None,
            "completion_tokens": r.get("completion_tokens"),
            "reasoning_tokens": r.get("reasoning_tokens"),
            "finish_reason": r.get("finish_reason"), "error_body": r.get("error_body"),
        })
        if cv.turns < turn_cap: free.append(cv)   # recycle warm conversation
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
            cv = free.pop(); cold = False
        else:
            tot = sizes[cold_n % len(sizes)] if sizes else None  # fixed schedule (deterministic) or random
            cv = Conv(role, random.Random(conv_rng.random()), total=tot); cold_n += 1; cold = True
        fire_rel = time.monotonic()-t0
        st["all_fire_t"].append(fire_rel)
        if fire_rel >= warmup:
            st["fire_t"].append(fire_rel)
        tasks.append(asyncio.create_task(one(cv, fire_rel, cold, lag * 1000.0)))
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
    return {"offered_qps":qps,"completion_qps":completion_qps,"goodput_qps":goodput_qps,
            "backlog_end":backlog_end,"backlog_growth":backlog_growth,
            "drain_s":drain_seconds,"pending_after_grace":len(pending),
            "scheduled":st["scheduled"],"generator_dropped":st["generator_dropped"],
            "arrival_lag_p95_ms":(lag_p95 * 1000.0) if lag_p95 is not None else None,
            "arrival_lag_max_ms":(lag_max * 1000.0) if lag_max is not None else None,
            "ttft_p50_s":pct(st["ttft"],.5),"ttft_p95_s":pct(st["ttft"],.95),"ttft_p99_s":pct(st["ttft"],.99),
            "e2e_p50_s":pct(st["e2e"],.5),"e2e_p95_s":pct(st["e2e"],.95),"e2e_p99_s":pct(st["e2e"],.99),
            "error_count":err,"error_counts":errors,
            "output_tokens_p50":out_p50,"output_tokens_max":out_max,
            "input_tokens_p50":in_p50,"input_tokens_mean":in_mean,"cache_hit_p50":pct(st["hit"],.5),
            # The cache-hit MEDIAN is pinned at the design target until more than half of
            # requests are cold, then flips to the cold mode: it read 0.5885/0.5891/0.5892 across
            # a 4x rate change while the mean moved 0.375/0.420/0.453. Report the mean.
            "cache_hit_mean":(sum(st["hit"])/len(st["hit"])) if st["hit"] else None,
            "cold_conversations":cold_n,"cold_pct":(st["cold_counted"]/len(events)) if events else None,
            "inflight_peak":st["inflight_peak"],"max_turns":turn_cap,
            "stalled_streams":st["stalled"],
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
    ap.add_argument("--max-turns",dest="max_turns",type=int,default=MAX_TURNS,
                    help=f"turns per conversation before it is retired (default {MAX_TURNS}). Effective "
                         "sample size is conversations, not requests, because input size is constant "
                         "within a conversation; lower values buy independent size draws at the same "
                         "token spend. Runs at different values are not comparable on cache_hit or "
                         "absolute latency, only size-controlled")
    ap.add_argument("--request-log",dest="request_log",
                    help="write one row per REQUEST to this .csv (flushed per row). Needed to "
                         "separate throttling from errors and client queueing from endpoint latency; "
                         "per-level rows cannot show either")
    ap.add_argument("--plan-log",dest="plan_log",
                    help="write the arrival times and cold-conversation sizes each level is about to "
                         "produce to this .csv/.jsonl BEFORE it runs. Does not change what is sent; "
                         "confirms the harness offered the workload you configured")
    ap.add_argument("--out",
                    help="write per-level results to .csv, .jsonl, or .json")
    ap.add_argument("--append",action="store_true",
                    help="append results to --out (useful when running multiple seeds)")
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
    req_handle = open_request_log(a.request_log)
    if req_handle: print(f"per-request log -> {a.request_log}")
    if a.plan_log:
        Path(a.plan_log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.plan_log).unlink(missing_ok=True)   # levels append; start clean
        print(f"pre-computed plan -> {a.plan_log}")
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=2000)) as client:
        for lv in levels:
            row = await run_level(client, prov, a.role, a.mode, lv, a.dur, use_affinity, a.seed, a.arrival,
                                  fixed_dist=a.fixed_dist, pool=a.pool, warmup=a.warmup,
                                  drain_grace=a.drain_grace, slo_e2e_ms=a.slo_e2e_ms,
                                  max_arrival_lag_ms=a.max_arrival_lag_ms,
                                  max_arrival_lag_intervals=a.max_arrival_lag_intervals,
                                  max_turns=a.max_turns, request_log=req_handle, plan_log=a.plan_log)
            row.update({
                "seed": a.seed, "role": a.role, "mode": a.mode, "arrival": a.arrival,
                "duration_s": a.dur, "warmup_s": a.warmup, "drain_grace_s": a.drain_grace,
                "slo_e2e_ms": a.slo_e2e_ms, "fixed_dist": a.fixed_dist,
                "max_pending_frac": a.max_pending_frac,
            })
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
                e95 = row.get("e2e_p95_s")
                e95_ms = (e95 * 1000) if e95 is not None else 0.0
                if row["error_count"] >= a.explode_errs or e95_ms >= a.explode_e2e_ms:
                    why = (f"errors={row['error_count']} (>= {a.explode_errs})" if row["error_count"] >= a.explode_errs
                           else f"E2E p95={e95_ms:.0f}ms (>= {a.explode_e2e_ms:.0f}ms)")
                    print(f"\n*** EXPLODE at offered {lv} QPS: {why}. Halting ramp -- higher levels skipped. ***",
                          flush=True)
                    break
    if req_handle:
        req_handle[0].close()
    if a.out:
        write_rows(a.out, rows, append=a.append)
        print(f"\nWrote {len(rows)} level result(s) to {a.out}")
    print("\n---DONE---")

if __name__ == "__main__":
    asyncio.run(main())
