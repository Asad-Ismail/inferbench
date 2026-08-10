# Inferbench

<p align="center">
  <img src="assets/inferbench-logo.png" alt="Inferbench logo" width="120">
</p>

Load test for LLM inference endpoints. Reports fixed-window completion QPS, bounded-drain diagnostics, optional SLO goodput, TTFT/E2E percentiles, prefix-cache hit rate, and a structured response curve.

Open-loop fires requests at a target QPS regardless of how slow the server is (arrival-driven capacity); closed-loop keeps N concurrent conversations in flight and only issues the next turn when one finishes (`run.py`). Prefer open-loop (`run_openloop.py`) for the goodput knee.

## Install

```
pip install -e .
pip install -e ".[plots]"   # optional plotting deps
```

## Run

```bash
export MYAPI_BASE_URL="https://your-host/v1"
export MYAPI_MODEL="your-model"
export MYAPI_API_KEY="sk-..."

python3 inferbench/run_openloop.py --provider myapi --mode real \
    --arrival poisson --seed 7 --dur 600 --warmup 60 \
    --drain-grace 90 --slo-e2e-ms 60000 \
    --levels 2,4,6,8,10 --fixed-dist --stop-on-explode \
    --out results-seed-7.jsonl
```

Output includes `Offer`, `Sched`, `N`, `GDrop`, `LagP95`, `LagMax`, `Done@T`, `CompQPS`, `B@T`, `dB`, `Drain`, `Pending@G`, `SLOQPS`, and latency/error columns. `GDrop` is an overdue arrival dropped by the load generator rather than burst-fired late; any drop halts the ramp. `CompQPS` is successful completions during the fixed post-warm-up offer window, not completions divided by an unbounded drain. `B@T` is all in-flight work when arrivals stop, including warm-up requests; `dB` is total backlog growth across the measurement window. `Pending@G` is work still pending after `--drain-grace`; strict mode halts on any pending request. Set `--slo-e2e-ms` to report per-request SLO goodput.

Structured output uses explicit field units: latency and duration columns end in `_s`, arrival-lag columns end in `_ms`, token columns include `_tokens`, and throughput columns end in `_qps`.

Inferbench does not choose a knee or fleet size. The result rows are the measurement; select an operating point with your own SLOs for TTFT, E2E, error rate, backlog, and cost.

## Useful flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--provider` | required | `PROVIDERS` key, or env: `<NAME>_BASE_URL` / `_MODEL` / `_API_KEY` |
| `--levels` | `0.1,…,5` | Offered QPS points |
| `--dur` | `90` | Seconds per level; longer → stabler tails |
| `--warmup` | `0` | Discard first N seconds from all metrics |
| `--drain-grace` | `300` | Bounded seconds to wait after arrivals stop |
| `--slo-e2e-ms` | off | Per-request E2E deadline for SLO goodput |
| `--max-arrival-lag-ms` | `50` | Absolute floor (ms) for arrival-lag budget |
| `--max-arrival-lag-intervals` | `1.0` | Lag budget in mean inter-arrivals |
| `--max-pending-frac` | `0` | Strict default; positive value is diagnostic only |
| `--max-turns` | `2` | Turns per conversation before it retires; see Sample size |
| `--request-log` | off | Write one row per request to a `.csv` |
| `--plan-log` | off | Write each level's predicted arrivals and sizes before it runs |
| `--out` | off | Write per-level rows to `.csv`, `.jsonl`, or `.json` |
| `--append` | off | Append to `--out`, useful for multiple seeds |
| `--seed` | `7` | Repeat across seeds for a defensible p95 |
| `--arrival` | `uniform` | `poisson` or `uniform` |
| `--mode` | `real` | Cache breaks: `real` / `fixed` (never) / `broken` (every turn) |
| `--fixed-dist` | off | Same mean-stabilized size schedule every level |
| `--pool` | `40` | Number of sizes in the fixed schedule |
| `--stop-on-explode` | off | Stop ramp on congestion (`--explode-e2e-ms`, `--explode-errs`) |
| `--explode-e2e-ms` | `45000` | E2E p95 threshold for congestion stop |
| `--explode-errs` | `3` | Error-count threshold for congestion stop |
| `--affinity` | off | Sticky session header (default `X-Session-Id`; override with `INFERBENCH_SESSION_HEADER`) |
| `--role` | `heavy` | Size profile: `heavy` or `light` |
| `--scale` | `1.0` | Scale token sizes (`<1` for dry runs) |

Env: `INFERBENCH_CACHE_FRACTION` (default `0.59`), `INFERBENCH_REASONING_EFFORT`, `INFERBENCH_SESSION_HEADER`. Optional: `INFERBENCH_ENV=/path/to/file` loads `KEY=value` lines into the process (real env vars win). Nothing auto-loads a local `.env`.

`--dur` tip: use `--warmup 30–60`, then hold until you have ~1k steady-state completions and stable backlog. A level is not sustained merely because requests eventually drain after arrivals stop.

## Per-request log

`--request-log requests.csv` writes one row per request, flushed as each completes. Per-level
rows cannot separate two things that look identical in aggregate:

- **Throttling from failure.** `http_status` and `retry_after` carry a 429 as a quota signal
  instead of collapsing it into an error count. On one endpoint this is how a rate limit was
  found reproducing at 2.74M input tokens per trailing minute across separate runs, far below
  the quota the key was rated for.
- **Client queueing from endpoint latency.** `inflight` at dispatch, plus `arrival_lag_ms` and
  the `cold` flag, show whether a slow number came from the server or from the load generator.

`finish_reason` is worth reading directly: `length` means the output cap bound rather than the
model stopping. A row with HTTP 200, no `prompt_tokens` and no `finish_reason` is a **stalled
stream** — it contributes no tokens but its E2E still lands in the tail. Those are counted in
`stalled_streams`. Filter on `prompt_tokens` before computing any token statistic.

**Report `cache_hit_mean`, not `cache_hit_p50`.** The median is pinned near the design target
until more than half of requests are cold, then flips to the cold mode, so it is wrong in both
regimes. Measured on one run it read 0.5885 / 0.5891 / 0.5892 across a 4x change in rate while
the mean moved 0.375 / 0.420 / 0.453.

## Sample size

Effective sample size is **conversations, not requests**. Input size is constant within a
conversation, so its 2nd..Nth turns are near-duplicates that cost tokens and buy no precision.
Bootstrap by conversation; resampling by request treats near-duplicate turns as independent and
gives intervals roughly 2x too narrow.

`--max-turns` trades duplicate turns for independent size draws at the same request count and
token spend. Two real runs, 95% CI on TTFT p50 by conversation bootstrap:

```
  37 requests,   7 conversations -> TTFT p50 5.30s, 95% CI [2.50, 11.16]
 486 requests, 129 conversations -> TTFT p50 4.37s, 95% CI [ 3.79,  5.12]
```

The first spans both passing and failing a 3s SLO. Roughly 250 conversations per level resolves
a 1s difference. The default of 2 yields 0.508 conversations per request against 0.184 at 8.

The trade-off: turn 1 is always cold, so a lower cap raises cold share, lowers cache hit, and
raises absolute TTFT. Runs at different `--max-turns` are comparable only size-controlled.
Use `--max-turns 8` for longer conversations.

## Plan log

`--plan-log plan.csv` writes the arrival times and cold-conversation sizes each level is about
to produce, before it runs. **It does not change what is sent** — it reads from throwaway
`Random` copies of the live streams, so it predicts the schedule rather than replacing it.

It confirms the harness offered the workload you configured. Arrivals are fully determined by
`(seed, qps)`, so a mismatch means the generator fell behind rather than the RNG drifting. A
seeded Poisson stream at a nominal 0.2 QPS held 139 arrivals in 900s, not 180 — that level
offered 0.156 QPS, and anything dividing by the nominal rate would understate it by 30%.

Conversation sizes are determined only up to order: the Nth cold conversation always has the
same size, but how many a level creates depends on latency, because a slow endpoint starves the
recycle pool and starts more cold, more expensive conversations. Comparing `cold_conversations`
against the plan measures that drift.

## Multi-seed summary

Run each seed into a separate structured output, then summarize the curve without applying a policy:

```bash
python -m inferbench.summarize results-seed-*.jsonl --out summary.csv
```

The summary reports run count plus median, minimum, maximum, and spread for every numeric metric at each offered QPS. It does not grade SLOs or select a knee.

## Scripts

| Script | Role |
| --- | --- |
| `run_openloop.py` | Open-loop QPS ramp → knee |
| `knee_at_hit.py` | Hold at a locked cache-hit; drift check |
| `scaling_probe.py` | Prefill TTFT vs context length |
| `decode_probe.py` | Decode tok/s |
| `run.py` | Closed-loop concurrency ramp |
| `summarize.py` | Cross-seed median/min/max/spread; no policy grading |
| `validate_comp.py` | Smoke-check warm-turn cache-hit under affinity |
| `workload.py` | Size distribution + cache composition (`ROLE_PROFILES`) |
| `providers.py` | OpenAI-compatible client + registry |
| `plot_scaling.py` | Plots for `scaling_probe.py` CSVs in `docs/data/` (not shipped) |

## Tests

Deterministic open-loop accounting tests use a fake provider and require no network:

```bash
python tests/test_run_openloop.py
```

They cover fixed-window completion, tail-in-grace, pending-after-grace, scheduler drops, SLO boundaries, error accounting, warm-up exclusion, and strict versus diagnostic pending policy.

## Endpoint

Export `<NAME>_BASE_URL`, `<NAME>_MODEL`, `<NAME>_API_KEY` (as in Run), or register the provider in `providers.py`. Edit `ROLE_PROFILES` / `INFERBENCH_CACHE_FRACTION` to match your traffic before sizing. For cache affinity, set `INFERBENCH_SESSION_HEADER` (or `<NAME>_SESSION_HEADER`) to whatever your router expects.

## Contributing

Contributions are welcome: open an issue or send a focused pull request.
For measurement changes, include the command you used and what you observed.

---

If Inferbench helps you size or compare serving stacks, please [star the repository](https://github.com/Asad-Ismail/inferbench) or share it with others doing the same work.
