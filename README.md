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

Contributions are welcome — bug fixes, clearer docs, new probes, provider quirks, and measurement methodology improvements.

1. Fork the repo and create a branch from `main`.
2. Make a focused change (prefer one concern per PR).
3. Open a pull request against [Asad-Ismail/inferbench](https://github.com/Asad-Ismail/inferbench) with a short summary of *what* changed and *why*.
4. If the change affects reported metrics, note how you validated it (command + rough expected effect).

Issues for bugs or design questions are also fine before investing in a large PR.

---

If Inferbench helps you size or compare serving stacks, please [star the repository](https://github.com/Asad-Ismail/inferbench) or share it with others doing the same work.
