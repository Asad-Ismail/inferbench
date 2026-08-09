# Inferbench

<p align="center">
  <img src="assets/inferbench-logo.png" alt="Inferbench logo" width="120">
</p>

Load test for LLM inference endpoints. Reports sustained QPS, TTFT/E2E percentiles, prefix-cache hit rate, and the goodput knee.

Open-loop fires requests at a target QPS regardless of how slow the server is (arrival-driven capacity); closed-loop keeps N concurrent conversations in flight and only issues the next turn when one finishes (`run.py`). Prefer open-loop (`run_openloop.py`) for the goodput knee.

If you find this useful, [star the repo](https://github.com/Asad-Ismail/inferbench) or share it with someone sizing LLM serving.

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
    --arrival poisson --seed 7 --dur 120 --levels 2,4,6,8,10 --fixed-dist --stop-on-explode
```

Example row (latencies in ms):

| QPS | Sustained | In p50 | In mean | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | Out p50 | Cache | N | Err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.00 | 3.59 | 80k | 110k | 6000 | 20000 | 32000 | 10000 | 25000 | 43000 | 170 | 0.80 | 650 | 0 |

Ends with the goodput knee and a fleet-size estimate. Reverse `--levels` to confirm order doesn't matter.

## Useful flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--provider` | required | `PROVIDERS` key, or env: `<NAME>_BASE_URL` / `_MODEL` / `_API_KEY` |
| `--levels` | `0.1,…,5` | Offered QPS points |
| `--dur` | `90` | Seconds per level; longer → stabler tails |
| `--warmup` | `0` | Discard first N seconds from all metrics |
| `--drain-grace` | `300` | Max seconds to wait for in-flight after each level, then cancel |
| `--seed` | `7` | Repeat across seeds for a defensible p95 |
| `--arrival` | `uniform` | `poisson` or `uniform` |
| `--mode` | `real` | Cache breaks: `real` / `fixed` (never) / `broken` (every turn) |
| `--fixed-dist` | off | Same mean-stabilized size schedule every level |
| `--stop-on-explode` | off | Stop ramp on congestion (`--explode-e2e-ms`, `--explode-errs`) |
| `--affinity` | off | Sticky session header (default `X-Session-Id`; override with `INFERBENCH_SESSION_HEADER`) |
| `--role` | `heavy` | Size profile: `heavy` or `light` |
| `--scale` | `1.0` | Scale token sizes (`<1` for dry runs) |

Env: `INFERBENCH_CACHE_FRACTION` (default `0.59`), `INFERBENCH_REASONING_EFFORT`, `INFERBENCH_SESSION_HEADER`. Optional: `INFERBENCH_ENV=/path/to/file` loads `KEY=value` lines into the process (real env vars win). Nothing auto-loads a local `.env`.

`--dur` tip: use `--warmup 30–60`, then hold until you have ~1k steady-state completions (or until sustained/p95 stop moving).

## Scripts

| Script | Role |
| --- | --- |
| `run_openloop.py` | Open-loop QPS ramp → knee |
| `knee_at_hit.py` | Hold at a locked cache-hit; drift check |
| `scaling_probe.py` | Prefill TTFT vs context length |
| `decode_probe.py` | Decode tok/s |
| `run.py` | Closed-loop concurrency ramp |
| `validate_comp.py` | Smoke-check warm-turn cache-hit under affinity |
| `workload.py` | Size distribution + cache composition (`ROLE_PROFILES`) |
| `providers.py` | OpenAI-compatible client + registry |
| `plot_scaling.py` | Plots for `scaling_probe.py` CSVs in `docs/data/` (not shipped) |

## Endpoint

Export `<NAME>_BASE_URL`, `<NAME>_MODEL`, `<NAME>_API_KEY` (as in Run), or register the provider in `providers.py`. Edit `ROLE_PROFILES` / `INFERBENCH_CACHE_FRACTION` to match your traffic before sizing. For cache affinity, set `INFERBENCH_SESSION_HEADER` (or `<NAME>_SESSION_HEADER`) to whatever your router expects.
