# Inferbench

<p align="center">
  <img src="assets/inferbench-logo.png" alt="Inferbench" width="120">
</p>

Load generator for long-context, multi-turn LLM endpoints.

- Open-loop QPS ramp: fire on schedule, drop late arrivals, count in-window completions
- Closed-loop concurrency ramp (`run.py`)
- Multi-turn prefix cache with session affinity
- `--max-ctx` clamp (default `heavy` draws up to 780k tokens)

## Run

```bash
pip install -e .

export MYAPI_BASE_URL="https://your-host/v1"
export MYAPI_MODEL="your-model"
export MYAPI_API_KEY="sk-..."

python3 inferbench/run_openloop.py --provider myapi --mode real --affinity \
    --max-ctx 128000 --arrival poisson --seed 7 --dur 600 --warmup 60 \
    --drain-grace 90 --levels 2,4,6,8,10 --fixed-dist --stop-on-explode \
    --out results-seed-7.jsonl
```

Other flags: `--help`.

## Reading the table

OfferedQPS is the rate you asked for. DoneQPS is successes that finished inside the window, divided by that window.

A level **held** when DoneQPS ≈ OfferedQPS, Growth ≈ 0, Dropped = 0, Left = 0.

You pick the SLO. Goodput is the highest OfferedQPS that held **and** whose latency (usually E2Ep95) is under that SLO.

```
OfferedQPS  DoneQPS  Growth  Dropped  Left  E2Ep95
      2.00     2.00       0        0     0     12s   held, under a 20s p95 SLO
      4.00     3.10      40        0     0     45s   missed the rate and the SLO
```

Goodput at a 20s E2E p95 SLO is 2 QPS, not 4. p50/p95/p99 tell you which side of the bar the bulk sits on, not the exact fraction under it.

Latency and tokens are the same population as Done. Drain is leftover work after arrivals stop; finishing later does not raise DoneQPS.

```bash
python -m inferbench.summarize results-seed-*.jsonl --out summary.csv
```

## Things that bite

`--affinity` is required on a multi-replica router. It sends a sticky session header (`X-Session-Id` by default; override with `INFERBENCH_SESSION_HEADER`). Without it, prefix cache misses.

`--max-ctx` clamps every prompt. Against a 32k or 128k model, uncapped `heavy` is errors or silent truncation, not a smaller workload.

`INFERBENCH_CACHE_FRACTION` (default `0.59`) is prompt composition — stable vs fresh — not a traffic-weighted hit. `--mode fixed` means never break, not 100% hit; turn 1 is always cold. `INFERBENCH_ENV=/path` loads `KEY=value` lines; real env vars win. Nothing auto-loads a local `.env`.

Defaults are `--dur 90 --warmup 30` and levels from 1 QPS. That is tens of completions, not ~1k. p99 is noise until you have enough in-window samples and Growth ≈ 0.

`--timeout` defaults to 1800s. The heavy output tail is 35k tokens; one decode can outlast a shorter timeout and look like an outage. `--timeout 0` disables it.

`Dropped > 0` means this process missed the schedule (one asyncio client, 2000 connections). That is often not the endpoint. Any drop halts the ramp.

`locked_hit.py` pins cache-hit by priming a pool, then ramps. A busy pool waits (closed-loop). The `ok/wall` column includes drain. Failed primes abort.

## Other scripts

- `locked_hit.py` — pin cache-hit, then ramp
- `scaling_probe.py` — prefill TTFT vs context length
- `decode_probe.py` — decode tok/s
- `run.py` — closed-loop concurrency ramp
- `validate_comp.py` — smoke-check warm-turn cache-hit under affinity

Edit `ROLE_PROFILES` / `INFERBENCH_CACHE_FRACTION` to match your traffic.

## Tests

```
python tests/test_run_openloop.py
```
