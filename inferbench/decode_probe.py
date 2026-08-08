"""Clean decode-rate probe: measure output tokens/sec.

Short prompt (so prefill/TTFT is negligible) + a fixed large output, single request. Then
  tps = completion_tokens / (E2E - TTFT)
per request — the correct per-request decode rate, not percentile arithmetic. Sweeps output
length because decode slows as the KV cache grows during generation. The result is comparable
to public per-provider "output speed" benchmarks.

Usage:
  python3 decode_probe.py --provider myapi --outs 256,512,1024,2048 --repeats 3
"""
import argparse, asyncio, random, statistics, os, sys
import httpx

try:
    from inferbench import workload as W
    from inferbench.providers import build_provider
except ImportError:
    import workload as W
    from providers import build_provider

RUN_SALT = os.environ.get("SALT") or os.urandom(6).hex()


def short_prompt(seed):
    # ~1k-token prompt so prefill is small and TTFT ~ negligible vs decode.
    rng = random.Random(str((RUN_SALT, seed)))
    body = " ".join(rng.choice(W._VOCAB) for _ in range(1000))
    # ask for a long free-form continuation so the model actually emits many tokens
    return [{"role": "system", "content": "You are a verbose assistant. Keep writing in detail."},
            {"role": "user", "content": "Summarize and then expand at length on the following notes: " + body}]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--outs", default="256,512,1024,2048")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    try:
        prov = build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(2)
    outs = [int(x) for x in a.outs.split(",")]
    print(f"DECODE PROBE | provider={a.provider} repeats={a.repeats} salt={RUN_SALT}")
    print(f"{'max_out':>8} | {'got_out':>8} | {'TTFT_s':>7} | {'E2E_s':>7} | {'decode_s':>8} | {'tps':>6} | n")
    print("-" * 70)
    async with httpx.AsyncClient(timeout=900) as client:
        for mo in outs:
            tpss, gots, ttfts, e2es = [], [], [], []
            for i in range(a.repeats):
                r = await prov.run_turn(client, short_prompt((mo, i)), mo, session_id=None)
                if not r.get("ok"):
                    print(f"    ! error max_out={mo}: {r.get('code')}", flush=True); continue
                ttft, e2e, ot = r.get("ttft"), r.get("e2e"), r.get("completion_tokens")
                if ttft is None or e2e is None or not ot or e2e <= ttft or ot <= 1:
                    continue
                tpss.append(ot / (e2e - ttft)); gots.append(ot); ttfts.append(ttft); e2es.append(e2e)
            if not tpss:
                print(f"{mo:>8} |  (no valid samples)"); continue
            md = statistics.median
            print(f"{mo:>8} | {md(gots):>8.0f} | {md(ttfts):>7.2f} | {md(e2es):>7.2f} | "
                  f"{md(e2es)-md(ttfts):>8.2f} | {md(tpss):>6.0f} | {len(tpss)}", flush=True)
    print("\n(Compare against public per-provider output-speed benchmarks for your model.)")
    print("---DONE---", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
