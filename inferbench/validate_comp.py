"""Smoke-check that warm turns land near INFERBENCH_CACHE_FRACTION under affinity.

Usage:
  python3 validate_comp.py --provider myapi
"""
import argparse
import asyncio
import random
import statistics
import sys

import httpx

try:
    from inferbench import workload as W
    from inferbench.providers import build_provider, ensure_affinity
except ImportError:
    import workload as W
    from providers import build_provider, ensure_affinity


async def main():
    ap = argparse.ArgumentParser(description="Validate warm-turn cache-hit composition")
    ap.add_argument("--provider", required=True)
    ap.add_argument("--role", default="heavy", choices=sorted(W.ROLE_PROFILES))
    ap.add_argument("--convs", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    W.SIZE_SCALE = 1.0
    try:
        prov = build_provider(a.provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    ensure_affinity(prov, True)
    target = W.CACHE_FRACTION
    rng = random.Random(a.seed)
    async with httpx.AsyncClient(timeout=600) as client:
        rows = []
        for c in range(a.convs):
            conv = W.Conversation(a.role, random.Random(rng.random()))
            sid = f"valconv-{c}"
            print(f"\nconv{c}: total~{conv.total:.0f} stable~{conv.stable_tokens} fresh~{conv.fresh_tokens}")
            for t in range(a.turns):
                msgs, mo = conv.next_turn(break_cache=False)
                r = await prov.run_turn(client, msgs, mo, session_id=sid)
                pt, ct = r.get("prompt_tokens"), r.get("cached_tokens")
                hit = (ct / pt) if (pt and ct is not None) else None
                print(f"  turn{t+1}: prompt_tok={pt} cached={ct} "
                      f"hit={hit if hit is None else round(hit, 3)} "
                      f"ttft={r.get('ttft')} ok={r.get('ok')} {r.get('code', '')}")
                if t >= 1 and hit is not None:
                    rows.append(hit)
        if rows:
            print(f"\nWARM-TURN (t2+) cache-hit: p50={statistics.median(rows):.3f} "
                  f"min={min(rows):.3f} max={max(rows):.3f}  (target ~{target})")
        else:
            print("\nno warm-turn hit samples (endpoint may not report cached_tokens, "
                  "or affinity/header mismatch)", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
