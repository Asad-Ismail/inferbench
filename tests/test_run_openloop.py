"""Deterministic accounting tests for the open-loop runner; no network required."""
import asyncio
import contextlib
import io
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inferbench import run_openloop as R
from inferbench import summarize as S
from inferbench import workload as W

W.SIZE_SCALE = 0.002


class FakeProvider:
    def __init__(self, latency=0.02, ok=True, code="Boom", block=False):
        self.latency, self.ok, self.code, self.block = latency, ok, code, block

    async def run_turn(self, client, messages, max_tokens, session_id=None):
        if self.block:
            time.sleep(self.latency)
        else:
            await asyncio.sleep(self.latency)
        if not self.ok:
            return {"ok": False, "code": self.code}
        return {
            "ok": True, "ttft": self.latency * 0.3, "e2e": self.latency,
            "prompt_tokens": 100_000, "cached_tokens": 60_000, "completion_tokens": 150,
        }


def run(**kw):
    options = dict(
        client=None, provider=FakeProvider(), role="heavy", mode="fixed",
        qps=5.0, dur=1.0, use_affinity=False, seed=7, arrival="uniform",
        fixed_dist=True, pool=20, warmup=0.0, drain_grace=1.0, slo_e2e_ms=None,
        max_arrival_lag_ms=1e6, max_arrival_lag_intervals=1e6,
    )
    options.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(R.run_level(**options))


def test_finish_in_window():
    result = run(provider=FakeProvider(latency=0.02))
    assert result["offered"] > 0
    assert result["error_count"] == 0
    assert result["generator_dropped"] == 0
    assert result["pending_after_grace"] == 0
    assert result["scheduled"] == result["offered"] + result["generator_dropped"]
    assert result["completed_by_end"] >= result["offered"] - 1


def test_tail_finishes_during_grace():
    result = run(provider=FakeProvider(latency=0.4), qps=20.0)
    assert result["pending_after_grace"] == 0
    assert result["completed_by_end"] < result["offered"]
    assert result["drain_s"] > 0


def test_pending_after_grace():
    result = run(provider=FakeProvider(latency=2.5), qps=4.0, drain_grace=0.5)
    assert result["completed_by_end"] == 0
    assert result["pending_after_grace"] >= 2


def test_generator_drops_late_arrivals():
    result = run(
        provider=FakeProvider(latency=0.01), qps=50.0,
        max_arrival_lag_ms=1e-4, max_arrival_lag_intervals=1e-9,
    )
    assert result["generator_dropped"] > 0
    assert result["scheduled"] == result["offered"] + result["generator_dropped"]


def test_slo_boundary():
    assert run(provider=FakeProvider(latency=0.2), slo_e2e_ms=300)["goodput_qps"] > 0
    assert run(provider=FakeProvider(latency=0.2), slo_e2e_ms=100)["goodput_qps"] == 0.0


def test_errors_and_warmup():
    errors = run(provider=FakeProvider(ok=False, code="503"))
    assert errors["error_count"] == errors["offered"] > 0
    assert errors["completion_qps"] == 0.0
    full = run(qps=10.0, dur=2.0)
    warm = run(qps=10.0, dur=2.0, warmup=1.0)
    assert 0 < warm["offered"] < full["offered"]


def test_strict_and_diagnostic_pending_tolerance():
    assert R.pending_tolerance(1920, 0.0) == 0
    assert R.pending_tolerance(1920, 0.01) == 20


def test_structured_output_and_cross_seed_summary():
    rows = [
        {"offered_qps": 4.0, "seed": 1, "completion_qps": 3.8, "e2e_p95_s": 1.2, "error_counts": {}},
        {"offered_qps": 4.0, "seed": 2, "completion_qps": 3.6, "e2e_p95_s": 1.4, "error_counts": {}},
    ]
    with tempfile.TemporaryDirectory() as directory:
        csv_path = os.path.join(directory, "results.csv")
        jsonl_path = os.path.join(directory, "results.jsonl")
        R.write_rows(csv_path, rows)
        R.write_rows(jsonl_path, rows)
        assert len(S.read_rows([csv_path])) == 2
        summary = S.summarize(S.read_rows([jsonl_path]))
    assert summary[0]["runs"] == 2
    assert summary[0]["offered_qps"] == 4.0
    assert summary[0]["completion_qps_median"] == 3.7
    assert round(summary[0]["e2e_p95_s_spread"], 6) == 0.2


def test_max_turns_caps_conversation_reuse():
    """Lower cap => more cold conversations for the same request count."""
    few = run(qps=20.0, dur=1.0, max_turns=2)
    many = run(qps=20.0, dur=1.0, max_turns=8)
    assert few["max_turns"] == 2 and many["max_turns"] == 8
    assert few["cold_conversations"] > many["cold_conversations"]
    assert few["offered"] == many["offered"] > 0


def test_request_log_has_one_row_per_request():
    import csv as _csv
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "requests.csv")
        handle = R.open_request_log(path)
        result = run(qps=10.0, dur=1.0, request_log=handle)
        handle[0].close()
        with open(path) as f:
            rows = list(_csv.DictReader(f))
    assert len(rows) == result["completed_by_end"] > 0
    assert {"conv_id", "cold", "inflight", "http_status", "retry_after", "ttft_s"} <= set(rows[0])
    assert all(int(r["inflight"]) >= 1 for r in rows)
    assert all(r["conv_id"] for r in rows)


def test_request_log_records_throttling_not_just_error():
    """A 429 must arrive as a quota signal, not collapse into a generic error count."""
    import csv as _csv

    class Throttled(FakeProvider):
        async def run_turn(self, client, messages, max_tokens, session_id=None):
            await asyncio.sleep(self.latency)
            return {"ok": False, "code": 429, "http_status": 429, "retry_after": "60"}

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "requests.csv")
        handle = R.open_request_log(path)
        result = run(provider=Throttled(), qps=10.0, dur=1.0, request_log=handle)
        handle[0].close()
        with open(path) as f:
            rows = list(_csv.DictReader(f))
    assert result["error_count"] > 0
    assert all(r["http_status"] == "429" and r["retry_after"] == "60" for r in rows)


def test_plan_predicts_live_arrivals_without_changing_them():
    """The plan must reproduce the live schedule, and running with one must change nothing."""
    uniform, sizes = R.build_plan("heavy", 5.0, 1.0, 7, "uniform")
    assert len(uniform) == 4 and uniform[0] == 0.2   # 1/qps spacing, last arrival < dur
    assert len(sizes) == len(uniform) + 8
    # uniform arrivals are seed-independent by construction; only Poisson consumes the RNG
    assert uniform == R.build_plan("heavy", 5.0, 1.0, 8, "uniform")[0]
    poisson, _ = R.build_plan("heavy", 5.0, 1.0, 7, "poisson")
    assert poisson == R.build_plan("heavy", 5.0, 1.0, 7, "poisson")[0]   # deterministic in (seed, qps)
    assert poisson != R.build_plan("heavy", 5.0, 1.0, 8, "poisson")[0]   # and sensitive to seed
    with tempfile.TemporaryDirectory() as directory:
        plain = run(qps=5.0, dur=1.0, arrival="poisson")
        planned = run(qps=5.0, dur=1.0, arrival="poisson",
                      plan_log=os.path.join(directory, "plan.csv"))
        assert os.path.getsize(os.path.join(directory, "plan.csv")) > 0
    assert plain["offered"] == planned["offered"]


def test_cache_hit_mean_reported_alongside_median():
    result = run(qps=10.0, dur=1.0)
    assert result["cache_hit_mean"] is not None
    assert abs(result["cache_hit_mean"] - 0.6) < 1e-9   # FakeProvider: 60k cached of 100k


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(failures)


LADDER = [10_000.0, 25_000.0, 50_000.0, 100_000.0, 200_000.0, 400_000.0]


def test_ladder_deals_every_rung_an_equal_share():
    """Each block covers every rung once, so any prefix is balanced to within one block."""
    sizes = W.ladder_totals(LADDER, 40, seed=7)
    assert len(sizes) % len(LADDER) == 0
    for start in range(0, len(sizes), len(LADDER)):
        assert sorted(sizes[start:start + len(LADDER)]) == sorted(LADDER)
    # cycling the schedule (how run_level indexes it) must stay balanced
    cycled = [sizes[i % len(sizes)] for i in range(1000)]
    counts = [cycled.count(r) for r in LADDER]
    assert max(counts) - min(counts) <= 1


def test_ladder_order_is_not_a_repeating_pattern():
    sizes = W.ladder_totals(LADDER, 60, seed=7)
    blocks = [tuple(sizes[i:i + len(LADDER)]) for i in range(0, 60, len(LADDER))]
    assert len(set(blocks)) > 1, "every block identical => order is a fixed pattern"


def test_ladder_gives_every_level_the_same_size_mix():
    """The point of the ladder: a level-to-level difference cannot come from the size draw."""
    low = run(qps=5.0, dur=1.0, size_ladder=LADDER)
    high = run(qps=20.0, dur=1.0, size_ladder=LADDER)
    assert low["offered"] > 0 and high["offered"] > low["offered"]
    # same schedule walked at both rates => mean input agrees despite very different counts
    a, b = low["input_tokens_mean"], high["input_tokens_mean"]
    assert abs(a - b) / max(a, b) < 0.15, (a, b)


def test_ladder_plan_predicts_the_sizes_actually_sent():
    arrivals, sizes = R.build_plan("heavy", 5.0, 1.0, 7, "uniform", size_ladder=LADDER)
    assert arrivals and sizes
    assert set(sizes) <= set(LADDER)


def test_ladder_takes_precedence_over_fixed_dist():
    sizes = W.ladder_totals(LADDER, 12, seed=1)
    assert set(sizes) == set(LADDER)
    assert set(W.stratified_totals("heavy", 12)) != set(LADDER)


def _band(v):
    for hi in (20e3, 50e3, 100e3, 200e3, 400e3):
        if v < hi:
            return hi
    return 1e18


def test_stratified_keeps_the_real_distribution_shape():
    """The point of stratifying rather than laddering: the size mix must still be the
    profile's own heavy-tailed mix, not a flat one."""
    import random
    rng = random.Random(5)
    drawn = [W.sample_pct(W.ROLE_PROFILES["heavy"]["total"], rng) for _ in range(60_000)]
    sched = W.stratified_totals("heavy", 120)
    for hi in (20e3, 50e3, 100e3, 200e3, 400e3, 1e18):
        want = sum(1 for v in drawn if _band(v) == hi) / len(drawn)
        got = sum(1 for v in sched if _band(v) == hi) / len(sched)
        assert abs(got - want) < 0.03, (hi, got, want)


def test_every_level_sees_the_same_size_mix_whatever_its_length():
    """Levels differ in length -- a fast level completes more conversations than a slow one --
    so the mix a level actually sees must not depend on how far it got. Guards the ordering,
    not the set: an ordering that front-loads mid-sizes passes a whole-schedule check and
    still hands the two levels different workloads.

    Checked from one full cycle upward, which is the operating regime (levels run hundreds of
    conversations against a 120-long schedule). Below one cycle the mean still lags, because a
    heavy tail puts most of the mean in a handful of rare large sizes that no ordering can
    deliver early without distorting the mix.
    """
    sched = W.stratified_totals("heavy", 120)
    full = {hi: sum(1 for v in sched if _band(v) == hi) / len(sched)
            for hi in (20e3, 50e3, 100e3, 200e3, 400e3, 1e18)}
    mean_full = sum(sched) / len(sched)
    for n in (120, 300, 650, 1600):
        pref = [sched[i % len(sched)] for i in range(n)]
        for hi, want in full.items():
            got = sum(1 for v in pref if _band(v) == hi) / n
            assert abs(got - want) < 0.01, (n, hi, got, want)
        assert abs((sum(pref) / n) / mean_full - 1) < 0.03, (n, sum(pref) / n)
