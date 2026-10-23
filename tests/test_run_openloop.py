"""Deterministic accounting tests for the open-loop runner; no network required."""
import asyncio
import contextlib
import io
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inferbench import run_openloop as R
from inferbench import summarize as S
from inferbench import workload as W
from inferbench.providers import resolve_token_param, token_budget_fields
from inferbench.utils import pct

W.SIZE_SCALE = 0.002


class FakeProvider:
    def __init__(self, latency=0.02, ok=True, code="Boom"):
        self.latency, self.ok, self.code = latency, ok, code

    async def run_turn(self, client, messages, max_tokens, session_id=None):
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
        fixed_dist=True, pool=20, warmup=0.0, drain_grace=1.0,
        max_arrival_lag_ms=1e6, max_arrival_lag_intervals=1e6,
    )
    options.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(R.run_level(**options))


def test_pct():
    assert pct([], 0.5) is None
    assert pct([10], 0.5) == 10
    assert pct([0, 10], 0.5) == 5


def test_finish_in_window():
    result = run(provider=FakeProvider(latency=0.02))
    assert result["fired"] > 0
    assert result["error_count"] == 0
    assert result["dropped"] == 0
    assert result["left"] == 0
    assert result["scheduled"] == result["fired"] + result["dropped"]
    assert result["done"] >= result["fired"] - 1


def test_tail_finishes_during_grace():
    result = run(provider=FakeProvider(latency=0.4), qps=20.0)
    assert result["left"] == 0
    assert result["done"] < result["fired"]
    assert result["drain_s"] > 0


def test_left_after_grace():
    result = run(provider=FakeProvider(latency=2.5), qps=4.0, drain_grace=0.5)
    assert result["done"] == 0
    assert result["done_qps"] == 0.0
    assert result["e2e_p50_s"] is None
    assert result["left"] >= 2


def test_window_excludes_drain_from_done_and_latency():
    """Requests that only finish in grace are not Done and do not enter E2E."""
    result = run(provider=FakeProvider(latency=0.4), qps=20.0, dur=1.0, drain_grace=1.0)
    assert result["done"] < result["fired"]
    assert result["e2e_p50_s"] == 0.4
    assert result["done_qps"] == result["done"] / 1.0


def test_generator_drops_late_arrivals():
    result = run(
        provider=FakeProvider(latency=0.01), qps=50.0,
        max_arrival_lag_ms=1e-4, max_arrival_lag_intervals=1e-9,
    )
    assert result["dropped"] > 0
    assert result["scheduled"] == result["fired"] + result["dropped"]


def test_errors_and_warmup():
    errors = run(provider=FakeProvider(ok=False, code="503"))
    assert errors["error_count"] == errors["fired"] > 0
    assert errors["done_qps"] == 0.0
    full = run(qps=10.0, dur=2.0)
    warm = run(qps=10.0, dur=2.0, warmup=1.0)
    assert 0 < warm["fired"] < full["fired"]


def test_token_budget_fields():
    assert token_budget_fields(230) == {"max_tokens": 230, "max_completion_tokens": 230}
    assert token_budget_fields(8, "max_tokens") == {"max_tokens": 8}
    assert token_budget_fields(8, "max_completion_tokens") == {"max_completion_tokens": 8}
    try:
        token_budget_fields(8, "max_output_tokens")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert resolve_token_param(None, "", "both") == "both"
    assert resolve_token_param("max_tokens", "both") == "max_tokens"


def test_max_ctx_clamps_totals():
    old = W.MAX_CTX
    try:
        W.MAX_CTX = 8000
        totals = W.stratified_totals("heavy", 20)
        assert max(totals) <= 8000
        conv = W.Conversation("heavy", random.Random(1), total=100_000)
        assert conv.total == 8000
        W.MAX_CTX = 0
        conv2 = W.Conversation("heavy", random.Random(1), total=100_000)
        assert conv2.total == 100_000
    finally:
        W.MAX_CTX = old


def test_strict_and_diagnostic_pending_tolerance():
    assert R.pending_tolerance(1920, 0.0) == 0
    assert R.pending_tolerance(1920, 0.01) == 20


def test_structured_output_and_cross_seed_summary():
    rows = [
        {"offered_qps": 4.0, "seed": 1, "done_qps": 3.8, "e2e_p95_s": 1.2, "error_counts": {}},
        {"offered_qps": 4.0, "seed": 2, "done_qps": 3.6, "e2e_p95_s": 1.4, "error_counts": {}},
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
    assert summary[0]["done_qps_median"] == 3.7
    assert round(summary[0]["e2e_p95_s_spread"], 6) == 0.2


def test_stabilization_verdict_is_sample_aware():
    """Steady state = backlog not growing; the drift test only bites with enough samples.
    Anchored on a real case: 1 QPS x 60s gives ~62 completions whose 45% half-vs-half drift
    is pure noise while the backlog is draining -- that must count as stable."""
    assert R.stabilization_verdict(62, 0.45, -3, 60, 0.20) == (True, True)
    assert R.stabilization_verdict(62, 0.10, 30, 60, 0.20)[0] is False   # backlog grows: unstable
    assert R.stabilization_verdict(200, 0.08, 2, 200, 0.20) == (True, False)
    assert R.stabilization_verdict(200, 0.35, 0, 200, 0.20)[0] is False  # real drift
    assert R.stabilization_verdict(300, 0.05, 40, 300, 0.20)[0] is False # backlog past tolerance


def test_level_duration_auto_sizes_to_min_samples():
    base, warm = 90, 30
    assert R.level_duration(10, base, warm) == 90                    # high QPS: base wins
    d1 = R.level_duration(1, base, warm)
    assert d1 > base and (d1 - warm) * 1 >= R.STAB_MIN_SAMPLES       # 1 QPS: window*qps >= 100
    assert R.level_duration(0.001, base, warm) == R.STAB_MAX_DUR     # capped, never hangs


def test_stabilized_true_on_steady_state():
    r = run(provider=FakeProvider(latency=0.02), qps=10.0, dur=2.0, drain_grace=1.0)
    assert r["stabilized"] is True
    assert r["stab_drift"] < 0.20 or r["stab_undersampled"]


def test_not_stabilized_when_backlog_grows():
    r = run(provider=FakeProvider(latency=5.0), qps=10.0, dur=1.0, drain_grace=0.2)
    assert r["stabilized"] is False
    assert r["backlog_growth"] > 0


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
