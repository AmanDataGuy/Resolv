"""
# tests/test_metrics.py

**What:** the scoring math: pass^k, Wilson, McNemar, the scorecard, and the task-clustered
bootstrap, paired comparisons and Holm correction.
**Why:** every published number goes through these functions; expected values are worked by hand, never captured
from a previous run.
**Used by:** pytest (keyless).
"""
import pytest

from evals.metrics import (cluster_bootstrap, design_effect, holm, mcnemar, paired_cluster_diff, pass_hat_k,
                          pass_k_stat, quantile_ci, rate_stat, scorecard, standard_error, wilson)


# rows for tasks with given (successes, runs)
def _rows(spec: dict[str, tuple[int, int]]) -> list[dict]:
    return [{"task_id": t, "resolved": i < c} for t, (c, n) in spec.items() for i in range(n)]


# unbiased pass^k known values
@pytest.mark.parametrize("n, c, k, expected", [(5, 4, 3, 0.4), (5, 5, 3, 1.0), (5, 2, 3, 0.0), (5, 0, 3, 0.0),
                                               (5, 4, 1, 0.8), (10, 7, 1, 0.7), (5, 3, 3, 0.1)])
def test_pass_hat_k_known_values(n, c, k, expected):
    assert pass_hat_k(n, c, k) == pytest.approx(expected)


# the naive (c/n)^k form overstates
def test_naive_form_overstates():
    assert (4 / 5) ** 3 > pass_hat_k(5, 4, 3)


# drawing more attempts than were run is a bug
def test_k_greater_than_n_raises():
    with pytest.raises(ValueError):
        pass_hat_k(3, 3, 5)


# Wilson stays sensible at the boundary
def test_wilson():
    lo, hi = wilson(20, 20)
    assert 0.83 < lo < 0.85 and hi == 1.0
    assert wilson(0, 0) == (0.0, 0.0)
    assert (wilson(900, 1000)[1] - wilson(900, 1000)[0]) < (wilson(9, 10)[1] - wilson(9, 10)[0])


# the +0.016 extractor gain sat inside one standard error
def test_standard_error():
    assert standard_error(0.759, 466) == pytest.approx(0.0198, abs=1e-3)
    assert standard_error(1.0, 20) == 0.0 and standard_error(0.5, 0) == 0.0


# McNemar: lopsided is significant, balanced is not, symmetric in its arguments
def test_mcnemar():
    assert mcnemar(30, 10)[1] is True and mcnemar(20, 18)[1] is False
    assert mcnemar(0, 0) == (0.0, False)
    assert mcnemar(30, 10)[0] == mcnemar(10, 30)[0]


# clustered CI is wider than Wilson when runs within a task are identical and tasks vary
def test_clustered_ci_wider_than_wilson_with_identical_within_task_outcomes():
    rows = _rows({f"t{i}": (5 if i < 30 else 0, 5) for i in range(40)})
    lo, hi = cluster_bootstrap(rows, rate_stat("resolved"))
    wlo, whi = wilson(150, 200)
    assert (hi - lo) > (whi - wlo)


# with one run per task the clustered and Wilson intervals agree closely
def test_one_run_per_task_matches_wilson():
    rows = _rows({f"t{i}": (1 if i < 150 else 0, 1) for i in range(200)})
    lo, hi = cluster_bootstrap(rows, rate_stat("resolved"))
    wlo, whi = wilson(150, 200)
    assert abs(lo - wlo) < 0.02 and abs(hi - whi) < 0.02


# the bootstrap is deterministic for a seed
def test_seed_is_deterministic():
    # 13 tasks x 5 attempts: varied enough that two seeds land on different intervals (4 x 4 collided).
    rows = _rows({f"t{i}": (i % 5, 5) for i in range(13)})
    assert cluster_bootstrap(rows, pass_k_stat(2)) == cluster_bootstrap(rows, pass_k_stat(2))
    assert cluster_bootstrap(rows, pass_k_stat(2), seed=1) != cluster_bootstrap(rows, pass_k_stat(2), seed=2)


# identical arms give a paired diff of 0 with a CI containing 0
def test_paired_diff_of_identical_arms_contains_zero():
    rows = _rows({f"t{i}": (i % 3, 3) for i in range(12)})
    diff, lo, hi, p = paired_cluster_diff(rows, rows, pass_k_stat(1))
    assert diff == 0.0 and lo <= 0 <= hi and p == 1.0


# a clear drop is detected with an upper bound below zero
def test_paired_diff_detects_a_drop():
    base = _rows({f"t{i}": (2, 2) for i in range(10)})
    now = _rows({f"t{i}": (0, 2) for i in range(10)})
    diff, lo, hi, p = paired_cluster_diff(now, base, pass_k_stat(1))
    assert diff == -1.0 and hi < 0 and p <= 1e-3


# only clusters present in both arms are paired
def test_paired_diff_uses_shared_clusters_only():
    a = _rows({"x": (1, 1), "y": (1, 1)})
    b = _rows({"x": (1, 1), "z": (0, 1)})
    assert paired_cluster_diff(a, b, pass_k_stat(1))[0] == 0.0


# Holm on the textbook example
def test_holm_known_example():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "b": False, "c": False}


# a clustered quantile CI brackets the observed value
def test_quantile_ci_brackets():
    rows = [{"task_id": f"t{i % 5}", "e2e_s": float(i)} for i in range(50)]
    lo, hi = quantile_ci(rows, "e2e_s", 50, B=500)
    assert lo <= 25 <= hi


# ICC is high when tasks are all-or-nothing and the design effect follows
def test_design_effect():
    out = design_effect(_rows({f"t{i}": (5 if i % 2 else 0, 5) for i in range(10)}))
    assert out["icc"] == pytest.approx(1.0) and out["deff"] == pytest.approx(5.0) and out["n_eff"] == 10.0


# scorecard on 2 tasks x 5 runs, one perfect and one flaky
class TestScorecard:
    # the shared runs
    @staticmethod
    def _runs():
        return ([{"task_id": "a", "resolved": True, "steps": 3} for _ in range(5)]
                + [{"task_id": "b", "resolved": i < 4, "steps": 4} for i in range(5)])

    # per-task averaging is the estimator
    def test_pass_k_averages_per_task(self):
        card = scorecard(self._runs(), k=3)
        assert card["tasks"] == 2 and card["runs"] == 10
        assert card["pass^3"] == pytest.approx(0.7) and card["pass^1"] == pytest.approx(0.9)

    # clustered CIs sit next to the point estimates
    def test_clustered_fields_present(self):
        card = scorecard(self._runs(), k=3)
        lo, hi = card["resolve_ci95_clustered"]
        assert lo <= 0.9 <= hi and card["pass^3_ci95_clustered"] is not None

    # safety and block rates with the right denominators
    def test_rates(self):
        runs = self._runs()
        runs[0]["unauthorized"] = True
        runs[0]["blocked_harmful"], runs[1]["blocked_harmful"] = True, False
        card = scorecard(runs, k=3)
        assert card["unauthorized_rate"] == pytest.approx(0.1)
        assert card["harmful_block_rate"] == pytest.approx(0.5)
        assert scorecard(self._runs(), k=3)["harmful_block_rate"] is None

    # crashes are counted by kind and infra crashes excluded only in the side metric
    def test_crashes_by_kind_and_excl_infra(self):
        runs = self._runs()
        runs[9].update(error="x", error_kind="rate_limit")  # task b's one failed run
        card = scorecard(runs, k=1)
        assert card["crashes_by_kind"] == {"rate_limit": 1}
        assert card["resolve_rate"] == pytest.approx(0.9)
        assert card["resolve_rate_excl_infra"] == pytest.approx(9 / 9)

    # invariant violation rate is None until rows carry the field
    def test_invariant_violation_rate(self):
        runs = self._runs()
        assert scorecard(runs)["invariant_violation_rate"] is None
        for r in runs:
            r["invariant_violations"] = []
        runs[0]["invariant_violations"] = ["boom"]
        assert scorecard(runs)["invariant_violation_rate"] == pytest.approx(0.1)

    # optional rates divide by the rows that posed the question
    def test_optional_rates(self):
        runs = self._runs()
        assert scorecard(runs)["reply_grounded_rate"] is None
        runs[0]["looked_up_first"], runs[1]["looked_up_first"] = True, False
        assert scorecard(runs)["lookup_before_refund_rate"] == pytest.approx(0.5)

    # latency tail, cost, cost per resolution, None usd tolerated
    def test_latency_and_cost(self):
        runs = self._runs()
        for i, r in enumerate(runs):
            r["latency_s"] = 1.0 if i < 9 else 90.0
            r["usd"] = 0.0125
        runs[0]["usd"] = None
        card = scorecard(runs, k=3)
        assert card["p50_latency_s"] == 1.0 and card["p95_latency_s"] == 90.0
        assert card["usd_total"] == pytest.approx(0.1125)
        assert card["cost_per_resolution"] == pytest.approx(0.1125 / 9, abs=1e-6)

    # an empty sweep does not crash
    def test_empty(self):
        card = scorecard([], k=3)
        assert card["runs"] == 0 and card["resolve_ci95_clustered"] is None
