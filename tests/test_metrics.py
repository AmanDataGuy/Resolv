"""The scoring math, gated in CI — because every published number depends on it.

eval/metrics.py already carried these checks in demo(), where they only ran if someone
remembered to type `python -m eval.metrics`. A test that isn't in the gate is a script, not a
test: the whole scorecard could silently start lying and the build would stay green. These are
the same assertions, wired so a regression fails the build.

Every expected value here is hand-worked in study/maths.md. Nothing is captured from a previous
run's output — a test that asserts "whatever it printed last time" only proves the code hasn't
changed, not that it's right.
"""
import pytest

from eval.metrics import mcnemar, pass_hat_k, scorecard, standard_error, trajectory_score, wilson


class TestPassHatK:
    """The unbiased pass^k estimator: C(c,k)/C(n,k)."""

    @pytest.mark.parametrize(
        "n, c, k, expected",
        [
            (5, 4, 3, 0.4),    # C(4,3)/C(5,3) = 4/10 — the doc's worked example
            (5, 5, 3, 1.0),    # every attempt passed
            (5, 2, 3, 0.0),    # fewer successes than draws is a HARD zero
            (5, 0, 3, 0.0),
            (5, 4, 1, 0.8),    # pass^1 is just the success rate
            (10, 7, 1, 0.7),
            (5, 3, 3, 0.1),    # C(3,3)/C(5,3) = 1/10
        ],
    )
    def test_known_values(self, n, c, k, expected):
        assert pass_hat_k(n, c, k) == pytest.approx(expected)

    def test_naive_form_overstates(self):
        """Why the combinatorial form exists at all: (c/n)^k flatters by ~11 points at n=5,c=4."""
        naive = (4 / 5) ** 3
        assert naive == pytest.approx(0.512)
        assert naive > pass_hat_k(5, 4, 3)

    def test_k_greater_than_n_raises(self):
        """You cannot draw more attempts than were run — that's a bug, not a score of 0."""
        with pytest.raises(ValueError):
            pass_hat_k(3, 3, 5)

    def test_reliability_decays_with_steps(self):
        """A 90%-per-attempt agent fails a 10-step task ~65% of the time."""
        assert 0.9**10 == pytest.approx(0.3487, abs=1e-3)


class TestTrajectoryScore:
    """Geometric mean: one catastrophic step must annihilate the trajectory."""

    def test_single_zero_annihilates(self):
        steps = [1.0] * 19 + [0.0]
        assert sum(steps) / len(steps) == 0.95      # arithmetic mean forgives it
        assert trajectory_score(steps) == 0.0        # geometric mean does not

    def test_negative_also_annihilates(self):
        assert trajectory_score([1.0, -0.5, 1.0]) == 0.0

    def test_geometric_mean_of_equal_scores(self):
        assert trajectory_score([0.5, 0.5]) == pytest.approx(0.5)

    def test_all_perfect(self):
        assert trajectory_score([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_empty_is_zero(self):
        assert trajectory_score([]) == 0.0

    def test_long_run_of_small_scores_does_not_underflow(self):
        """The clamp's real job: 20 steps at 0.9 stay in range instead of drifting to a denormal."""
        assert trajectory_score([0.9] * 20) == pytest.approx(0.9)


class TestWilson:
    """Wilson interval — the only correct choice near p=1, which is where safety metrics live."""

    def test_perfect_score_does_not_claim_certainty(self):
        lo, hi = wilson(20, 20)
        assert 0.83 < lo < 0.85, "20/20 must not report a lower bound of 1.0"
        assert hi == 1.0

    def test_zero_score_lower_bound_is_zero(self):
        lo, hi = wilson(0, 20)
        assert lo == 0.0 and 0.0 < hi < 0.2

    def test_interval_contains_point_estimate(self):
        lo, hi = wilson(198, 200)
        assert lo < 198 / 200 < hi

    def test_interval_narrows_with_more_samples(self):
        wide = wilson(9, 10)
        narrow = wilson(900, 1000)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_samples_is_not_a_crash(self):
        assert wilson(0, 0) == (0.0, 0.0)


class TestStandardError:
    """The number that killed the previous fine-tune's headline."""

    def test_the_delta_that_was_noise(self):
        se = standard_error(0.759, 466)
        assert se == pytest.approx(0.0198, abs=1e-3)
        assert (0.775 - 0.759) < se, "the reported +0.016 gain sits inside one SE"

    def test_normal_approximation_fails_at_boundary(self):
        """SE=0 at p=1 is exactly why wilson() is used instead."""
        assert standard_error(1.0, 20) == 0.0

    def test_zero_samples(self):
        assert standard_error(0.5, 0) == 0.0


class TestMcNemar:
    """Paired comparison for two model versions. Ties carry no information and are discarded."""

    def test_lopsided_disagreement_is_significant(self):
        chi2, significant = mcnemar(30, 10)
        assert significant is True and chi2 > 3.841

    def test_balanced_disagreement_is_not(self):
        _, significant = mcnemar(20, 18)
        assert significant is False

    def test_no_disagreement_at_all(self):
        assert mcnemar(0, 0) == (0.0, False)

    def test_symmetric_in_its_arguments(self):
        """Direction of the difference doesn't change the statistic."""
        assert mcnemar(30, 10)[0] == mcnemar(10, 30)[0]


class TestScorecard:
    """The aggregate the whole project reports. 2 tasks x 5 runs: one perfect, one flaky."""

    @staticmethod
    def _runs():
        perfect = [{"task_id": "a", "resolved": True, "steps": 3} for _ in range(5)]
        flaky = [{"task_id": "b", "resolved": i < 4, "steps": 4} for i in range(5)]
        return perfect + flaky

    def test_shape_and_counts(self):
        card = scorecard(self._runs(), k=3)
        assert card["tasks"] == 2 and card["runs"] == 10

    def test_pass_k_averages_per_task_not_pooled(self):
        """Per-task estimates averaged IS the estimator — pooling would weight by run count."""
        card = scorecard(self._runs(), k=3)
        assert card["pass^3"] == pytest.approx((1.0 + 0.4) / 2)
        assert card["pass^1"] == pytest.approx((1.0 + 0.8) / 2)

    def test_safety_metrics_default_clean(self):
        card = scorecard(self._runs(), k=3)
        assert card["unauthorized_rate"] == 0.0
        assert card["over_block_rate"] == 0.0

    def test_unauthorized_is_counted(self):
        runs = self._runs()
        runs[0]["unauthorized"] = True
        assert scorecard(runs, k=3)["unauthorized_rate"] == pytest.approx(0.1)

    def test_harmful_block_uses_only_deny_tasks_as_denominator(self):
        """blocked_harmful is None on tasks that shouldn't be blocked; those must not inflate it."""
        runs = self._runs()
        runs[0]["blocked_harmful"] = True
        runs[1]["blocked_harmful"] = False
        assert scorecard(runs, k=3)["harmful_block_rate"] == pytest.approx(0.5)

    def test_harmful_block_is_none_when_no_deny_tasks(self):
        assert scorecard(self._runs(), k=3)["harmful_block_rate"] is None

    def test_resolve_rate_and_ci(self):
        card = scorecard(self._runs(), k=3)
        assert card["resolve_rate"] == pytest.approx(0.9)
        lo, hi = card["resolve_ci95"]
        assert lo < 0.9 < hi

    def test_mean_steps(self):
        assert scorecard(self._runs(), k=3)["mean_steps"] == pytest.approx(3.5)

    def test_empty_runs_does_not_crash(self):
        card = scorecard([], k=3)
        assert card["runs"] == 0 and card["tasks"] == 0

    def test_deny_everything_scores_half_on_a_balanced_set(self):
        """The floor a benchmark must have: a constant policy cannot beat a coin flip."""
        owed = [{"task_id": f"owed{i}", "resolved": False, "steps": 1} for i in range(10)]
        not_owed = [{"task_id": f"deny{i}", "resolved": True, "steps": 1} for i in range(10)]
        assert scorecard(owed + not_owed, k=1)["resolve_rate"] == pytest.approx(0.5)
