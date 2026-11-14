"""
# tests/test_extractor_eval.py

**What:** the extractor scorecard and its regression gate (evals/extractor.py): the split, the rates, the
denominators and the pass/fail verdict, with no model involved.
**Why:** `_one()` is the only part that costs money and is a provider call in a try; everything that decides what
the numbers mean is pure and tested here. The gate matters most: a wrong scorecard misreports once, a wrong gate
lets every later regression through.
**Used by:** pytest.
"""
import json

import pytest

from evals.extractor import TEST_FRACTION, _norm, compare, load_cases, score


# The held-out split: size, determinism, and that it keeps abstention cases.
class TestSplit:
    # The split constants are copied from scripts/train_extractor.py (Kaggle-standalone); this keeps the copy honest.
    def test_held_out_split_matches_the_fine_tunes(self):
        every = load_cases("all")
        assert len(load_cases("test")) == int(len(every) * TEST_FRACTION)

    # Same split on every call.
    def test_the_split_is_deterministic(self):
        assert [c["case_id"] for c in load_cases("test")] == [c["case_id"] for c in load_cases("test")]

    # The test split is drawn from the full pool.
    def test_the_test_split_is_a_subset_of_all(self):
        every = {c["case_id"] for c in load_cases("all")}
        assert {c["case_id"] for c in load_cases("test")} <= every

    # No-order-number messages must stay in the pool, or hallucination_rate silently becomes None.
    def test_the_pool_still_contains_abstention_cases(self):
        assert sum(1 for c in load_cases("all") if not c["ground_truth"]["order_id"]) > 0


# Order-id normalisation.
class TestNorm:
    # None and "none" are the same answer: no usable number.
    @pytest.mark.parametrize("value", [None, "none", "NONE", " none "])
    def test_absence_has_one_spelling(self, value):
        assert _norm(value) == "none"

    # Case and whitespace never change an id.
    def test_case_and_whitespace_do_not_change_an_id(self):
        assert _norm(" ORD-1000 ") == _norm("ord-1000") == "ord-1000"


# One scored extractor row with sensible defaults.
def _row(case_id="c1", order_ok=True, claim_ok=True, hallucinated=None,
         difficulty="easy", truth_claim="late_delivery", got_claim="late_delivery"):
    return {
        "case_id": case_id, "difficulty": difficulty,
        "truth_claim": truth_claim, "got_claim": got_claim,
        "order_ok": order_ok, "claim_ok": claim_ok, "both_ok": order_ok and claim_ok,
        "hallucinated": hallucinated, "latency_s": 1.0, "usd": 0.001,
    }


# The scorecard arithmetic.
class TestScore:
    # Both-right needs both fields right on the same case.
    def test_both_requires_both(self):
        rows = [_row("a"), _row("b", claim_ok=False), _row("c", order_ok=False), _row("d")]
        card = score(rows)
        assert card["order_acc"] == 0.75 and card["claim_acc"] == 0.75
        assert card["both_acc"] == 0.5, "getting each field right 75% of the time is not 75% both-right"

    # Hallucination is measured only over cases that could abstain (no order number given).
    def test_hallucination_divides_by_the_cases_that_offered_the_chance(self):
        rows = [_row(f"c{i}") for i in range(8)]
        rows += [_row("x", hallucinated=True), _row("y", hallucinated=False)]
        card = score(rows)
        assert card["hallucination_rate"] == 0.5
        assert card["abstention_n"] == 2

    # "Never tested" is None, not a perfect 0.0.
    def test_hallucination_is_none_when_no_case_could_abstain(self):
        assert score([_row("a"), _row("b")])["hallucination_rate"] is None

    # The Wilson CI brackets the estimate and stays inside [0, 1].
    def test_confidence_interval_brackets_the_estimate(self):
        rows = [_row(f"c{i}", claim_ok=i < 18) for i in range(20)]
        card = score(rows)
        lo, hi = card["both_ci95"]
        assert lo < card["both_acc"] < hi
        assert hi <= 1.0, "Wilson stays in range where the normal approximation does not"

    # Per-difficulty rows keep their own denominators.
    def test_difficulty_breakdown_keeps_its_denominators(self):
        rows = [_row("a", difficulty="easy"), _row("b", difficulty="hard", claim_ok=False),
                _row("c", difficulty="hard")]
        by = score(rows)["by_difficulty"]
        assert by["easy"] == {"n": 1, "both": 1.0}
        assert by["hard"] == {"n": 2, "both": 0.5}

    # The confusion matrix names what each claim type was mistaken for.
    def test_confusion_matrix_names_what_it_confused_things_for(self):
        rows = [_row("a", truth_claim="never_arrived", got_claim="late_delivery", claim_ok=False),
                _row("b", truth_claim="never_arrived", got_claim="never_arrived"),
                _row("c", truth_claim="late_delivery", got_claim="late_delivery")]
        confusion = score(rows)["claim_confusion"]
        assert confusion["never_arrived"] == {"late_delivery": 1, "never_arrived": 1}
        assert confusion["late_delivery"] == {"late_delivery": 1}

    # An empty run returns a card instead of crashing.
    def test_an_empty_run_does_not_crash(self):
        assert score([]) == {"n": 0}

    # A crashed extraction counts as wrong, not missing.
    def test_errors_are_counted_not_dropped(self):
        rows = [_row("a"), {**_row("b", order_ok=False, claim_ok=False), "error": "Timeout"}]
        card = score(rows)
        assert card["errors"] == 1
        assert card["both_acc"] == 0.5, "a crashed extraction is a wrong answer, not a missing one"


# A baseline file's content for these rows.
def _baseline(rows):
    return {"model": "test", "split": "test", "score": score(rows), "rows": rows}


# The pass/fail verdict: both failure conditions and both non-failures.
class TestRegressionGate:
    # Identical runs pass with no movement.
    def test_an_identical_run_passes(self):
        rows = [_row(f"c{i}") for i in range(10)]
        verdict = compare(rows, _baseline(rows))
        assert verdict["failed"] is False
        assert verdict["regressed"] == 0 and verdict["improved"] == 0

    # A significant accuracy drop fails.
    def test_a_significant_accuracy_drop_fails(self):
        before = [_row(f"c{i}") for i in range(40)]
        after = [_row(f"c{i}", claim_ok=i >= 20) for i in range(40)]  # 20 cases lost
        verdict = compare(after, _baseline(before))
        assert verdict["regressed"] == 20 and verdict["improved"] == 0
        assert verdict["significant"] is True and verdict["failed"] is True
        assert verdict["both_acc_delta"] == -0.5

    # A significant improvement never fails the build.
    def test_a_significant_improvement_does_not_fail_the_build(self):
        before = [_row(f"c{i}", claim_ok=i >= 20) for i in range(40)]
        after = [_row(f"c{i}") for i in range(40)]
        verdict = compare(after, _baseline(before))
        assert verdict["significant"] is True and verdict["improved"] == 20
        assert verdict["failed"] is False

    # Two lost and two gained is churn, not a regression (the paired test tells them apart).
    def test_noise_sized_movement_does_not_fail(self):
        before = [_row(f"c{i}", claim_ok=i >= 2) for i in range(40)]
        after = [_row(f"c{i}", claim_ok=i < 38) for i in range(40)]
        verdict = compare(after, _baseline(before))
        assert verdict["failed"] is False

    # Any rise in hallucination fails, even with accuracy unchanged and no significance.
    def test_any_rise_in_hallucination_fails_even_when_accuracy_holds(self):
        before = [_row("a", hallucinated=False), _row("b", hallucinated=False)]
        after = [_row("a", hallucinated=False), _row("b", hallucinated=True)]
        verdict = compare(after, _baseline(before))
        assert verdict["both_acc_delta"] == 0.0
        assert verdict["significant"] is False
        assert verdict["hallucination_delta"] == 0.5 and verdict["failed"] is True

    # A fall in hallucination is not a regression.
    def test_a_fall_in_hallucination_is_not_a_regression(self):
        before = [_row("a", hallucinated=True), _row("b", hallucinated=True)]
        after = [_row("a", hallucinated=False), _row("b", hallucinated=True)]
        assert compare(after, _baseline(before))["failed"] is False

    # Only cases in both runs are paired; new cases are not "improvements".
    def test_only_cases_present_in_both_runs_are_paired(self):
        before = [_row("a"), _row("b")]
        after = [_row("a"), _row("b"), _row("new")]
        assert compare(after, _baseline(before))["paired_cases"] == 2

    # The gate reads JSON, where tuples become lists; the verdict must not care.
    def test_the_baseline_file_round_trips_through_json(self):
        rows = [_row(f"c{i}") for i in range(5)]
        restored = json.loads(json.dumps(_baseline(rows)))
        assert compare(rows, restored)["failed"] is False
