"""The extractor scorecard and its regression gate — the arithmetic, with no model involved.

_one() is the only function here that costs money, and it is deliberately untested: it is a
provider call wrapped in a try. Everything that decides what the numbers MEAN — the split, the
rates, the denominators, the pass/fail verdict — is pure and lives below.

The gate matters more than the scorecard. A scorecard that is wrong reports a bad number once; a
gate that is wrong lets every future bad number through silently.
"""
import json

import pytest

from eval.extractor import TEST_FRACTION, _norm, compare, load_cases, score


class TestSplit:
    def test_held_out_split_matches_the_fine_tunes(self):
        """These constants are duplicated from scripts/train_extractor.py because that script must
        run standalone on Kaggle. This is the assertion that keeps the copy honest — if the split
        drifts, the both-right number stops being comparable to 0.489/0.707 while still looking
        like it is."""
        every = load_cases("all")
        assert len(load_cases("test")) == int(len(every) * TEST_FRACTION)

    def test_the_split_is_deterministic(self):
        assert [c["case_id"] for c in load_cases("test")] == [c["case_id"] for c in load_cases("test")]

    def test_the_test_split_is_a_subset_of_all(self):
        every = {c["case_id"] for c in load_cases("all")}
        assert {c["case_id"] for c in load_cases("test")} <= every

    def test_the_pool_still_contains_abstention_cases(self):
        """Messages with no order number are what hallucination_rate is measured on. If a data
        regeneration ever drops them, the metric silently becomes None instead of failing."""
        assert sum(1 for c in load_cases("all") if not c["ground_truth"]["order_id"]) > 0


class TestNorm:
    @pytest.mark.parametrize("value", [None, "none", "NONE", " none "])
    def test_absence_has_one_spelling(self, value):
        """None and 'none' are the same answer — 'the customer gave no usable number'."""
        assert _norm(value) == "none"

    def test_case_and_whitespace_do_not_change_an_id(self):
        assert _norm(" ORD-1000 ") == _norm("ord-1000") == "ord-1000"


def _row(case_id="c1", order_ok=True, claim_ok=True, hallucinated=None,
         difficulty="easy", truth_claim="late_delivery", got_claim="late_delivery"):
    return {
        "case_id": case_id, "difficulty": difficulty,
        "truth_claim": truth_claim, "got_claim": got_claim,
        "order_ok": order_ok, "claim_ok": claim_ok, "both_ok": order_ok and claim_ok,
        "hallucinated": hallucinated, "latency_s": 1.0, "usd": 0.001,
    }


class TestScore:
    def test_both_requires_both(self):
        rows = [_row("a"), _row("b", claim_ok=False), _row("c", order_ok=False), _row("d")]
        card = score(rows)
        assert card["order_acc"] == 0.75 and card["claim_acc"] == 0.75
        assert card["both_acc"] == 0.5, "getting each field right 75% of the time is not 75% both-right"

    def test_hallucination_divides_by_the_cases_that_offered_the_chance(self):
        """8 of 10 rows gave an order number, so they cannot hallucinate one and must stay out of
        the denominator. Pooling them would report 0.10 for a model that invents on HALF the
        cases where it should abstain."""
        rows = [_row(f"c{i}") for i in range(8)]
        rows += [_row("x", hallucinated=True), _row("y", hallucinated=False)]
        card = score(rows)
        assert card["hallucination_rate"] == 0.5
        assert card["abstention_n"] == 2

    def test_hallucination_is_none_when_no_case_could_abstain(self):
        """None, not 0.0 — 'never tested' must not print as 'perfectly safe'."""
        assert score([_row("a"), _row("b")])["hallucination_rate"] is None

    def test_confidence_interval_brackets_the_estimate(self):
        rows = [_row(f"c{i}", claim_ok=i < 18) for i in range(20)]
        card = score(rows)
        lo, hi = card["both_ci95"]
        assert lo < card["both_acc"] < hi
        assert hi <= 1.0, "Wilson stays in range where the normal approximation does not"

    def test_difficulty_breakdown_keeps_its_denominators(self):
        rows = [_row("a", difficulty="easy"), _row("b", difficulty="hard", claim_ok=False),
                _row("c", difficulty="hard")]
        by = score(rows)["by_difficulty"]
        assert by["easy"] == {"n": 1, "both": 1.0}
        assert by["hard"] == {"n": 2, "both": 0.5}

    def test_confusion_matrix_names_what_it_confused_things_for(self):
        """An overall claim accuracy of 0.67 here is one type perfect and one type half wrong —
        the aggregate hides exactly the thing worth fixing."""
        rows = [_row("a", truth_claim="never_arrived", got_claim="late_delivery", claim_ok=False),
                _row("b", truth_claim="never_arrived", got_claim="never_arrived"),
                _row("c", truth_claim="late_delivery", got_claim="late_delivery")]
        confusion = score(rows)["claim_confusion"]
        assert confusion["never_arrived"] == {"late_delivery": 1, "never_arrived": 1}
        assert confusion["late_delivery"] == {"late_delivery": 1}

    def test_an_empty_run_does_not_crash(self):
        assert score([]) == {"n": 0}

    def test_errors_are_counted_not_dropped(self):
        rows = [_row("a"), {**_row("b", order_ok=False, claim_ok=False), "error": "Timeout"}]
        card = score(rows)
        assert card["errors"] == 1
        assert card["both_acc"] == 0.5, "a crashed extraction is a wrong answer, not a missing one"


def _baseline(rows):
    return {"model": "test", "split": "test", "score": score(rows), "rows": rows}


class TestRegressionGate:
    """The pass/fail verdict. Both of its failure conditions, and both of its non-failures."""

    def test_an_identical_run_passes(self):
        rows = [_row(f"c{i}") for i in range(10)]
        verdict = compare(rows, _baseline(rows))
        assert verdict["failed"] is False
        assert verdict["regressed"] == 0 and verdict["improved"] == 0

    def test_a_significant_accuracy_drop_fails(self):
        before = [_row(f"c{i}") for i in range(40)]
        after = [_row(f"c{i}", claim_ok=i >= 20) for i in range(40)]  # 20 cases lost
        verdict = compare(after, _baseline(before))
        assert verdict["regressed"] == 20 and verdict["improved"] == 0
        assert verdict["significant"] is True and verdict["failed"] is True
        assert verdict["both_acc_delta"] == -0.5

    def test_a_significant_improvement_does_not_fail_the_build(self):
        """The direction check. A gate that fails on any significant change punishes progress."""
        before = [_row(f"c{i}", claim_ok=i >= 20) for i in range(40)]
        after = [_row(f"c{i}") for i in range(40)]
        verdict = compare(after, _baseline(before))
        assert verdict["significant"] is True and verdict["improved"] == 20
        assert verdict["failed"] is False

    def test_noise_sized_movement_does_not_fail(self):
        """Two cases lost and two gained is churn, not a regression. Without the paired test this
        would be indistinguishable from the real drop above."""
        before = [_row(f"c{i}", claim_ok=i >= 2) for i in range(40)]
        after = [_row(f"c{i}", claim_ok=i < 38) for i in range(40)]
        verdict = compare(after, _baseline(before))
        assert verdict["failed"] is False

    def test_any_rise_in_hallucination_fails_even_when_accuracy_holds(self):
        """A safety regression does not get to hide behind a small sample. Accuracy is untouched
        here — only the invented order numbers moved, and that alone fails the build."""
        before = [_row("a", hallucinated=False), _row("b", hallucinated=False)]
        after = [_row("a", hallucinated=False), _row("b", hallucinated=True)]
        verdict = compare(after, _baseline(before))
        assert verdict["both_acc_delta"] == 0.0
        assert verdict["significant"] is False
        assert verdict["hallucination_delta"] == 0.5 and verdict["failed"] is True

    def test_a_fall_in_hallucination_is_not_a_regression(self):
        before = [_row("a", hallucinated=True), _row("b", hallucinated=True)]
        after = [_row("a", hallucinated=False), _row("b", hallucinated=True)]
        assert compare(after, _baseline(before))["failed"] is False

    def test_only_cases_present_in_both_runs_are_paired(self):
        """Adding cases to the pool must not register as improvements — McNemar compares the same
        messages twice, and an unpaired count would make a bigger dataset look like a better model."""
        before = [_row("a"), _row("b")]
        after = [_row("a"), _row("b"), _row("new")]
        assert compare(after, _baseline(before))["paired_cases"] == 2

    def test_the_baseline_file_round_trips_through_json(self):
        """The gate reads a file, not an in-memory dict. Tuples become lists on the way through,
        so the comparison must not depend on their type."""
        rows = [_row(f"c{i}") for i in range(5)]
        restored = json.loads(json.dumps(_baseline(rows)))
        assert compare(rows, restored)["failed"] is False
