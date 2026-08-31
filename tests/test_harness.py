"""Harness invariants, gated in CI — the properties the rest of the system assumes.

Three of these modules already carried assertions in demo(), which only ran when someone typed
`python -m harness.tools`. They're wired here so a regression fails the build instead of waiting
to be noticed. harness/validity.py had NO checks at all, which is the gap worth closing first:
it is the deterministic verifier that gates production AND supplies the RLVR training reward, so
a silent change there moves both at once.

Fixtures are selected from the REAL data/db/orders.json rather than hand-built dicts. A fixture
DB can drift from the shape the rules assume and then the tests pass while production breaks;
picking live records means the suite fails loudly if the DB is ever regenerated incompatibly.

No API key, no network. That is the point of keeping enforcement out of the model.
"""
from datetime import date

import pytest

from config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
from harness import audit
from harness.policy import NOW, Decision
from harness.routing import _team_for, _ticket_id, route
from harness.tools import escalate_to_human, issue_refund, lookup_order
from harness.validity import get_order, verify_claim
from schemas import CustomerClaim

_ORDERS = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]

# The DB stores an order's real outcome as `situation`; a claim of the matching type is TRUE.
# 'canceled' (not 'order_canceled') is the situation string — verified against the live DB.
SITUATION_TO_CLAIM = {
    "late": "late_delivery",
    "never_arrived": "never_arrived",
    "canceled": "order_canceled",
}


def _age(order: dict) -> int:
    return (NOW - date.fromisoformat(order["promised_date"])).days


def _pick(situation: str, in_window: bool | None = None, over_limit: bool = False) -> dict:
    """First real order matching the conditions, or skip — never a vacuous pass on empty data."""
    for o in _ORDERS:
        if o["situation"] != situation:
            continue
        if in_window is True and _age(o) > CLAIM_WINDOW_DAYS:
            continue
        if in_window is False and _age(o) <= CLAIM_WINDOW_DAYS:
            continue
        if over_limit and o["amount_usd"] <= AUTO_APPROVE_MAX_USD:
            continue
        return o
    pytest.skip(f"no order matches situation={situation} in_window={in_window} over_limit={over_limit}")


@pytest.fixture(autouse=True)
def _isolated_order_index(tmp_path, monkeypatch):
    """Redirect audit.ORDER_DIR to a fresh temp dir for every test in this module.

    Several tests below pick a real order via _pick(), which is deterministic — the same order
    every time within a run. Since rule 4 now checks a cross-case order-level index (2026-08-15
    fix), two different tests that happen to pick the same real order would otherwise see each
    other's refunds and fail for a reason that has nothing to do with what they're testing. Same
    pattern tests/test_monitor.py already uses for TELEMETRY.
    """
    monkeypatch.setattr(audit, "ORDER_DIR", tmp_path / "_by_order")


@pytest.fixture
def case(request):
    """A private, empty audit trail per test — cleaned up either side so runs can't bleed."""
    case_id = f"test-{request.node.name}"
    audit.clear(case_id)
    yield case_id
    audit.clear(case_id)


def _allow() -> Decision:
    return Decision(action="allow", rule_id="within_policy", reason="ok")


class TestDatabase:
    def test_db_is_present_and_non_trivial(self):
        assert len(_ORDERS) > 100, "the real order DB should be present and substantial"

    def test_every_situation_has_orders(self):
        situations = {o["situation"] for o in _ORDERS}
        assert {"late", "on_time", "never_arrived", "canceled"} <= situations


class TestValidity:
    """The verifier: production gate AND the RLVR reward. Previously untested entirely."""

    def test_known_order_is_found(self):
        order = _pick("late")
        assert get_order(order["order_id"])["order_id"] == order["order_id"]

    def test_unknown_order_returns_empty_dict(self):
        assert get_order("ORD-999999") == {}

    @pytest.mark.parametrize("situation, claim_type", sorted(SITUATION_TO_CLAIM.items()))
    def test_matching_claim_is_true(self, situation, claim_type):
        """A claim that matches what actually happened must verify as true."""
        order = _pick(situation)
        finding = verify_claim(CustomerClaim(order_id=order["order_id"], claim_type=claim_type))
        assert finding.order_found is True
        assert finding.claim_true is True

    def test_on_time_order_cannot_support_a_late_claim(self):
        """The 'found but false' outcome — without it the verifier could only ever say yes."""
        order = _pick("on_time")
        finding = verify_claim(CustomerClaim(order_id=order["order_id"], claim_type="late_delivery"))
        assert finding.order_found is True
        assert finding.claim_true is False

    def test_unknown_order_is_none_not_false(self):
        """'Can't judge' is a distinct outcome from 'the claim is false' — they drive different
        actions (ask for the number vs reject the claim)."""
        finding = verify_claim(CustomerClaim(order_id="ORD-999999", claim_type="late_delivery"))
        assert finding.claim_true is None and finding.order_found is False

    def test_missing_order_id_is_none_not_false(self):
        finding = verify_claim(CustomerClaim(order_id=None, claim_type="late_delivery"))
        assert finding.claim_true is None and finding.order_found is False

    def test_finding_carries_the_real_amount(self):
        """The harness supplies the number; the model is never asked for it."""
        order = _pick("late")
        finding = verify_claim(CustomerClaim(order_id=order["order_id"], claim_type="late_delivery"))
        assert finding.amount_usd == order["amount_usd"]

    def test_stated_amount_is_never_consulted(self):
        """A customer's figure cannot change the verdict — that's the whole defence."""
        order = _pick("on_time")
        claim = CustomerClaim(order_id=order["order_id"], claim_type="late_delivery",
                              stated_amount_usd=99999.0)
        assert verify_claim(claim).claim_true is False


class TestAuditTrail:
    """Append-only, and the exact record shape policy rule 4 reads."""

    def test_empty_case_reads_as_empty_list(self, case):
        assert audit.read(case) == []

    def test_round_trip_returns_what_was_written(self, case):
        record = audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 12.5},
                              _allow(), True, "refunded $12.50")
        assert audit.read(case) == [record]

    def test_record_carries_the_rule_id_flattened(self, case):
        """rule_id is what you group by when asking what's actually stopping the agent."""
        audit.append(case, "issue_refund", "ORD-1000", {}, _allow(), True, "ok")
        assert audit.read(case)[0]["rule_id"] == "within_policy"

    def test_append_only_never_replaces(self, case):
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 1.0}, _allow(), True, "a")
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 2.0}, _allow(), False, "b")
        trail = audit.read(case)
        assert len(trail) == 2 and trail[0]["args"]["amount_usd"] == 1.0

    def test_failed_attempts_are_recorded_too(self, case):
        """A denied call is the most informative line in the trail — it must not be dropped."""
        denied = Decision(action="deny", rule_id="refund_exceeds_cap", reason="too much")
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 900.0}, denied, False, "Refused")
        row = audit.read(case)[0]
        assert row["ok"] is False and row["action"] == "deny"

    def test_rule_four_read_path_is_intact(self, case):
        """The exact access pattern check_refund() uses. Renaming a key here would become a
        silent double refund rather than a loud failure."""
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 12.5}, _allow(), True, "ok")
        prior = [h for h in audit.read(case)
                 if h["tool"] == "issue_refund" and h["order_id"] == "ORD-1000" and h["ok"]]
        assert prior and prior[0]["args"]["amount_usd"] == 12.5

    def test_clear_removes_the_trail(self, case):
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "ok")
        audit.clear(case)
        assert audit.read(case) == []

    def test_cases_are_isolated_from_each_other(self, case):
        other = f"{case}-other"
        audit.clear(other)
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "ok")
        assert audit.read(other) == []
        audit.clear(other)


class TestTools:
    """tools.py is the only write path, and every mutation goes through policy first."""

    def test_lookup_returns_the_real_amount(self, case):
        """What lets the agent catch an inflated claim: the truth is one tool call away."""
        order = _pick("late", in_window=True)
        assert f"${order['amount_usd']:.2f}" in lookup_order(case, order["order_id"])

    def test_lookup_is_recorded_as_evidence(self, case):
        """Whether the agent LOOKED BEFORE IT ACTED is the most valuable fact about a run."""
        order = _pick("late", in_window=True)
        lookup_order(case, order["order_id"])
        assert audit.read(case)[-1]["rule_id"] == "lookup_hit"

    def test_lookup_miss_is_recorded(self, case):
        assert "not found" in lookup_order(case, "ORD-999999")
        assert audit.read(case)[-1]["rule_id"] == "lookup_miss"

    def test_over_cap_refund_is_refused_as_a_result_not_an_exception(self, case):
        """A refusal the model can read and explain, not a crash that strands the customer."""
        order = _pick("late", in_window=True)
        out = issue_refund(case, order["order_id"], "late_delivery", order["amount_usd"])
        assert out.startswith("Refused:")
        assert audit.read(case)[-1]["rule_id"] == "refund_exceeds_cap"
        assert audit.read(case)[-1]["ok"] is False

    def test_within_cap_refund_succeeds(self, case):
        order = _pick("late", in_window=True)
        cap = round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
        amount = min(cap, AUTO_APPROVE_MAX_USD)
        assert issue_refund(case, order["order_id"], "late_delivery", amount).startswith("Refunded")

    def test_second_refund_is_denied_off_the_trail_alone(self, case):
        """Nothing about the second request is wrong on its face — only the past makes it wrong."""
        order = _pick("late", in_window=True)
        cap = round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
        amount = min(cap, AUTO_APPROVE_MAX_USD)
        issue_refund(case, order["order_id"], "late_delivery", amount)
        again = issue_refund(case, order["order_id"], "late_delivery", amount)
        assert "already refunded" in again.lower()

    def test_non_positive_refund_is_refused(self, case):
        """Rule 0. A negative refund is a charge to the customer wearing a refund's name."""
        order = _pick("late", in_window=True)
        assert issue_refund(case, order["order_id"], "late_delivery", -5.0).startswith("Refused")
        assert audit.read(case)[-1]["rule_id"] == "invalid_amount"

    def test_escalation_is_never_denied(self, case):
        """Gating the escape hatch is how an agent ends up with nowhere to go and improvises."""
        order = _pick("late", in_window=True)
        out = escalate_to_human(case, order["order_id"], "customer asked for a manager")
        assert out.startswith("Escalated")
        assert audit.read(case)[-1]["rule_id"] == "escalation_always_permitted"

    def test_refused_refund_moves_no_money(self, case):
        """The enforcement claim: a denial must leave no successful refund in the trail."""
        order = _pick("late", in_window=True)
        issue_refund(case, order["order_id"], "late_delivery", order["amount_usd"])
        assert not [r for r in audit.read(case) if r["tool"] == "issue_refund" and r["ok"]]


class TestRouting:
    """Pure function of the trail: same trail, same ticket, always."""

    LATE = {"order_id": "ORD-1000", "claim_type": "late_delivery"}
    MISSING = {"order_id": "ORD-1200", "claim_type": "never_arrived"}

    @staticmethod
    def _refund_row(amount=159.86, ok=True, action="allow", rule="within_policy"):
        return {"tool": "issue_refund", "order_id": "ORD-1000", "ok": ok, "action": action,
                "args": {"amount_usd": amount}, "rule_id": rule, "reason": "r"}

    def test_successful_refund_resolves_and_closes(self):
        ticket = route("case-a", [self._refund_row()], self.LATE)
        assert ticket.outcome == "resolved" and ticket.status == "closed" and ticket.team is None
        assert "159.86" in ticket.customer_message

    def test_ticket_id_is_idempotent(self):
        """A retry must not page a second human."""
        trail = [self._refund_row()]
        assert route("case-a", trail, self.LATE).ticket_id == route("case-a", trail, self.LATE).ticket_id

    def test_different_cases_get_different_tickets(self):
        assert _ticket_id("case-a") != _ticket_id("case-b")

    def test_over_limit_escalates_and_stays_open(self):
        trail = [self._refund_row(500.0, ok=False, action="escalate", rule="over_auto_approve_limit")]
        ticket = route("case-b", trail, self.MISSING)
        assert ticket.outcome == "escalated" and ticket.status == "open"

    def test_money_authority_beats_domain(self):
        """An over-limit refund is a billing sign-off whatever the delivery problem was."""
        trail = [self._refund_row(500.0, ok=False, action="escalate", rule="over_auto_approve_limit")]
        assert _team_for(trail, self.MISSING) == "billing"

    def test_never_arrived_routes_to_logistics(self):
        esc = [{"tool": "escalate_to_human", "order_id": "ORD-1200", "ok": True, "action": "allow",
                "args": {"reason": "manager"}, "rule_id": "escalation_always_permitted", "reason": "m"}]
        assert route("case-c", esc, self.MISSING).team == "logistics"

    @pytest.mark.parametrize("claim_type", ["late_delivery", "order_canceled"])
    def test_billing_domains(self, claim_type):
        assert _team_for([], {"claim_type": claim_type}) == "billing"

    def test_unknown_claim_falls_back_to_general(self):
        assert _team_for([], {"order_id": None, "claim_type": None}) == "general"

    def test_denial_carries_the_policy_reason_to_the_customer(self):
        """A denial the customer can't understand is a denial they'll dispute."""
        trail = [self._refund_row(10.0, ok=False, action="deny", rule="outside_claim_window")]
        trail[0]["reason"] = "Order is 200 days old; claims close after 90 days"
        ticket = route("case-d", trail, self.LATE)
        assert ticket.outcome == "denied" and "90 days" in ticket.customer_message

    def test_empty_trail_denies_rather_than_inventing_an_outcome(self):
        ticket = route("case-e", [], self.LATE)
        assert ticket.outcome == "denied" and ticket.status == "closed"

    def test_missing_order_id_still_produces_readable_message(self):
        esc = [{"tool": "escalate_to_human", "order_id": None, "ok": True, "action": "allow",
                "args": {"reason": "x"}, "rule_id": "escalation_always_permitted", "reason": "x"}]
        ticket = route("case-f", esc, {"order_id": None, "claim_type": None})
        assert "your order" in ticket.customer_message and ticket.team == "general"
