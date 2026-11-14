"""
# tests/test_harness.py

**What:** harness invariants: the verifier, the audit trail and its keyed chain, the tools, and routing.
**Why:** the properties the rest of the system assumes fail CI here instead of waiting for someone to run a demo.
Orders come from the real data/db/orders.json, so the suite fails loudly if the DB is regenerated incompatibly.
No API key, no network; tests/conftest.py gives every test its own empty ledger.
**Used by:** pytest.
"""
from datetime import date

import pytest

from resolv.config import AUTO_APPROVE_MAX_USD, CLAIM_WINDOW_DAYS, REFUND_CAP_FRACTION
from resolv.harness import audit, ledger
from resolv.harness.policy import NOW, Decision
from resolv.harness.routing import _team_for, _ticket_id, route
from resolv.harness.tools import escalate_to_human, issue_refund, lookup_order
from resolv.harness.validity import get_order, verify_claim
from resolv.schemas import CustomerClaim

_ORDERS = [o for o in (get_order(f"ORD-{i}") for i in range(1000, 1460)) if o]

# The DB stores an order's real outcome as `situation`; a claim of the matching type is TRUE.
SITUATION_TO_CLAIM = {"late": "late_delivery", "never_arrived": "never_arrived", "canceled": "order_canceled"}


# Days between the order's promised date and the demo clock.
def _age(order: dict) -> int:
    return (NOW - date.fromisoformat(order["promised_date"])).days


# First real order matching the conditions, or skip: never a vacuous pass on empty data.
def _pick(situation: str, in_window: bool | None = None, over_limit: bool = False) -> dict:
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


# A private case id per test; tests/conftest.py already gives each test an empty ledger.
@pytest.fixture
def case(request):
    return f"test-{request.node.name}"


# A plain allow decision for hand-written records.
def _allow() -> Decision:
    return Decision(action="allow", rule_id="within_policy", reason="ok")


# Overwrite one column of one stored audit row, the way someone with disk access would.
def _tamper(seq: int, column: str, value) -> None:
    conn = ledger.connect()
    conn.execute(f"UPDATE audit SET {column}=? WHERE seq=?", (value, seq))
    conn.close()


# The real order DB is present and covers every situation.
class TestDatabase:
    # More than a hundred orders are loaded.
    def test_db_is_present_and_non_trivial(self):
        assert len(_ORDERS) > 100

    # Every situation the claim types map to exists.
    def test_every_situation_has_orders(self):
        assert {"late", "on_time", "never_arrived", "canceled"} <= {o["situation"] for o in _ORDERS}


# The verifier: the production gate and the extractor's reward.
class TestValidity:
    # A known order is found by id.
    def test_known_order_is_found(self):
        order = _pick("late")
        assert get_order(order["order_id"])["order_id"] == order["order_id"]

    # An unknown order is an empty dict, not an error.
    def test_unknown_order_returns_empty_dict(self):
        assert get_order("ORD-999999") == {}

    # A claim that matches what happened verifies as true.
    @pytest.mark.parametrize("situation, claim_type", sorted(SITUATION_TO_CLAIM.items()))
    def test_matching_claim_is_true(self, situation, claim_type):
        order = _pick(situation)
        finding = verify_claim(CustomerClaim(order_id=order["order_id"], claim_type=claim_type))
        assert finding.order_found is True and finding.claim_true is True

    # Found but false: without it the verifier could only ever say yes.
    def test_on_time_order_cannot_support_a_late_claim(self):
        order = _pick("on_time")
        finding = verify_claim(CustomerClaim(order_id=order["order_id"], claim_type="late_delivery"))
        assert finding.order_found is True and finding.claim_true is False

    # "Can't judge" (None) is distinct from "false".
    def test_unknown_order_is_none_not_false(self):
        finding = verify_claim(CustomerClaim(order_id="ORD-999999", claim_type="late_delivery"))
        assert finding.claim_true is None and finding.order_found is False

    # No order id is "can't judge" too.
    def test_missing_order_id_is_none_not_false(self):
        finding = verify_claim(CustomerClaim(order_id=None, claim_type="late_delivery"))
        assert finding.claim_true is None and finding.order_found is False

    # A message with no stated problem must not crash and is "can't judge".
    def test_missing_claim_type_does_not_crash_and_is_none_not_false(self):
        order = _pick("late")
        finding = verify_claim(CustomerClaim(order_id=order["order_id"], claim_type=None))
        assert finding.order_found is True and finding.claim_true is None
        assert finding.amount_usd == order["amount_usd"]

    # The harness supplies the number; the model is never asked for it.
    def test_finding_carries_the_real_amount(self):
        order = _pick("late")
        finding = verify_claim(CustomerClaim(order_id=order["order_id"], claim_type="late_delivery"))
        assert finding.amount_usd == order["amount_usd"]

    # A customer's figure cannot change the verdict.
    def test_stated_amount_is_never_consulted(self):
        order = _pick("on_time")
        claim = CustomerClaim(order_id=order["order_id"], claim_type="late_delivery", stated_amount_usd=99999.0)
        assert verify_claim(claim).claim_true is False


# Append-only, keyed-chained, and the exact record shape policy rule 5 reads.
class TestAuditTrail:
    # A case with no records reads as [].
    def test_empty_case_reads_as_empty_list(self, case):
        assert audit.read(case) == []

    # What append returns is exactly what read returns.
    def test_round_trip_returns_what_was_written(self, case):
        record = audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 12.5}, _allow(), True, "refunded $12.50")
        assert audit.read(case) == [record]

    # rule_id is flattened onto the record, ready to group by.
    def test_record_carries_the_rule_id_flattened(self, case):
        audit.append(case, "issue_refund", "ORD-1000", {}, _allow(), True, "ok")
        assert audit.read(case)[0]["rule_id"] == "within_policy"

    # A second append adds, never replaces.
    def test_append_only_never_replaces(self, case):
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 1.0}, _allow(), True, "a")
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 2.0}, _allow(), False, "b")
        trail = audit.read(case)
        assert len(trail) == 2 and trail[0]["args"]["amount_usd"] == 1.0

    # A denied call is recorded too.
    def test_failed_attempts_are_recorded_too(self, case):
        denied = Decision(action="deny", rule_id="refund_exceeds_cap", reason="too much")
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 900.0}, denied, False, "Refused")
        row = audit.read(case)[0]
        assert row["ok"] is False and row["action"] == "deny"

    # The access pattern check_refund rule 5 uses still works.
    def test_rule_five_read_path_is_intact(self, case):
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 12.5}, _allow(), True, "ok")
        prior = [h for h in audit.read(case) if h["tool"] == "issue_refund" and h["order_id"] == "ORD-1000" and h["ok"]]
        assert prior and prior[0]["args"]["amount_usd"] == 12.5

    # clear empties the trail (allowed because RESOLV_ENV=test).
    def test_clear_removes_the_trail(self, case):
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "ok")
        audit.clear(case)
        assert audit.read(case) == []

    # One case's records never show up in another's trail.
    def test_cases_are_isolated_from_each_other(self, case):
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "ok")
        assert audit.read(f"{case}-other") == []

    # The first record in the ledger chains to genesis.
    def test_first_record_chains_to_genesis(self, case):
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "ok")
        assert audit.read(case)[0]["prev_mac"] == audit.GENESIS_HASH

    # Each record's prev_mac is the mac of the record before it.
    def test_each_record_chains_to_the_one_before_it(self, case):
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "a")
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "b")
        trail = audit.read(case)
        assert trail[1]["prev_mac"] == trail[0]["mac"]

    # Every record gets its own id.
    def test_record_ids_are_unique(self, case):
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "a")
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "b")
        trail = audit.read(case)
        assert trail[0]["record_id"] != trail[1]["record_id"]

    # An untouched chain verifies.
    def test_verify_chain_passes_on_an_untouched_trail(self, case):
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "a")
        audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 5.0}, _allow(), True, "b")
        assert audit.verify_chain(case) == (True, None)

    # No records, or one, is an intact chain.
    def test_verify_chain_passes_on_an_empty_or_single_record_trail(self, case):
        assert audit.verify_chain(case) == (True, None)
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "a")
        assert audit.verify_chain(case) == (True, None)

    # Flipping a real refund's ok after the fact fails that record's own mac.
    def test_verify_chain_catches_an_edited_record(self, case):
        first = audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 900.0}, _allow(), True, "a")
        audit.append(case, "lookup_order", "ORD-1000", {}, _allow(), True, "b")
        _tamper(first["seq"], "ok", 0)
        assert audit.verify_chain(case) == (False, first["seq"])

    # Unlike the old unkeyed per-file chain, editing the LAST record is caught too: each record carries its own mac.
    def test_verify_chain_catches_tampering_with_the_last_record(self, case):
        only = audit.append(case, "issue_refund", "ORD-1000", {"amount_usd": 900.0}, _allow(), True, "a")
        _tamper(only["seq"], "ok", 0)
        assert audit.verify_chain(case) == (False, only["seq"])


# tools.py is the only write path, and every mutation goes through policy first.
class TestTools:
    # The truth is one tool call away for the order's owner.
    def test_lookup_returns_the_real_amount(self, case):
        order = _pick("late", in_window=True)
        assert f"${order['amount_usd']:.2f}" in lookup_order(case, order["order_id"], order["customer_id"])

    # Whether the agent looked before it acted is recorded.
    def test_lookup_is_recorded_as_evidence(self, case):
        order = _pick("late", in_window=True)
        lookup_order(case, order["order_id"], order["customer_id"])
        assert audit.read(case)[-1]["rule_id"] == "lookup_hit"

    # A lookup of an unknown order is recorded as a miss.
    def test_lookup_miss_is_recorded(self, case):
        assert "not found" in lookup_order(case, "ORD-999999", "anyone")
        assert audit.read(case)[-1]["rule_id"] == "lookup_miss"

    # An over-cap refund is a recorded refusal string, not an exception.
    def test_over_cap_refund_is_refused_as_a_result_not_an_exception(self, case):
        order = _pick("late", in_window=True)
        out = issue_refund(case, order["order_id"], "late_delivery", order["amount_usd"], order["customer_id"])
        assert out.startswith("Refused:")
        assert audit.read(case)[-1]["rule_id"] == "refund_exceeds_cap" and audit.read(case)[-1]["ok"] is False

    # A within-cap refund from the owner is paid.
    def test_within_cap_refund_succeeds(self, case):
        order = _pick("late", in_window=True)
        amount = min(round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2), AUTO_APPROVE_MAX_USD)
        assert issue_refund(case, order["order_id"], "late_delivery", amount, order["customer_id"]).startswith("Refunded")

    # The same request a second time is denied off the ledger alone.
    def test_second_refund_is_denied_off_the_trail_alone(self, case):
        order = _pick("late", in_window=True)
        amount = min(round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2), AUTO_APPROVE_MAX_USD)
        issue_refund(case, order["order_id"], "late_delivery", amount, order["customer_id"])
        again = issue_refund(case, order["order_id"], "late_delivery", amount, order["customer_id"])
        assert "already refunded" in again.lower()

    # Rule 0: a negative refund is a charge wearing a refund's name.
    def test_non_positive_refund_is_refused(self, case):
        order = _pick("late", in_window=True)
        assert issue_refund(case, order["order_id"], "late_delivery", -5.0, order["customer_id"]).startswith("Refused")
        assert audit.read(case)[-1]["rule_id"] == "invalid_amount"

    # Rule 2: a true claim on a real order is refused for a caller who does not own it.
    def test_caller_who_does_not_own_the_order_is_refused(self, case):
        order = _pick("late", in_window=True)
        cap = round(order["amount_usd"] * REFUND_CAP_FRACTION["late_delivery"], 2)
        assert issue_refund(case, order["order_id"], "late_delivery", cap, "not_the_owner").startswith("Refused:")
        assert audit.read(case)[-1]["rule_id"] == "caller_not_order_owner"

    # Escalation is never denied.
    def test_escalation_is_never_denied(self, case):
        order = _pick("late", in_window=True)
        assert escalate_to_human(case, order["order_id"], "customer asked for a manager").startswith("Escalated")
        assert audit.read(case)[-1]["rule_id"] == "escalation_always_permitted"

    # A rule 7 escalate reaches a human without a separate escalate_to_human call.
    def test_over_limit_refund_auto_escalates_without_a_separate_tool_call(self, case):
        big = _pick("never_arrived", in_window=True, over_limit=True)
        out = issue_refund(case, big["order_id"], "never_arrived", AUTO_APPROVE_MAX_USD + 0.01, big["customer_id"])
        assert "sent to a human" in out.lower()
        trail = audit.read(case)
        assert trail[-1]["tool"] == "escalate_to_human" and trail[-1]["rule_id"] == "escalation_always_permitted"
        assert trail[-2]["rule_id"] == "over_auto_approve_limit"

    # The same over-limit call twice files one escalation (one pending per action hash).
    def test_auto_escalation_is_not_duplicated_if_already_escalated(self, case):
        big = _pick("never_arrived", in_window=True, over_limit=True)
        amount = AUTO_APPROVE_MAX_USD + 0.01
        issue_refund(case, big["order_id"], "never_arrived", amount, big["customer_id"])
        issue_refund(case, big["order_id"], "never_arrived", amount, big["customer_id"])
        assert len([r for r in audit.read(case) if r["tool"] == "escalate_to_human"]) == 1

    # A denial leaves no successful refund behind.
    def test_refused_refund_moves_no_money(self, case):
        order = _pick("late", in_window=True)
        issue_refund(case, order["order_id"], "late_delivery", order["amount_usd"], order["customer_id"])
        assert not [r for r in audit.read(case) if r["tool"] == "issue_refund" and r["ok"]]
        assert ledger.refunds() == []


# Pure function of the trail: same trail, same ticket, always.
class TestRouting:
    LATE = {"order_id": "ORD-1000", "claim_type": "late_delivery"}
    MISSING = {"order_id": "ORD-1200", "claim_type": "never_arrived"}

    # A hand-built issue_refund record.
    @staticmethod
    def _refund_row(amount=159.86, ok=True, action="allow", rule="within_policy"):
        return {"tool": "issue_refund", "order_id": "ORD-1000", "ok": ok, "action": action,
                "args": {"amount_usd": amount}, "rule_id": rule, "reason": "r"}

    # A paid refund resolves and closes the ticket.
    def test_successful_refund_resolves_and_closes(self):
        ticket = route("case-a", [self._refund_row()], self.LATE)
        assert ticket.outcome == "resolved" and ticket.status == "closed" and ticket.team is None
        assert "159.86" in ticket.customer_message

    # A retry must not page a second human.
    def test_ticket_id_is_idempotent(self):
        trail = [self._refund_row()]
        assert route("case-a", trail, self.LATE).ticket_id == route("case-a", trail, self.LATE).ticket_id

    # Different cases get different tickets.
    def test_different_cases_get_different_tickets(self):
        assert _ticket_id("case-a") != _ticket_id("case-b")

    # An over-limit refund escalates and stays open.
    def test_over_limit_escalates_and_stays_open(self):
        trail = [self._refund_row(500.0, ok=False, action="escalate", rule="over_auto_approve_limit")]
        ticket = route("case-b", trail, self.MISSING)
        assert ticket.outcome == "escalated" and ticket.status == "open"

    # An over-limit refund is a billing sign-off whatever the delivery problem was.
    def test_money_authority_beats_domain(self):
        trail = [self._refund_row(500.0, ok=False, action="escalate", rule="over_auto_approve_limit")]
        assert _team_for(trail, self.MISSING) == "billing"

    # A refund with an unconfirmed payment stays open on billing.
    def test_unknown_payment_escalates_to_billing(self):
        trail = [self._refund_row(10.0, ok=False, action="allow", rule="payment_unknown")]
        ticket = route("case-u", trail, self.MISSING)
        assert ticket.outcome == "escalated" and ticket.team == "billing"

    # A missing package is logistics' to chase.
    def test_never_arrived_routes_to_logistics(self):
        esc = [{"tool": "escalate_to_human", "order_id": "ORD-1200", "ok": True, "action": "allow",
                "args": {"reason": "manager"}, "rule_id": "escalation_always_permitted", "reason": "m"}]
        assert route("case-c", esc, self.MISSING).team == "logistics"

    # Late and canceled claims route to billing.
    @pytest.mark.parametrize("claim_type", ["late_delivery", "order_canceled"])
    def test_billing_domains(self, claim_type):
        assert _team_for([], {"claim_type": claim_type}) == "billing"

    # No claim type falls back to general.
    def test_unknown_claim_falls_back_to_general(self):
        assert _team_for([], {"order_id": None, "claim_type": None}) == "general"

    # The policy's reason reaches the customer.
    def test_denial_carries_the_policy_reason_to_the_customer(self):
        trail = [self._refund_row(10.0, ok=False, action="deny", rule="outside_claim_window")]
        trail[0]["reason"] = "Order is 200 days old; claims close after 90 days"
        ticket = route("case-d", trail, self.LATE)
        assert ticket.outcome == "denied" and "90 days" in ticket.customer_message

    # An empty trail is a denial, not an invented outcome.
    def test_empty_trail_denies_rather_than_inventing_an_outcome(self):
        ticket = route("case-e", [], self.LATE)
        assert ticket.outcome == "denied" and ticket.status == "closed"

    # A case with no order id still produces a readable message.
    def test_missing_order_id_still_produces_readable_message(self):
        esc = [{"tool": "escalate_to_human", "order_id": None, "ok": True, "action": "allow",
                "args": {"reason": "x"}, "rule_id": "escalation_always_permitted", "reason": "x"}]
        ticket = route("case-f", esc, {"order_id": None, "claim_type": None})
        assert "your order" in ticket.customer_message and ticket.team == "general"
