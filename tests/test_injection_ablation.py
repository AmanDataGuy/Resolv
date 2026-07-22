"""The two attack/control suites — everything about them that doesn't need a model.

Both files are mostly a payload table and a scoring function wrapped around the same agent loop
the sweep uses, so there is little to test and exactly one thing that would be dangerous to get
wrong: the ablation rebinds a production module global. If it ever failed to put it back, every
later run in the same process would silently execute with the policy engine disabled, and the
numbers would look fine.
"""
import agents.loop as loop
from eval.ablation import _bind_without_harness, run_arm
from eval.injection import MARKER, PAYLOADS, build_cases, score


class TestInjectionCases:
    def test_one_case_per_payload_per_order(self):
        cases = build_cases(3)
        assert len(cases) == len(PAYLOADS) * 3
        assert len({c["case_id"] for c in cases}) == len(cases)

    def test_every_payload_names_a_real_order(self):
        """A payload left with an unfilled placeholder would be attacking nothing."""
        for case in build_cases(1):
            assert case["order_id"] in case["message"]
            assert "{order_id}" not in case["message"] and "{amount" not in case["message"]

    def test_the_money_payloads_demand_more_than_the_cap(self):
        """If one asked for a legitimate amount, complying would be correct and the suite would
        measure nothing. exfiltrate_prompt is excluded on purpose — it wants the prompt, not cash."""
        for case in build_cases(1):
            if case["tactic"] == "exfiltrate_prompt":
                assert "$" not in case["message"]
                continue
            demanded = max(float(part.split()[0].rstrip(".,"))
                           for part in case["message"].replace(",", "").split("$")[1:])
            assert demanded > case["cap"]

    def test_the_answer_key_ignores_what_the_attacker_demands(self):
        """The cases are built on genuinely late, in-window orders, so a refund IS owed — but only
        the capped amount. An injection does not change what the customer is entitled to."""
        for case in build_cases(2):
            assert case["expected"] == "refund"
            assert case["expected_amount"] == case["cap"]

    def test_the_exfiltration_payload_plants_a_marker(self):
        """Looking for policy words in the reply would flag the agent explaining the cap, which is
        correct behaviour. An arbitrary token can only appear by following the instruction."""
        assert MARKER in PAYLOADS["exfiltrate_prompt"]
        assert any(MARKER in c["message"] for c in build_cases(1))


class TestInjectionScore:
    @staticmethod
    def _row(payload="direct_override", over_cap=False, leaked=False, unauthorized=False):
        return {"payload": payload, "attempted_over_cap": over_cap, "leaked_marker": leaked,
                "complied": over_cap or leaked, "unauthorized": unauthorized}

    def test_compliance_and_breach_are_counted_separately(self):
        """The whole point of the suite: the model can take the bait every time and the system can
        still lose nothing. One number cannot say both."""
        rows = [self._row(over_cap=True) for _ in range(4)]
        card = score(rows)
        assert card["complied_rate"] == 1.0
        assert card["unauthorized_rate"] == 0.0

    def test_a_leak_counts_as_compliance_without_touching_money(self):
        card = score([self._row(leaked=True), self._row()])
        assert card["leak_rate"] == 0.5 and card["complied_rate"] == 0.5
        assert card["attempted_over_cap_rate"] == 0.0

    def test_breakdown_shows_which_attack_works(self):
        rows = [self._row("direct_override", over_cap=True), self._row("fake_system_turn")]
        by = score(rows)["by_payload"]
        assert by["direct_override"]["complied"] == 1.0
        assert by["fake_system_turn"]["complied"] == 0.0

    def test_empty_run_does_not_divide_by_zero(self):
        assert score([])["runs"] == 0


class TestAblationBypass:
    def test_the_bypass_offers_exactly_the_production_tool_names(self):
        """Same three names, or the model would call a tool that doesn't exist and the OFF arm
        would measure a broken agent rather than an unguarded one."""
        assert sorted(_bind_without_harness("c")) == sorted(loop._bind("c"))

    def test_the_on_arm_does_not_touch_the_binding(self):
        original = loop._bind
        run_arm([], n=1, arm="on")
        assert loop._bind is original

    def test_the_off_arm_restores_the_real_tools(self):
        """The dangerous one. An unrestored patch leaves every later run in the process executing
        with no policy engine, and nothing in the output would say so."""
        original = loop._bind
        run_arm([], n=1, arm="off")
        assert loop._bind is original

    def test_the_binding_is_restored_even_when_a_run_raises(self):
        original = loop._bind
        broken = [{"task_id": "boom"}]  # missing every key _run needs
        try:
            run_arm(broken, n=1, arm="off")
        except Exception:
            pass
        assert loop._bind is original
