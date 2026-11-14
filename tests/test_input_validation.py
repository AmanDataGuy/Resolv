"""
# tests/test_input_validation.py

**What:** tool-call JSON parsing in resolv/agents/loop.py: NaN / Infinity / non-object arguments are rejected.
**Why:** a non-finite amount must never reach resolv/harness/policy.py as a number; it is refused at parse time.
**Used by:** pytest.
"""
import pytest

from resolv.agents.loop import _parse_args


# Bad tool arguments raise ValueError.
@pytest.mark.parametrize("raw", [
    '{"order_id": "ORD-1000", "claim_type": "late_delivery", "amount_usd": NaN}',
    '{"amount_usd": Infinity}',
    '{"amount_usd": -Infinity}',
    '[1, 2, 3]',
    '"just a string"',
    '{not json',
])
def test_bad_tool_arguments_raise_value_error(raw):
    with pytest.raises(ValueError):
        _parse_args(raw)


# Valid arguments parse.
def test_valid_arguments_parse():
    args = _parse_args('{"order_id": "ORD-1000", "claim_type": "late_delivery", "amount_usd": 12.5}')
    assert args == {"order_id": "ORD-1000", "claim_type": "late_delivery", "amount_usd": 12.5}


# Empty arguments are an empty object.
def test_empty_arguments_are_an_empty_object():
    assert _parse_args("") == {}
