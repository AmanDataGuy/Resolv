"""
# tests/test_pii.py

**What:** resolv/harness/pii.py masks emails, phones, cards and simple street addresses, and never touches order ids or money.
**Why:** masked text is what reaches traces; a redacted order id would break the product.
**Used by:** pytest.
"""
import pytest

from resolv.harness import pii


# Each kind is masked.
@pytest.mark.parametrize("text,kind", [
    ("write to jane.doe@example.com please", "EMAIL"),
    ("call me on 555-123-4567", "PHONE"),
    ("call me on (555) 123-4567", "PHONE"),
    ("call me on +1 555 123 4567", "PHONE"),
    ("card 4111 1111 1111 1111", "CARD"),
    ("card 4111-1111-1111-1111", "CARD"),
    ("ship it to 221 Baker Street", "ADDRESS"),
])
def test_each_kind_is_masked(text, kind):
    assert f"[{kind}]" in pii.mask(text)
    assert kind in pii.find(text)


# Order ids, amounts and dates are untouched.
@pytest.mark.parametrize("text", [
    "Order ORD-1181 cost $603.52 and was promised 2018-08-17.",
    "I paid $1,234.56 for ORD-1000.",
    "It took 3 days on the road.",
])
def test_order_ids_amounts_and_dates_untouched(text):
    assert pii.mask(text) == text
    assert pii.find(text) == []


# Card is not half masked as phone.
def test_card_is_not_half_masked_as_phone():
    assert pii.mask("4111111111111111") == "[CARD]"


# mask_any walks nested data.
def test_mask_any_walks_nested_data():
    data = {"messages": [{"content": "mail a@b.co"}], "n": 3}
    assert pii.mask_any(data) == {"messages": [{"content": "mail [EMAIL]"}], "n": 3}
