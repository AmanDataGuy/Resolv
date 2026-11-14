"""
# tests/test_serve_bench.py

**What:** keyless checks for inference/serve_bench.py: parsers, the regex arm, scoring rows, percentiles.
**Why:** these functions define the numbers in every serving result; a wrong percentile or a
miscounted TTFT would make every table quietly wrong.
**Used by:** pytest (no network, no server, no GPU).
"""
from inference import serve_bench as sb

CASE = {"case_id": "C-0001", "difficulty": "easy", "message": "x",
        "ground_truth": {"order_id": "ORD-1234", "claim_type": "never_arrived"}}
NULL_CASE = {"case_id": "C-0002", "difficulty": "hard", "message": "x",
             "ground_truth": {"order_id": None, "claim_type": "late_delivery"}}


# Line parser: order normalised, "none" kept, unknown type empty.
def test_parse_line():
    assert sb.parse_line("order=ORD-1234 type=never_arrived") == ("ord-1234", "never_arrived")
    assert sb.parse_line("order=none type=garbage") == ("none", "")


# JSON parser: valid CustomerClaim parsed, null order normalised, invalid raises ValueError.
def test_parse_json():
    assert sb.parse_json('{"order_id": "ORD-1234", "claim_type": "late_delivery"}') == ("ord-1234", "late_delivery")
    assert sb.parse_json('{"order_id": null, "claim_type": null}') == ("none", "")
    for bad in ("not json", '{"order_id": "ORD-1", "claim_type": "refund_please"}'):
        try:
            sb.parse_json(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


# Regex arm: numerals, spelled-out digits, abstention, and claim keywords.
def test_regex_extract():
    assert sb.regex_extract("my order ORD-1234 never arrived") == "order=ORD-1234 type=never_arrived"
    assert sb.regex_extract("order number one zero zero seven was late") == "order=ORD-1007 type=late_delivery"
    assert sb.regex_extract("you cancelled it and I have no number") == "order=none type=order_canceled"


# Invalid JSON output is a counted error row, never a silent default.
def test_row_invalid_json_is_error():
    row = sb._row(CASE, "{oops", 0.1, 0.05, 3, None, "json")
    assert row["error"].startswith("invalid_output") and not row["both_ok"]


# Hallucination is only defined on null-order cases.
def test_row_hallucination():
    assert sb._row(NULL_CASE, "order=ORD-9999 type=late_delivery", 0.1, None, 0, None)["hallucinated"] is True
    assert sb._row(CASE, "order=ORD-1234 type=never_arrived", 0.1, None, 0, None)["hallucinated"] is None


# Nearest-rank percentile, None-safe.
def test_pct_nearest_rank():
    values = list(range(1, 101))
    assert sb.pct(values, 0.5) == 50 and sb.pct(values, 0.99) == 99 and sb.pct(values, 1.0) == 100
    assert sb.pct([None], 0.5) is None and sb.pct([3, None], 0.5) == 3
