"""
# tests/test_mcp_server.py

**What:** every tool on resolv/mcp_server.py, called through the `mcp` package's in-process client (no network).
**Why:** the server must give the harness's own answers and must never move money: check_refund on a payable
refund returns `allow` and leaves zero refund rows.
**Used by:** pytest.
"""
import pytest
from mcp import Client

from resolv.harness import ledger, tools
from resolv.mcp_server import server


# Call one tool in-process and return its structured result.
async def _call(name: str, args: dict):
    async with Client(server) as client:
        result = await client.call_tool(name, args)
    assert not result.is_error, result
    return result.structured_content


# Only the three read-only tools are exposed; there is no refund tool.
@pytest.mark.asyncio
async def test_only_read_only_tools_are_listed():
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {"lookup_order", "check_refund", "list_escalations"}


# lookup_order returns the record, and an unknown id is an error value, not an exception.
@pytest.mark.asyncio
async def test_lookup_order(fresh_order):
    found = await _call("lookup_order", {"order_id": fresh_order["order_id"]})
    assert found["amount_usd"] == fresh_order["amount_usd"]
    assert await _call("lookup_order", {"order_id": "ORD-999999"}) == {"error": "unknown_order",
                                                                      "order_id": "ORD-999999"}


# check_refund gives the policy verdict for the order's owner by default and never pays.
@pytest.mark.asyncio
async def test_check_refund_is_a_dry_run(fresh_order):
    oid = fresh_order["order_id"]
    ok = await _call("check_refund", {"order_id": oid, "claim_type": "late_delivery", "amount_usd": 1.0})
    assert (ok["action"], ok["rule_id"]) == ("allow", "within_policy")
    too_much = await _call("check_refund", {"order_id": oid, "claim_type": "late_delivery",
                                            "amount_usd": fresh_order["amount_usd"]})
    assert too_much["rule_id"] == "refund_exceeds_cap"
    stranger = await _call("check_refund", {"order_id": oid, "claim_type": "late_delivery", "amount_usd": 1.0,
                                            "caller_id": "someone_else"})
    assert stranger["rule_id"] == "caller_not_order_owner"
    unknown = await _call("check_refund", {"order_id": "ORD-999999", "claim_type": "late_delivery", "amount_usd": 1.0})
    assert unknown["rule_id"] == "unknown_order"
    assert ledger.refunds() == []


# check_refund sees refunds already paid (policy rule 5).
@pytest.mark.asyncio
async def test_check_refund_reads_live_refunds(fresh_order):
    oid, owner = fresh_order["order_id"], fresh_order["customer_id"]
    assert tools.issue_refund("_mcp_case", oid, "late_delivery", 1.0, owner).startswith("Refunded")
    again = await _call("check_refund", {"order_id": oid, "claim_type": "late_delivery", "amount_usd": 1.0})
    assert again["rule_id"] == "already_refunded"


# list_escalations shows the approval queue as the console summarises it.
@pytest.mark.asyncio
async def test_list_escalations(big_order):
    assert (await _call("list_escalations", {}))["result"] == []
    tools.issue_refund("_mcp_case", big_order["order_id"], "never_arrived", big_order["amount_usd"],
                       big_order["customer_id"])
    [esc] = (await _call("list_escalations", {"status": "pending"}))["result"]
    assert (esc["order_id"], esc["kind"], esc["amount"]) == (big_order["order_id"], "approval", big_order["amount_usd"])
    assert (await _call("list_escalations", {"status": "approved"}))["result"] == []
