"""
# resolv/mcp_server.py

**What:** a read-only MCP server with three tools: `lookup_order` (the order record), `check_refund` (the policy
verdict on a refund, as a dry run) and `list_escalations` (the human approval queue).
**Why:** lets an operator ask Claude Desktop or Claude Code "can ORD-1181 get $900 back?" and get the same answer
the harness would give, without being able to move money. There is deliberately no refund tool: the only write
path stays resolv/harness/tools.py behind the agent loop and the approvals console. Nothing here writes a record.
Run: `python -m resolv.mcp_server` (stdio). Setup for MCP clients: docs/mcp.md.
**Used by:** MCP clients over stdio, tests/test_mcp_server.py.
"""
from typing import Any

from mcp.server.mcpserver import MCPServer

from resolv.api.escalations import _summary
from resolv.harness import ledger, policy
from resolv.harness.validity import get_order

server = MCPServer("resolv", instructions="Read-only refund policy tools. They never issue a refund.")


# The order record for an id, or {"error": "unknown_order"} when there is none.
@server.tool()
def lookup_order(order_id: str) -> dict[str, Any]:
    """The stored record for an order id like ORD-1181: amount paid, promised and delivered dates, status."""
    return get_order(order_id) or {"error": "unknown_order", "order_id": order_id}


# What policy would decide for this refund right now, using the order's live refunds; never pays anything.
@server.tool()
def check_refund(order_id: str, claim_type: str, amount_usd: float,
                 caller_id: str | None = None) -> dict[str, Any]:
    """Dry run of the refund policy: returns action (allow/deny/escalate), rule_id and reason; never refunds.
    claim_type is never_arrived, order_canceled or late_delivery. caller_id defaults to the order's owner."""
    caller = caller_id or get_order(order_id).get("customer_id")
    history = ledger.order_refund_records(order_id)
    return policy.check_refund(order_id, claim_type, amount_usd, history, caller).model_dump()


# Escalations with a status (pending by default), newest first, as the approvals console lists them.
@server.tool()
def list_escalations(status: str = "pending", limit: int = 50) -> list[dict]:
    """Escalations by status (pending, approved, denied, expired, executed, failed). Read-only: a pending item
    past its expires_at is shown as stored; the approvals console marks it expired on its next read."""
    return [_summary(e) for e in ledger.escalations(status, max(1, min(limit, 200)))]


if __name__ == "__main__":
    server.run()  # stdio
