"""Order data access — plain functions, no LLM involved.

This used to live under mcp/order_server.py. Renamed: it was never a real MCP
(Model Context Protocol) server — no stdio transport, no cross-process protocol,
just Python functions called in-process. "mcp/" was a misleading folder name
copied from resolv.md's original naming, not an actual architectural choice.
Real MCP is worth using when a tool needs to be shared across multiple different
agent frameworks/clients — Resolv's tools are used by exactly one app, so plain
functions are the right choice, just correctly named now.

Data source: reads data/mock_orders.json. Every function here is async even though
a JSON read is synchronous — that's deliberate. It keeps the call sites identical
to what they'd be against a real datastore (Firestore, Postgres), so swapping the
backing store later is a change inside these functions only, never at any caller.
"""
import json

from config import MOCK_DATA_DIR

_ORDERS_FILE = f"{MOCK_DATA_DIR}/mock_orders.json"


def _load_orders() -> list[dict]:
    with open(_ORDERS_FILE) as f:
        return json.load(f)


async def query_orders_by_supplier(
    supplier_id: str,
    status_filter: list[str] | None = None,
) -> list[dict]:
    """Returns orders for a supplier, optionally filtered by status."""
    orders = [o for o in _load_orders() if o["supplier_id"] == supplier_id]
    if status_filter:
        orders = [o for o in orders if o["status"] in status_filter]
    return orders


async def get_order_details(order_id: str) -> dict:
    """Returns a single order document, or {} if not found."""
    for order in _load_orders():
        if order["order_id"] == order_id:
            return order
    return {}


async def get_orders_by_ids(order_ids: list[str]) -> list[dict]:
    """Returns orders matching any of the given order_ids."""
    return [o for o in _load_orders() if o["order_id"] in order_ids]
