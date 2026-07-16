"""Claim validity — the deterministic verifier. The production gate AND the RLVR reward,
one function, no LLM.

A customer claims something ("my order was late", "it never arrived"). The extractor turns
the messy message into a CustomerClaim{order_id, claim_type}. This module decides whether that
claim is TRUE by looking the order up in the demo DB (data/db/orders.json) and checking the
real record — dates and status. Same exact-match discipline as harness/guardrails.py.

Three outcomes, three actions:
  order not found      -> claim_true = None  -> ask the customer to confirm the order number
  found, claim true    -> claim_true = True  -> valid, proceed (refund/credit/etc.)
  found, claim false   -> claim_true = False -> reject, explain to the customer

The "found but false" outcome is the whole reason on-time orders are in the DB: without it the
verifier could only ever say yes, and a reward that can only say yes has no signal.
"""
import json

from config import DB_DIR
from schemas import ClaimFinding, CustomerClaim

_ORDERS_FILE = f"{DB_DIR}/orders.json"
_orders: dict | None = None


def _load() -> dict:
    """order_id -> record, loaded once."""
    global _orders
    if _orders is None:
        _orders = {o["order_id"]: o for o in json.loads(open(_ORDERS_FILE).read())}
    return _orders


def get_order(order_id: str) -> dict:
    """The order record, or {} if unknown."""
    return _load().get(order_id, {})


def _claim_holds(claim_type: str, order: dict) -> bool:
    """Does this claim match what actually happened to the order? Pure record check."""
    delivered = order.get("delivered_date")
    status = order.get("status")
    if claim_type == "late_delivery":
        return delivered is not None and delivered > order["promised_date"]
    if claim_type == "never_arrived":
        return delivered is None and status == "shipped"
    if claim_type == "order_canceled":
        return status == "canceled"
    if claim_type == "item_unavailable":
        return status == "unavailable"
    return False


def verify_claim(claim: CustomerClaim) -> ClaimFinding:
    """Deterministic verdict on a customer claim against the order record."""
    if not claim.order_id:
        return ClaimFinding(
            order_found=False,
            claim_true=None,
            reason="No order number given — ask the customer to confirm it.",
        )

    order = get_order(claim.order_id)
    if not order:
        return ClaimFinding(
            order_found=False,
            claim_true=None,
            order_id=claim.order_id,
            reason=f"Order {claim.order_id} is not in our records — ask the customer to re-check the number.",
        )

    holds = _claim_holds(claim.claim_type, order)
    if holds:
        reason = f"Confirmed against the record: {claim.claim_type.replace('_', ' ')} is accurate."
    else:
        reason = (
            f"Record does not support '{claim.claim_type.replace('_', ' ')}' "
            f"(status={order['status']}, delivered={order.get('delivered_date')})."
        )
    return ClaimFinding(
        order_found=True,
        claim_true=holds,
        order_id=claim.order_id,
        amount_usd=order.get("amount_usd"),
        reason=reason,
    )
