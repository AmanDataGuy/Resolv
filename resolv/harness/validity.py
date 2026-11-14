"""
# resolv/harness/validity.py

**What:** the deterministic claim verifier: is "my order was late / never arrived / was canceled" true against
the order record in data/db/orders.json? Also the order lookup every other harness module uses.
**Why:** one function, no LLM, serves as the production gate (policy rule 3) and the extractor's training reward.
Three outcomes drive three actions: order not found -> claim_true None (ask for the number); found and true ->
proceed; found and false -> reject. Dates are compared as ISO "YYYY-MM-DD" strings, which sort correctly;
scripts/gen_complaint_cases.py buckets orders the same way and asserts the two agree.
**Used by:** resolv/harness/policy.py, resolv/harness/tools.py, resolv/api/escalations.py, evals/*, scripts/gen_complaint_cases.py.
"""
import json

from resolv.config import DB_DIR
from resolv.schemas import ClaimFinding, CustomerClaim

_ORDERS_FILE = f"{DB_DIR}/orders.json"
_orders: dict | None = None


# order_id -> record, loaded on first use (never at import).
def _load() -> dict:
    global _orders
    if _orders is None:
        _orders = {o["order_id"]: o for o in json.loads(open(_ORDERS_FILE).read())}
    return _orders


# The order record, or {} if unknown.
def get_order(order_id: str) -> dict:
    return _load().get(order_id, {})


# Does this claim match what actually happened to the order? Pure record check.
def _claim_holds(claim_type: str, order: dict) -> bool:
    delivered = order.get("delivered_date")
    status = order.get("status")
    if claim_type == "late_delivery":
        return delivered is not None and delivered > order["promised_date"]
    if claim_type == "never_arrived":
        return delivered is None and status == "shipped"
    if claim_type == "order_canceled":
        return status == "canceled"
    return False


# Deterministic verdict on a customer claim against the order record.
def verify_claim(claim: CustomerClaim) -> ClaimFinding:
    if not claim.order_id:
        return ClaimFinding(order_found=False, claim_true=None,
                            reason="No order number given — ask the customer to confirm it.")

    order = get_order(claim.order_id)
    if not order:
        return ClaimFinding(
            order_found=False, claim_true=None, order_id=claim.order_id,
            reason=f"Order {claim.order_id} is not in our records — ask the customer to re-check the number.",
        )

    if not claim.claim_type:  # order found but no problem stated yet: "can't judge" is the honest answer
        return ClaimFinding(
            order_found=True, claim_true=None, order_id=claim.order_id, amount_usd=order.get("amount_usd"),
            reason="Order found, but no problem was stated — ask the customer what went wrong.",
        )

    holds = _claim_holds(claim.claim_type, order)
    label = claim.claim_type.replace("_", " ")
    if holds:
        reason = f"Confirmed against the record: {label} is accurate."
    else:
        reason = f"Record does not support '{label}' (status={order['status']}, delivered={order.get('delivered_date')})."
    return ClaimFinding(order_found=True, claim_true=holds, order_id=claim.order_id,
                        amount_usd=order.get("amount_usd"), reason=reason)
