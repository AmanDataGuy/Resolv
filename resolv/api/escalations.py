"""
# resolv/api/escalations.py

**What:** the human approval API: list and inspect escalations, approve or deny one. Every route needs an approver
bearer token.
**Why:** an approval is bound to one concrete tool call (its action_hash), and approving executes that exact call
through the same tools.issue_refund gate, so rules 0-6 are re-checked and a human can only approve what they saw
(the raw action JSON, not the agent's summary). Approve requires an Idempotency-Key header, so a retried click never
pays twice. An approver can never approve their own refund.
**Used by:** resolv/api/main.py (`app.include_router(escalations.router)`), resolv/mcp_server.py (`_summary`).
"""
import hashlib
import json

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from resolv.api import auth
from resolv.harness import audit, ledger, tools
from resolv.harness.policy import Decision
from resolv.harness.validity import get_order
from resolv.integrations import notify

router = APIRouter(prefix="/escalations")


# The approver's note on a decision.
class Decide(BaseModel):
    note: str = Field(default="", max_length=2000)


# The approver id behind a bearer token: 401 if the token is bad, 403 if it does not name an approver.
def approver(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    sub = auth.verify_token(token, role="approver")
    if sub:
        return sub
    if auth.token_claims(token):
        raise HTTPException(403, "Approver role required.")
    raise HTTPException(401, "Missing, invalid, or expired token.")


# One escalation or 404.
def _load(esc_id: str) -> dict:
    esc = ledger.escalation(esc_id)
    if esc is None:
        raise HTTPException(404, f"No escalation {esc_id}.")
    return esc


# The list view of one escalation.
def _summary(e: dict) -> dict:
    amount = json.loads(e["action_json"])["amount_cents"] / 100 if e["action_json"] else None
    return {k: e[k] for k in ("id", "case_id", "order_id", "kind", "status", "reason", "created_at", "expires_at")} \
        | {"amount": amount}


# List escalations by status (pending by default), expiring stale ones first.
@router.get("")
def list_escalations(status: str = "pending", limit: int = 50, _: str = Depends(approver)) -> list[dict]:
    ledger.expire_escalations()
    return [_summary(e) for e in ledger.escalations(status, max(1, min(limit, 200)))]


# Everything an approver needs to decide: the raw action, the order facts, the full trail and every rule hit.
@router.get("/{esc_id}")
def get_escalation(esc_id: str, _: str = Depends(approver)) -> dict:
    ledger.expire_escalations()
    esc = _load(esc_id)
    trail = audit.read(esc["case_id"])
    return {
        "escalation": esc,
        "action_json": esc["action_json"],
        "action_hash": esc["action_hash"],
        "order": get_order(esc["order_id"]) if esc["order_id"] else {},
        "trail": trail,
        "policy_explanation": [{"rule_id": r["rule_id"], "reason": r["reason"]} for r in trail],
    }


# Approve a pending approval escalation and execute its exact action through the gate (Idempotency-Key required).
@router.post("/{esc_id}/approve")
def approve(esc_id: str, body: Decide, idempotency_key: str = Header(alias="Idempotency-Key"),
            who: str = Depends(approver)) -> dict:
    request_sha = hashlib.sha256(json.dumps([esc_id, body.note, who]).encode()).hexdigest()
    state, saved = ledger.claim_key(idempotency_key, "approve", request_sha)
    if state == "replay":
        return saved
    if state == "conflict":
        raise HTTPException(422, "Idempotency-Key reused with a different request")
    if state == "in_flight":
        raise HTTPException(409, "A request with this Idempotency-Key is still running")
    try:
        response = _approve(esc_id, body.note, who)
    except BaseException:
        ledger.finish_key(idempotency_key, None)  # release the key so the client can retry
        raise
    ledger.finish_key(idempotency_key, response)
    return response


# Mark approved, run tools.issue_refund with the bound approval, record the result, notify the customer.
def _approve(esc_id: str, note: str, who: str) -> dict:
    esc = _load(esc_id)
    if esc["kind"] != "approval":
        raise HTTPException(409, "Only approval escalations carry an action to execute.")
    action = json.loads(esc["action_json"])
    if who == action["caller_id"]:
        raise HTTPException(403, "An approver cannot approve their own refund.")
    with ledger.tx() as conn:
        if not ledger.decide_escalation(conn, esc_id, "approved", who, note):
            raise HTTPException(409, f"Escalation is {esc['status']}, not pending (or it expired).")

    result = tools.issue_refund(esc["case_id"], action["order_id"], action["claim_type"],
                                action["amount_cents"] / 100, action["caller_id"],
                                approval={"id": esc_id}, approver_id=who)
    with ledger.tx() as conn:  # executed by the refund itself; anything still "approved" was refused at execution
        if not ledger.move_escalation(conn, esc_id, "approved", "failed", result):
            ledger.move_escalation(conn, esc_id, "executed", "executed", result)
    notify.drain_outbox()
    return {"result": result, "escalation": ledger.escalation(esc_id), "trail": audit.read(esc["case_id"])}


# Deny a pending escalation, record the decision, and tell the customer it was reviewed.
@router.post("/{esc_id}/deny")
def deny(esc_id: str, body: Decide, who: str = Depends(approver)) -> dict:
    esc = _load(esc_id)
    with ledger.tx() as conn:
        if not ledger.decide_escalation(conn, esc_id, "denied", who, body.note):
            raise HTTPException(409, f"Escalation is {esc['status']}, not pending (or it expired).")
        decision = Decision(action="deny", rule_id="denied_by_human", reason=body.note or "Denied by a reviewer.")
        ledger.append_audit(conn, esc["case_id"], "review_escalation", esc["order_id"],
                            {"escalation_id": esc_id, "note": body.note}, decision, False,
                            "Reviewed; unchanged.", esc["requested_by"], who)
        ledger.add_outbox(conn, "escalation_denied", {"case_id": esc["case_id"], "order_id": esc["order_id"],
                                                      "escalation_id": esc_id, "message": "Reviewed; unchanged."})
    notify.drain_outbox()
    return {"escalation": ledger.escalation(esc_id), "trail": audit.read(esc["case_id"])}
