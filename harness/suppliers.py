"""Supplier data access — plain functions, no LLM involved.

Moved from mcp/supplier_server.py (see harness/orders.py's docstring for why the
"mcp/" folder name was misleading and has been dropped).

get_supplier_context() is new here: it composes get_supplier_profile() with
get_alternate_suppliers() using a plain if-statement, instead of an LLM agent
deciding whether to look up alternates. The original design had an LLM read an
instruction like "if exception_type is stockout or supplier_failure, call
get_alternate_suppliers" — that's a fixed rule, not a judgment call, so it's a
Python if-statement now.
"""
import json
from datetime import datetime, timedelta, timezone

from config import DB_DIR

_SUPPLIERS_FILE = f"{DB_DIR}/suppliers.json"

_NEEDS_ALTERNATES = {"stockout", "supplier_failure"}


def _load_suppliers() -> dict:
    with open(_SUPPLIERS_FILE) as f:
        return json.load(f)


async def get_supplier_profile(supplier_id: str) -> dict:
    """Returns supplier profile. {"contact_found": False} if unknown or no contact_email."""
    suppliers = _load_suppliers()
    profile = suppliers.get(supplier_id)
    if profile is None:
        return {"contact_found": False}
    profile = dict(profile)
    profile["contact_found"] = bool(profile.get("contact_email"))
    return profile


async def get_sla_terms(supplier_id: str) -> dict:
    """Returns the supplier's SLA terms dict, or {} if unknown."""
    suppliers = _load_suppliers()
    profile = suppliers.get(supplier_id, {})
    return profile.get("sla_terms", {})


async def check_notification_deadline(
    supplier_id: str,
    exception_type: str,
    detected_at: str,
) -> dict:
    """Returns {deadline_iso, hours_remaining, is_urgent}. is_urgent when <= 2h remain."""
    sla = await get_sla_terms(supplier_id)
    deadline_hours = sla.get("notification_deadline_hours", {}).get(exception_type, 24)
    detected = datetime.fromisoformat(detected_at)
    if detected.tzinfo is None:
        detected = detected.replace(tzinfo=timezone.utc)
    deadline = detected + timedelta(hours=deadline_hours)
    now = datetime.now(timezone.utc)
    hours_remaining = (deadline - now).total_seconds() / 3600
    return {
        "deadline_iso": deadline.isoformat(),
        "hours_remaining": round(hours_remaining, 1),
        "is_urgent": hours_remaining <= 2,
    }


async def calculate_penalty(
    supplier_id: str,
    exception_type: str,
    delay_days: int,
    revenue_at_risk_usd: float,
) -> dict:
    """Returns {penalty_usd, penalty_pct, clause_id}."""
    sla = await get_sla_terms(supplier_id)
    pct = sla.get("late_delivery_penalty_pct", 0) if exception_type == "late_shipment" else 0
    penalty = revenue_at_risk_usd * pct / 100
    return {
        "penalty_usd": round(penalty, 2),
        "penalty_pct": pct,
        "clause_id": sla.get("penalty_clause_id"),
    }


async def get_alternate_suppliers(product_category: str, exclude_supplier_id: str) -> list[dict]:
    """Returns backup suppliers (risk_score <= 40) for a product category."""
    suppliers = _load_suppliers()
    return [
        s
        for sid, s in suppliers.items()
        if sid != exclude_supplier_id
        and product_category in s.get("product_categories", [])
        and s.get("risk_score", 100) <= 40
    ]


async def get_supplier_context(supplier_id: str, exception_type: str) -> dict:
    """Composes profile + (conditionally) alternates into the SupplierContext shape.

    Alternates are searched using THIS supplier's own product_categories — the mock
    fixtures have no product_id -> category mapping, only supplier -> categories, so
    "find another supplier who serves the same categories this one does" is the
    closest available proxy for "find a backup for what this supplier was making."
    A real product catalog would let this key off the specific product_ids in the
    exception instead.
    """
    profile = await get_supplier_profile(supplier_id)
    if not profile.get("contact_found"):
        return {"contact_found": False, "alternate_suppliers": []}

    alternates: list[dict] = []
    if exception_type in _NEEDS_ALTERNATES:
        for category in profile.get("product_categories", []):
            alternates.extend(await get_alternate_suppliers(category, supplier_id))

    return {**profile, "alternate_suppliers": alternates}
