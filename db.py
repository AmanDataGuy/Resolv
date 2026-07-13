"""Storage interface for exception state.

Backing store is a plain in-memory dict keyed by exception_id. Every function is
async-shaped even though a dict access is synchronous, so swapping this for a real
datastore (Firestore, Postgres) is a change inside this module only — no caller
signature changes. Kept dependency-free on purpose: no cloud SDK import belongs here.
"""
from typing import Any

_EXCEPTIONS: dict[str, dict[str, Any]] = {}


async def get_exception(exception_id: str) -> dict[str, Any]:
    return _EXCEPTIONS.get(exception_id, {})


async def update_exception_state(exception_id: str, updates: dict[str, Any]) -> None:
    existing = _EXCEPTIONS.setdefault(exception_id, {})
    existing.update(updates)


async def list_exceptions(
    status: str | None = None,
    supplier_id: str | None = None,
    urgency: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    results = list(_EXCEPTIONS.values())
    if status:
        results = [r for r in results if r.get("status") == status]
    if supplier_id:
        results = [r for r in results if r.get("supplier_id") == supplier_id]
    if urgency:
        results = [r for r in results if r.get("urgency") == urgency]
    return results[:limit]


def _reset() -> None:
    """Test-only helper to clear state between test runs."""
    _EXCEPTIONS.clear()
