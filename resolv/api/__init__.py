"""
# resolv/api/__init__.py

**What:** the HTTP package: the FastAPI app, session tokens, rate limiting, and the escalation queue routes.
**Why:** transport concerns only; no business logic, so the API can't approve what the harness wouldn't.
**Used by:** uvicorn (resolv.api.main:app), tests.
"""
