"""
# resolv/harness/__init__.py

**What:** the deterministic harness around the agent: policy rules, the only write path (tools), the ledger and
audit trail, claim verification, guardrails and ticket routing.
**Why:** the agent proposes; this package decides and records. Nothing in it calls a model.
**Used by:** resolv/agents/loop.py, resolv/api/*, evals/*, app.py.
"""
