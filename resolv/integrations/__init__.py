"""
# resolv/integrations/__init__.py

**What:** stand-ins for outside systems: notify (customer email and team pages) and payments (FakePSP).
**Why:** only the transport is simulated, so the decision logic upstream is exercised exactly as in production.
**Used by:** resolv/api/*, resolv/harness/tools.py, app.py.
"""
