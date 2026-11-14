"""
# resolv/__init__.py

**What:** the Resolv application package: config, the agent, the policy harness, the API and its integrations.
**Why:** one importable namespace keeps the app separate from its offline tooling (evals/, inference/, loadtest/, scripts/).
**Used by:** every entry point (`uvicorn resolv.api.main:app`, app.py, evals, tests).
"""
