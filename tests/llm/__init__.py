"""
# tests/llm/__init__.py

**What:** marks tests/llm/ as a package of LLM-quality tests (deepeval / ragas).
**Why:** these need a provider key and are skipped in keyless CI, so they live apart from the keyless suite.
**Used by:** pytest.
"""
