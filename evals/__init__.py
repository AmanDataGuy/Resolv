"""
# evals/__init__.py

**What:** the evaluation package: the pass^k benchmark, its adversarial customer, the graders, the scoring math,
and the online monitors.
**Why:** everything that turns agent runs into published numbers lives here, so a reader can audit one folder.
Every eval entry point sets `RESOLV_ENV=eval` before importing anything else.
**Used by:** `python -m evals.<module>` entry points, tests, resolv/api/main.py (monitor), app.py.
"""
