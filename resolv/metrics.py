"""
# resolv/metrics.py

**What:** the Prometheus metrics Resolv exposes on `GET /metrics`: refund decisions, money paid, /resolve latency,
LLM spend, the kill-switch state and request errors.
**Why:** one small, fixed set, each counted at the single place its fact is decided: refund decisions in
resolv/harness/tools.py `issue_refund`, money in `_settle`, LLM cost in resolv/agents/llm.py `_on_success`, latency
and errors in resolv/api/main.py `_resolve`. Labels are rule ids and error kinds only, never customer data.
Alerts and the dashboard that read these live in ops/.
**Used by:** resolv/harness/tools.py, resolv/agents/llm.py, resolv/api/main.py.
"""
import os

from prometheus_client import Counter, Gauge, Histogram

# prometheus_client adds the `_total` suffix to counters on export.
REFUND_DECISIONS = Counter("resolv_refund_decisions", "Refund requests by harness decision and rule.",
                           ["decision", "rule"])
REFUND_USD = Counter("resolv_refund_usd", "US dollars refunded and settled with the payment provider.")
RESOLVE_SECONDS = Histogram("resolv_resolve_seconds", "Wall-clock seconds for one successful /resolve.",
                            buckets=(0.5, 1, 2, 5, 10, 15, 30, 60, 90))
LLM_COST_USD = Counter("resolv_llm_cost_usd", "LLM spend in US dollars, from the price table in resolv/config.py.")
ERRORS = Counter("resolv_errors", "Requests that failed with an error, by kind.", ["kind"])
REFUNDS_FROZEN = Gauge("resolv_refunds_frozen", "1 while the REFUNDS_FROZEN kill switch is on.")
# Read at scrape time, the same way tools.py reads the switch on every refund.
REFUNDS_FROZEN.set_function(lambda: float(os.environ.get("REFUNDS_FROZEN") == "1"))
