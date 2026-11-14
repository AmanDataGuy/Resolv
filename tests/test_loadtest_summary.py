"""
# tests/test_loadtest_summary.py

**What:** keyless checks of the load test's math: percentiles and the per-endpoint summary.
**Why:** a wrong p95 silently moves the saturation knee; this pins the arithmetic without a server.
**Used by:** pytest.
"""
from loadtest.resolve import percentile, summarize


# Linear interpolation between ranks, the same convention as numpy's default.
def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0
    assert abs(percentile(values, 95) - 4.8) < 1e-9
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0  # unsorted input


# Empty input gives None and a single value is every percentile.
def test_percentile_edges():
    assert percentile([], 95) is None
    assert percentile([2.5], 99) == 2.5


# Errors count toward the error rate but not toward latency percentiles; endpoints are summarized separately.
def test_summarize_splits_endpoints_and_errors():
    rows = [
        {"endpoint": "resolve", "latency_s": 1.0, "status": 200, "ok": True},
        {"endpoint": "resolve", "latency_s": 3.0, "status": 200, "ok": True},
        {"endpoint": "resolve", "latency_s": 99.0, "status": 503, "ok": False},
        {"endpoint": "resolve", "latency_s": 0.1, "status": 0, "ok": False},
        {"endpoint": "health", "latency_s": 0.01, "status": 200, "ok": True},
    ]
    out = summarize(rows, duration_s=2.0)
    r = out["resolve"]
    assert r["requests"] == 4 and r["errors"] == 2 and r["error_rate"] == 0.5
    assert r["ok_per_s"] == 1.0
    assert r["p50_s"] == 2.0 and r["p99_s"] < 3.0  # the 99 s failure is not a latency sample
    assert r["status_counts"] == {"0": 1, "200": 2, "503": 1}
    assert out["health"]["error_rate"] == 0.0 and out["health"]["p95_s"] == 0.01


# An endpoint with only failures has no latency percentiles.
def test_summarize_all_failed():
    out = summarize([{"endpoint": "resolve", "latency_s": 1.0, "status": 0, "ok": False}], duration_s=1.0)
    assert out["resolve"]["p95_s"] is None and out["resolve"]["error_rate"] == 1.0
