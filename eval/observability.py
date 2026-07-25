"""Langfuse tracing for the live serving path — the dashboard layer over eval/monitor.py.

OPTIONAL BY DESIGN, AND THAT IS THE IMPORTANT PART. If langfuse isn't installed, or the keys
aren't set, every function here is a silent no-op and /resolve runs exactly as before. A
third-party observability service must never be able to break the request path it is watching:
a monitoring tool that can take down production is a bigger liability than the blindness it cures.
So `_client()` returns None when unavailable, and `log_request()` swallows every error.

What Langfuse adds on top of eval/monitor.py: the same label-free signals, but as a hosted trace
per request with latency, cost, and the safety scores charted over time — the "monitoring" half of
observability, without building a dashboard by hand. monitor.py stays the source of truth on disk;
this is the view.

Setup (once): pip install "langfuse<3", then set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and
LANGFUSE_HOST for self-hosted) in .env. Without them this module does nothing, by design.
"""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _client():
    """The Langfuse client, or None if unavailable. Cached — one client per process.

    Three ways it ends up None, all handled the same: keys not set, package not installed, or
    construction failing. None means "observability off", never an error the caller must handle.
    """
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse()  # reads public/secret key + host from the environment
    except Exception:
        return None


def enabled() -> bool:
    """True only when a Langfuse client is actually live — used by the API to report status."""
    return _client() is not None


def log_request(case_id: str, message: str, result: dict, metrics: dict) -> None:
    """Send one live request to Langfuse as a trace, with the label-free metrics as scores.

    `metrics` is the exact dict eval/monitor.record() returned, so nothing is recomputed. The
    whole body is wrapped: a Langfuse outage, an API-version mismatch, or a network blip logs
    nothing and raises nothing. The endpoint must not care whether observability is healthy.
    """
    client = _client()
    if client is None:
        return
    try:
        trace = client.trace(
            name="resolve",
            input={"message": message},
            output={"reply": result.get("reply"), "action": metrics.get("action")},
            metadata={
                "case_id": case_id,
                "steps": result.get("steps"),
                "latency_s": metrics.get("latency_s"),
                "usd": metrics.get("usd"),
                "tokens": metrics.get("tokens"),
                "rules_hit": metrics.get("rules_hit"),
            },
        )
        # Booleans become 0/1 scores so they chart as rates in the Langfuse UI. unauthorized is the
        # one to alert on there — same meaning as everywhere else in the project.
        for name in ("unauthorized", "reply_grounded", "looked_up_first"):
            value = metrics.get(name)
            if value is not None:
                trace.score(name=name, value=float(bool(value)))
        client.flush()  # the API request is short-lived; push before the process moves on
    except Exception:
        pass  # observability is never allowed to break the endpoint it observes


if __name__ == "__main__":
    # No-network self-check: with no keys, the layer must be a clean no-op, not a crash.
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        assert enabled() is False
        log_request("selfcheck", "hi", {"reply": "x", "trail": []},
                    {"action": "deny", "unauthorized": False, "reply_grounded": True})
        print("observability selfcheck OK — no keys, clean no-op (set keys to enable Langfuse)")
    else:
        print(f"observability: Langfuse enabled={enabled()}")
