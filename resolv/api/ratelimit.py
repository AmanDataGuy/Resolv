"""
# resolv/api/ratelimit.py

**What:** a fixed-window, in-memory rate limiter keyed by any string (e.g. "resolve:<ip>", "token:<ip>").
**Why:** every /resolve hit is a real LLM call with real cost, and /token must make proof-code guessing
infeasible. The dict is bounded: expired keys are pruned once it grows past MAX_KEYS. Behind a platform proxy,
run uvicorn with `--proxy-headers --forwarded-allow-ips="*"` (only safe because the app port is reachable solely
through the proxy), or set TRUSTED_CLIENT_IP_HEADER (e.g. Fly-Client-IP) so the real client IP is used.
**Used by:** resolv/api/main.py.
"""
import os
import time
from collections import deque

WINDOW_SECONDS = 60
MAX_KEYS = 10_000

# ponytail: per-process state; a shared store (e.g. Redis) the moment there is a second instance.
_hits: dict[str, deque] = {}


# The caller's IP: the trusted proxy header when configured, else the socket peer (proxy-resolved by uvicorn).
def client_ip(request) -> str:
    header = os.environ.get("TRUSTED_CLIENT_IP_HEADER")
    if header and request.headers.get(header):
        return request.headers[header].strip()[:64]
    return request.client.host if request.client else "unknown"


# Drop every key whose newest hit is outside the window.
def _prune(window_start: float) -> None:
    for key in [k for k, hits in _hits.items() if not hits or hits[-1] < window_start]:
        del _hits[key]


# True if `key` is under `limit` hits in the current window; an allowed call counts as a hit.
def is_allowed(key: str, limit: int = 10) -> bool:
    now = time.monotonic()
    window_start = now - WINDOW_SECONDS
    if len(_hits) > MAX_KEYS:
        _prune(window_start)
    hits = _hits.setdefault(key, deque())
    while hits and hits[0] < window_start:
        hits.popleft()
    if len(hits) >= limit:
        return False
    hits.append(now)
    return True
