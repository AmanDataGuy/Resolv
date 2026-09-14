"""A minimal, dependency-free rate limiter for /resolve.

Why this exists: harness/policy.py guards the money, but nothing guarded the ENDPOINT itself --
no auth, no limit, so anyone who found the URL could hit /resolve as often as they liked, and
every hit is a real LLM call with real cost. This is the cheap, real fix: cap how often one caller
can hit the expensive endpoint. It is NOT a substitute for real authentication -- that's a bigger,
separate piece of work (see eval_report.md's open items) -- this only stops unauthenticated cost
abuse, not identity spoofing.

Fixed window, per-client-id, in memory. This carries the same single-process assumption
harness/audit.py's local-disk trail already depends on: fine for the one-instance deployment this
project actually runs today, and it would need a shared store (e.g. Redis) the moment a second
instance is added -- for exactly the same reason the audit trail would. Not a new dependency:
stdlib only, matching the project's general "add a dependency only when a few lines can't do it"
stance.
"""
import time
from collections import defaultdict

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 10  # generous for a demo; tune against real traffic before relying on it

_hits: dict[str, list[float]] = defaultdict(list)


def is_allowed(client_id: str) -> bool:
    """True if `client_id` (e.g. a caller's IP) is still under the limit for the current window.

    Records the hit as a side effect when allowed, so a caller's own successful request counts
    toward their own next check -- the point of a rate limiter is that calling it IS the count.
    """
    now = time.monotonic()
    window_start = now - WINDOW_SECONDS
    hits = _hits[client_id]
    # Drop hits outside the window before counting -- keeps memory bounded without a cron job,
    # since a client that stops calling eventually prunes itself down to an empty list.
    while hits and hits[0] < window_start:
        hits.pop(0)
    if len(hits) >= MAX_REQUESTS_PER_WINDOW:
        return False
    hits.append(now)
    return True


def demo() -> None:
    """No network. Fills one client's window, confirms the next call is refused, confirms a
    different client is unaffected."""
    client = "_demo_client"
    _hits.pop(client, None)

    for _ in range(MAX_REQUESTS_PER_WINDOW):
        assert is_allowed(client)
    assert not is_allowed(client), f"the {MAX_REQUESTS_PER_WINDOW + 1}th call in one window must be refused"
    assert is_allowed("_demo_other_client"), "a different client must not share the first one's count"

    print(f"ratelimit demo OK — {MAX_REQUESTS_PER_WINDOW} allowed then refused within {WINDOW_SECONDS}s")


if __name__ == "__main__":
    demo()
