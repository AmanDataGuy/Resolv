"""Mock session tokens — the honest half of "who is calling" that this project can actually build.

WHAT THIS IS NOT. There is no real login system here, no password, no user database beyond the
order records themselves — building one is out of scope for a refund-harness demo. This is NOT
authentication in the sense of proving a human's identity.

WHAT THIS IS. Before this file existed, /resolve trusted a bare `customer_id` string in the
request body -- anyone could type any customer_id and the harness's ownership check
(harness/policy.py rule 2) would happily "verify" a refund for a caller who never proved anything.
A signed, expiring token closes the gap that IS buildable without a real identity provider: the
caller must hold a token this server itself issued for that customer_id, not just know the string.
POST /token mocks "the customer already proved who they are somewhere else" (clicked a link in
their order confirmation email, was already logged into a storefront session) exactly the way
integrations/notify.py mocks a real support channel — the same "be honest about what's real and
what's a stand-in" pattern used everywhere else in this project.

DEPENDENCY-FREE ON PURPOSE. hmac + hashlib + base64 + json, same reasoning as api/ratelimit.py's
own docstring: a few lines of stdlib beats a JWT library for a token this simple (one claim, one
expiry, one signature, no key rotation, no algorithm negotiation to get wrong).
"""
import base64
import hashlib
import hmac
import json
import os
import time

# A real deployment sets AUTH_SECRET in its environment. The fallback exists so the demo and the
# test suite work with zero setup -- it is not a secret, and using it in anything but a local demo
# would defeat the whole point of signing. (ponytail: env-var secret with a loud dev fallback;
# a real deployment's ops checklist should treat an unset AUTH_SECRET as a deploy blocker.)
_SECRET = os.environ.get("AUTH_SECRET", "resolv-dev-only-secret-do-not-deploy-with-this").encode()

DEFAULT_TTL_SECONDS = 3600  # an hour is plenty for one support conversation


def _sign(payload_b64: bytes) -> str:
    return hmac.new(_SECRET, payload_b64, hashlib.sha256).hexdigest()


def issue_token(customer_id: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> str:
    """A signed, expiring token binding this session to one customer_id.

    Format is `<base64 payload>.<hex hmac>` -- deliberately not a JWT (no alg header to confuse an
    attacker into requesting "none", no library to keep patched). The payload is base64 only for
    safe transport in a header/JSON string; it is NOT encryption; customer_id is visible to
    anyone holding the token, exactly as visible as it already was in the old trusted field.
    """
    payload = json.dumps({"customer_id": customer_id, "exp": time.time() + ttl_seconds},
                         separators=(",", ":"), sort_keys=True).encode()
    payload_b64 = base64.urlsafe_b64encode(payload)
    return f"{payload_b64.decode()}.{_sign(payload_b64)}"


def verify_token(token: str) -> str | None:
    """The customer_id a valid, unexpired token was issued for -- or None if it's malformed,
    forged, or expired. Returning None rather than raising: the caller (api/main.py) turns this
    into one clean 401, the same "a bad token is data to reject, not an exception to crash on"
    reasoning harness/tools.py already uses for a malformed tool call.
    """
    try:
        payload_b64_str, sig = token.split(".", 1)
    except ValueError:
        return None
    payload_b64 = payload_b64_str.encode()
    # Constant-time comparison -- a signature check that short-circuits on the first mismatched
    # byte leaks how much of a forged signature is correct, one HTTP timing measurement at a time.
    if not hmac.compare_digest(_sign(payload_b64), sig):
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    if claims.get("exp", 0) < time.time():
        return None
    customer_id = claims.get("customer_id")
    return customer_id if isinstance(customer_id, str) and customer_id else None


def demo() -> None:
    """No network, no key needed -- the whole point of a stdlib-only signer."""
    token = issue_token("cust-123", ttl_seconds=60)
    assert verify_token(token) == "cust-123"

    # Tampering with either half invalidates it.
    payload, sig = token.split(".", 1)
    assert verify_token(f"{payload}.{'0' * len(sig)}") is None, "a forged signature must not verify"
    assert verify_token(f"{base64.urlsafe_b64encode(b'{}').decode()}.{sig}") is None, (
        "a swapped payload under someone else's signature must not verify"
    )

    # Expiry is enforced, not just present in the payload.
    expired = issue_token("cust-123", ttl_seconds=-1)
    assert verify_token(expired) is None, "an expired token must not verify"

    # Garbage input is rejected, not raised.
    assert verify_token("not-a-real-token") is None
    assert verify_token("") is None

    print("auth demo OK — sign, verify, tamper-detect, and expire all hold")


if __name__ == "__main__":
    demo()
