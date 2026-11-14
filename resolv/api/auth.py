"""
# resolv/api/auth.py

**What:** signed, expiring session tokens (`<base64 payload>.<hex hmac>`) and per-order proof codes.
**Why:** every customer_id is public in data/db/orders.json, so "type your customer id" proves nothing. A customer
gets a token only by showing an order's proof code, `base32(HMAC_SHA256(ORDER_PROOF_SECRET, order_id))[:10]`,
which a real retailer prints on the receipt. The token's subject is that order's customer. Approver tokens are
minted only locally (`python -m resolv.api.auth --mint-approver <id>`), never over HTTP. Stdlib hmac, not a JWT
library: no algorithm header to confuse. In prod both secrets are required (fail closed); in dev/test an unset
secret falls back to a random per-process key, never a known string.
**Used by:** resolv/api/main.py (/token, /resolve), resolv/api/escalations.py (approver tokens), app.py.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import uuid

from resolv import config  # noqa: F401  (loads .env before the secrets below are read)
from resolv.harness.validity import get_order

DEFAULT_TTL_SECONDS = 3600  # an hour is plenty for one support conversation
PROOF_LENGTH = 10  # 32^10 ~ 1.1e15 codes; at 5 guesses/min per IP, guessing one is infeasible


# A required secret in prod (>= 32 chars); in dev/test a random per-process key when unset.
def _secret(name: str) -> bytes:
    raw = os.environ.get(name, "")
    if os.environ.get("RESOLV_ENV", "dev") == "prod":
        if len(raw) < 32:
            raise RuntimeError(f"{name} must be set to at least 32 characters when RESOLV_ENV=prod (fail closed).")
    # ponytail: the dev fallback dies on restart and doesn't cross workers; set the env var for anything shared.
    return raw.encode() or secrets.token_bytes(32)


_SECRET = _secret("AUTH_SECRET")
_PROOF_SECRET = _secret("ORDER_PROOF_SECRET")


# HMAC-SHA256 hex signature of a token payload.
def _sign(payload_b64: bytes) -> str:
    return hmac.new(_SECRET, payload_b64, hashlib.sha256).hexdigest()


# A signed token for `subject` with a role (customer | approver), a unique jti, and an expiry.
def issue_token(subject: str, role: str = "customer", ttl_seconds: int = DEFAULT_TTL_SECONDS) -> str:
    now = time.time()
    payload = json.dumps(
        {"sub": subject, "role": role, "jti": uuid.uuid4().hex, "iat": now, "exp": now + ttl_seconds},
        separators=(",", ":"), sort_keys=True,
    ).encode()
    payload_b64 = base64.urlsafe_b64encode(payload)
    return f"{payload_b64.decode()}.{_sign(payload_b64)}"


# The claims of a valid, unexpired token, or None if it is malformed, forged or expired.
def token_claims(token: str) -> dict | None:
    try:
        payload_b64_str, sig = (token or "").split(".", 1)
        payload_b64 = payload_b64_str.encode()
        # Constant-time compare: a short-circuiting check leaks how much of a forged signature is right.
        if not hmac.compare_digest(_sign(payload_b64), sig):
            return None
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    if not isinstance(claims, dict) or claims.get("exp", 0) < time.time():
        return None
    return claims


# The subject of a valid token with the given role, or None. /resolve uses role="customer".
def verify_token(token: str, role: str = "customer") -> str | None:
    claims = token_claims(token)
    if not claims or claims.get("role") != role:
        return None
    subject = claims.get("sub")
    return subject if isinstance(subject, str) and subject else None


# The receipt code for an order: base32 of the HMAC, first PROOF_LENGTH characters.
def order_proof(order_id: str) -> str:
    digest = hmac.new(_PROOF_SECRET, order_id.encode(), hashlib.sha256).digest()
    return base64.b32encode(digest).decode()[:PROOF_LENGTH]


# The order's customer_id if `proof` matches, else None; unknown orders and wrong proofs look identical.
def check_order_proof(order_id: str, proof: str) -> str | None:
    matches = hmac.compare_digest(order_proof(order_id), (proof or "").upper())
    order = get_order(order_id)
    if not matches or not order:
        return None
    return order.get("customer_id")


if __name__ == "__main__":
    # Local CLI only (needs the secret): python -m resolv.api.auth --mint-approver alice
    if len(sys.argv) == 3 and sys.argv[1] == "--mint-approver":
        print(issue_token(sys.argv[2], role="approver"))
    else:
        print("usage: python -m resolv.api.auth --mint-approver <approver_id>")
