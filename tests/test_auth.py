"""
# tests/test_auth.py

**What:** resolv/api/auth.py tokens and order-proof codes, and the /token + /resolve gates in resolv/api/main.py.
**Why:** a token is issued only for a valid receipt proof, unknown orders and wrong proofs are indistinguishable,
approver tokens can't be minted over HTTP, prod fails closed without secrets, and every rejection happens
before any LLM call. Keyless.
**Used by:** pytest.
"""
import importlib

import pytest
from fastapi.testclient import TestClient

from resolv.api import auth, ratelimit
from resolv.api.main import app
from resolv.harness.validity import get_order

ORDER = "ORD-1000"


# A test client with a clean rate-limit table.
@pytest.fixture
def client():
    ratelimit._hits.clear()
    return TestClient(app)


# Signed customer and approver tokens.
class TestTokens:
    # Round trip.
    def test_round_trip(self):
        assert auth.verify_token(auth.issue_token("cust-1")) == "cust-1"

    # Claims carry role jti iat.
    def test_claims_carry_role_jti_iat(self):
        claims = auth.token_claims(auth.issue_token("cust-1"))
        assert claims["role"] == "customer" and claims["jti"] and claims["iat"] <= claims["exp"]

    # Forged signature rejected.
    def test_forged_signature_rejected(self):
        payload, sig = auth.issue_token("cust-1").split(".", 1)
        assert auth.verify_token(f"{payload}.{'0' * len(sig)}") is None

    # Expired rejected.
    def test_expired_rejected(self):
        assert auth.verify_token(auth.issue_token("cust-1", ttl_seconds=-1)) is None

    # Garbage rejected not raised.
    def test_garbage_rejected_not_raised(self):
        assert auth.verify_token("") is None
        assert auth.verify_token("not-a-token") is None

    # Approver token is not a customer token.
    def test_approver_token_is_not_a_customer_token(self):
        token = auth.issue_token("alice", role="approver")
        assert auth.verify_token(token) is None
        assert auth.verify_token(token, role="approver") == "alice"


# Order-proof check used to mint a customer token.
class TestOrderProof:
    # Proof shape.
    def test_proof_shape(self):
        proof = auth.order_proof(ORDER)
        assert len(proof) == auth.PROOF_LENGTH and proof == proof.upper()

    # Right proof gives owner.
    def test_right_proof_gives_owner(self):
        assert auth.check_order_proof(ORDER, auth.order_proof(ORDER)) == get_order(ORDER)["customer_id"]

    # Wrong proof and unknown order give none.
    def test_wrong_proof_and_unknown_order_give_none(self):
        assert auth.check_order_proof(ORDER, "AAAAAAAAAA") is None
        assert auth.check_order_proof("ORD-99999", auth.order_proof("ORD-99999")) is None

    # Proof compare is constant time.
    def test_proof_compare_is_constant_time(self, monkeypatch):
        calls = []
        real = auth.hmac.compare_digest
        monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: calls.append((a, b)) or real(a, b))
        auth.check_order_proof(ORDER, "AAAAAAAAAA")
        assert calls


# HTTP endpoints reject bad tokens, proofs and inputs.
class TestEndpoints:
    # Token requires valid proof.
    def test_token_requires_valid_proof(self, client):
        assert client.post("/token", json={"order_id": ORDER, "proof": "AAAAAAAAAA"}).status_code == 401
        r = client.post("/token", json={"order_id": ORDER, "proof": auth.order_proof(ORDER)})
        assert r.status_code == 200
        assert auth.verify_token(r.json()["token"]) == get_order(ORDER)["customer_id"]

    # Unknown order and bad proof are indistinguishable.
    def test_unknown_order_and_bad_proof_are_indistinguishable(self, client):
        a = client.post("/token", json={"order_id": "ORD-99999", "proof": "AAAAAAAAAA"})
        b = client.post("/token", json={"order_id": ORDER, "proof": "AAAAAAAAAA"})
        assert a.status_code == b.status_code == 401 and a.json() == b.json()

    # Approver role not mintable over http.
    def test_approver_role_not_mintable_over_http(self, client):
        r = client.post("/token", json={"order_id": ORDER, "proof": auth.order_proof(ORDER), "role": "approver"})
        assert auth.token_claims(r.json()["token"])["role"] == "customer"

    # Token rate limited.
    def test_token_rate_limited(self, client):
        codes = [client.post("/token", json={"order_id": ORDER, "proof": "AAAAAAAAAA"}).status_code
                 for _ in range(6)]
        assert codes[:5] == [401] * 5 and codes[5] == 429

    # Bad order id format is 422.
    def test_bad_order_id_format_is_422(self, client):
        assert client.post("/token", json={"order_id": "1000; drop", "proof": "x"}).status_code == 422

    # Missing token is 422.
    def test_missing_token_is_422(self, client):
        assert client.post("/resolve", json={"message": "hi"}).status_code == 422

    # Message too long is 422.
    def test_message_too_long_is_422(self, client):
        token = auth.issue_token("cust-1")
        assert client.post("/resolve", json={"message": "x" * 4001, "token": token}).status_code == 422

    # Empty message is 422.
    def test_empty_message_is_422(self, client):
        assert client.post("/resolve", json={"message": "", "token": auth.issue_token("c")}).status_code == 422

    # Invalid and expired tokens rejected before any agent call.
    def test_invalid_and_expired_tokens_rejected_before_any_agent_call(self, client):
        assert client.post("/resolve", json={"message": "hi", "token": "forged.garbage"}).status_code == 401
        expired = auth.issue_token("cust-1", ttl_seconds=-1)
        assert client.post("/resolve", json={"message": "hi", "token": expired}).status_code == 401


# Auth secrets fail closed outside dev.
class TestFailClosed:
    # Reload resolv.api.auth under the given env; restore the dev module afterwards.
    @pytest.fixture
    def reload_auth(self, monkeypatch):
        # Apply env and reload.
        def _reload(**env):
            for name, value in env.items():
                if value is None:
                    monkeypatch.delenv(name, raising=False)
                else:
                    monkeypatch.setenv(name, value)
            return importlib.reload(auth)
        yield _reload
        monkeypatch.undo()
        importlib.reload(auth)

    # Missing secret fails closed in prod.
    def test_missing_secret_fails_closed_in_prod(self, reload_auth):
        with pytest.raises(RuntimeError):
            reload_auth(RESOLV_ENV="prod", AUTH_SECRET=None, ORDER_PROOF_SECRET="p" * 40)

    # Short secret rejected in prod.
    def test_short_secret_rejected_in_prod(self, reload_auth):
        with pytest.raises(RuntimeError):
            reload_auth(RESOLV_ENV="prod", AUTH_SECRET="short", ORDER_PROOF_SECRET="p" * 40)

    # Missing proof secret fails closed in prod.
    def test_missing_proof_secret_fails_closed_in_prod(self, reload_auth):
        with pytest.raises(RuntimeError):
            reload_auth(RESOLV_ENV="prod", AUTH_SECRET="a" * 40, ORDER_PROOF_SECRET=None)

    # Dev falls back to random secret.
    def test_dev_falls_back_to_random_secret(self, reload_auth):
        module = reload_auth(RESOLV_ENV="dev", AUTH_SECRET=None, ORDER_PROOF_SECRET=None)
        assert module.verify_token(module.issue_token("c")) == "c"
