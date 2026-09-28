"""api/auth.py's mock session tokens, and the /token + /resolve gate that uses them.

No network, no API key: the token layer is pure stdlib crypto, and /resolve's auth check happens
before any agent call, so the rejection paths are testable without a live model. The happy path
through /resolve still needs one (exercised by the demo and the eval, not here).
"""
import pytest

from api import auth
from api.main import app


class TestTokens:
    def test_a_valid_token_round_trips_to_its_customer_id(self):
        token = auth.issue_token("cust-1")
        assert auth.verify_token(token) == "cust-1"

    def test_a_forged_signature_is_rejected(self):
        token = auth.issue_token("cust-1")
        payload, sig = token.split(".", 1)
        forged = f"{payload}.{'0' * len(sig)}"
        assert forged != token
        assert auth.verify_token(forged) is None

    def test_an_expired_token_is_rejected(self):
        token = auth.issue_token("cust-1", ttl_seconds=-1)
        assert auth.verify_token(token) is None

    def test_a_token_just_inside_its_ttl_still_verifies(self):
        token = auth.issue_token("cust-1", ttl_seconds=60)
        assert auth.verify_token(token) == "cust-1"

    def test_garbage_input_is_rejected_not_raised(self):
        assert auth.verify_token("") is None
        assert auth.verify_token("not-a-token") is None

    def test_two_tokens_for_different_customers_are_not_interchangeable(self):
        a = auth.issue_token("cust-a")
        b = auth.issue_token("cust-b")
        assert auth.verify_token(a) == "cust-a"
        assert auth.verify_token(b) == "cust-b"
        assert auth.verify_token(a) != auth.verify_token(b)


class TestResolveGate:
    """The HTTP boundary: /resolve must reject a bad token before any agent call happens."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_token_endpoint_issues_a_working_token(self, client):
        r = client.post("/token", json={"customer_id": "cust-1"})
        assert r.status_code == 200
        assert auth.verify_token(r.json()["token"]) == "cust-1"

    def test_missing_token_is_a_422_not_a_401(self, client):
        """A required field missing entirely is a validation error, not an auth decision."""
        r = client.post("/resolve", json={"message": "hi"})
        assert r.status_code == 422

    def test_invalid_token_is_rejected_before_any_agent_call(self, client):
        r = client.post("/resolve", json={"message": "hi", "token": "forged.garbage"})
        assert r.status_code == 401

    def test_expired_token_is_rejected(self, client):
        expired = auth.issue_token("cust-1", ttl_seconds=-1)
        r = client.post("/resolve", json={"message": "hi", "token": expired})
        assert r.status_code == 401
