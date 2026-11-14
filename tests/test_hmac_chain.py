"""
# tests/test_hmac_chain.py

**What:** the keyed (HMAC) audit chain catches edits, deletions, reordering and truncation, rejects a wrong key,
verifies across key rotation, and fails closed in prod without a key.
**Why:** the old chain was unkeyed SHA-256, so anyone could recompute it after an edit. With a key, only someone
holding AUDIT_HMAC_KEY can forge records, and an exported head catches a cut-off tail.
**Used by:** pytest.
"""
import pytest

from resolv.harness import audit, ledger
from resolv.harness.policy import Decision

CASE = "_test_chain"


# Append n simple records and return them.
def _records(n: int) -> list[dict]:
    d = Decision(action="allow", rule_id="within_policy", reason="ok")
    return [audit.append(CASE, "lookup_order", "ORD-1000", {"i": i}, d, True, f"r{i}", "cust") for i in range(n)]


# Run one raw SQL statement against the ledger, the way an attacker with disk access would.
def _sql(sql: str, params=()) -> None:
    conn = ledger.connect()
    conn.execute(sql, params)
    conn.close()


# An intact chain verifies.
def test_intact_chain_verifies():
    _records(4)
    assert audit.verify_chain() == (True, None)


# Editing any field of a record breaks that record's mac.
def test_edit_detected():
    recs = _records(4)
    _sql("UPDATE audit SET result='edited' WHERE seq=?", (recs[1]["seq"],))
    assert audit.verify_chain() == (False, recs[1]["seq"])


# Deleting a record from the middle breaks the next record's link.
def test_delete_middle_detected():
    recs = _records(4)
    _sql("DELETE FROM audit WHERE seq=?", (recs[1]["seq"],))
    assert audit.verify_chain() == (False, recs[2]["seq"])


# Cutting off the tail leaves a valid-looking chain; the exported head catches it.
def test_truncate_detected_against_exported_head():
    _records(3)
    head = ledger.head()
    _sql("DELETE FROM audit WHERE seq=?", (head["seq"],))
    assert audit.verify_chain() == (True, None)
    assert audit.verify_chain(head=head) == (False, head["seq"])


# Swapping two records' contents breaks the chain.
def test_reorder_detected():
    recs = _records(3)
    a, b = recs[0]["seq"], recs[1]["seq"]
    _sql("UPDATE audit SET result='tmp' WHERE seq=?", (a,))
    _sql("UPDATE audit SET result=? WHERE seq=?", (recs[0]["result"], b))
    _sql("UPDATE audit SET result=? WHERE seq=?", (recs[1]["result"], a))
    ok, _ = audit.verify_chain()
    assert ok is False


# Records signed with one key do not verify under another.
def test_wrong_key_fails_verification(monkeypatch):
    monkeypatch.setenv("AUDIT_HMAC_KEY", "right-key")
    monkeypatch.setenv("AUDIT_KEY_ID", "k1")
    recs = _records(2)
    monkeypatch.setenv("AUDIT_HMAC_KEY", "wrong-key")
    assert audit.verify_chain() == (False, recs[0]["seq"])


# After rotating keys, old records verify through AUDIT_OLD_KEYS and new ones through the current key.
def test_key_rotation_verifies_across_key_ids(monkeypatch):
    monkeypatch.setenv("AUDIT_HMAC_KEY", "old-secret")
    monkeypatch.setenv("AUDIT_KEY_ID", "k1")
    _records(2)
    monkeypatch.setenv("AUDIT_HMAC_KEY", "new-secret")
    monkeypatch.setenv("AUDIT_KEY_ID", "k2")
    monkeypatch.setenv("AUDIT_OLD_KEYS", "k1:old-secret")
    recs = _records(2)
    assert recs[-1]["key_id"] == "k2"
    assert audit.verify_chain() == (True, None)


# In prod a missing key refuses to write or verify; it never falls back to the dev key.
def test_missing_key_fails_closed_in_prod(monkeypatch):
    monkeypatch.setenv("RESOLV_ENV", "prod")
    monkeypatch.delenv("AUDIT_HMAC_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _records(1)
    with pytest.raises(RuntimeError):
        audit.verify_chain()


# The dev key is not accepted in prod, so dev-signed records do not verify there.
def test_dev_signed_records_do_not_verify_in_prod(monkeypatch):
    _records(1)
    monkeypatch.setenv("RESOLV_ENV", "prod")
    monkeypatch.setenv("AUDIT_HMAC_KEY", "prod-secret")
    assert audit.verify_chain()[0] is False
