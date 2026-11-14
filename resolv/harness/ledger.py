"""
# resolv/harness/ledger.py

**What:** the SQLite ledger: the audit chain, refunds, escalations, request idempotency keys, the outbox, and the
step journal resolv/agents/loop.py replays after a crash.
**Why:** database constraints, not Python locks, enforce "one live refund per order" and "one execution per key".
`BEGIN IMMEDIATE` makes writers take turns across threads and processes, and a partial UNIQUE index blocks a second
live refund even if a rule is ever skipped. Each audit record carries an HMAC over the previous record's mac, so
anyone without AUDIT_HMAC_KEY who edits, deletes or reorders records is caught by `verify()`. Money is stored in
integer cents. Threat limit: someone with DB write access AND the key can rewrite history; it is not a WORM store.
**Used by:** resolv/harness/audit.py (facade), resolv/harness/tools.py (refund saga), resolv/api/escalations.py, resolv/integrations/notify.py,
resolv/agents/loop.py (step journal).
"""
import hashlib
import hmac
import json
import os
import sqlite3
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

GENESIS = "0" * 64  # prev_mac of the first record: there is no earlier record to chain to
LIVE = ("pending", "settled", "unknown")  # refund statuses that hold the order's one refund slot
_DEV_KEY = ("dev", b"resolv-dev-audit-key-not-secret")

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT UNIQUE NOT NULL, ts TEXT NOT NULL,
  case_id TEXT NOT NULL, tool TEXT NOT NULL, order_id TEXT, args_json TEXT NOT NULL,
  caller_id TEXT, approver_id TEXT,
  action TEXT NOT NULL CHECK (action IN ('allow','deny','escalate')),
  rule_id TEXT NOT NULL, reason TEXT, ok INTEGER NOT NULL CHECK (ok IN (0,1)), result TEXT,
  key_id TEXT NOT NULL, prev_mac TEXT NOT NULL, mac TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS audit_by_case ON audit(case_id, seq);
CREATE INDEX IF NOT EXISTS audit_by_order ON audit(order_id, seq);
CREATE TABLE IF NOT EXISTS refunds(
  refund_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, order_id TEXT NOT NULL,
  case_id TEXT NOT NULL, claim_type TEXT NOT NULL,
  amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
  status TEXT NOT NULL CHECK (status IN ('pending','settled','unknown','failed','reversed')),
  psp_ref TEXT, approval_id TEXT, audit_seq INTEGER REFERENCES audit(seq),
  attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS one_live_refund_per_order
  ON refunds(order_id) WHERE status IN ('pending','settled','unknown');
CREATE TABLE IF NOT EXISTS escalations(
  id TEXT PRIMARY KEY, case_id TEXT NOT NULL, order_id TEXT,
  kind TEXT NOT NULL CHECK (kind IN ('approval','review')),
  action_json TEXT, action_hash TEXT, policy_rule_id TEXT, reason TEXT NOT NULL, requested_by TEXT,
  status TEXT NOT NULL CHECK (status IN ('pending','approved','denied','expired','executed','failed')),
  created_at TEXT NOT NULL, expires_at TEXT NOT NULL, decided_by TEXT, decided_at TEXT, note TEXT, result TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS one_pending_per_action ON escalations(action_hash) WHERE status = 'pending';
CREATE TABLE IF NOT EXISTS idempotency(
  key TEXT PRIMARY KEY, scope TEXT NOT NULL, request_sha TEXT NOT NULL,
  response_json TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox(
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL, sent_at TEXT);
CREATE TABLE IF NOT EXISTS steps(
  case_id TEXT NOT NULL, step INTEGER NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY (case_id, step));
"""

_ready: set[str] = set()
_init_lock = threading.Lock()


# Current UTC time (optionally shifted) as ISO text; ISO strings sort in time order, which expiry queries rely on.
def now(shift_s: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=shift_s)).isoformat()


# Open the ledger file named by RESOLV_DB (read on every call, so tests can point it at a temp file).
def connect() -> sqlite3.Connection:
    path = os.environ.get("RESOLV_DB", "data/resolv.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=FULL")
    if path not in _ready:
        with _init_lock:  # one thread per process creates the schema; the rest wait, then skip it
            if path not in _ready:
                _create_schema(conn)
                _ready.add(path)
    return conn


# Create the tables inside BEGIN IMMEDIATE, so a second process doing the same waits on the busy timeout instead of
# failing with "database is locked" (an autocommit CREATE that upgrades from read to write gets no busy wait in WAL).
def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")  # cannot change inside a transaction, so it runs first
    conn.executescript(f"BEGIN IMMEDIATE;{SCHEMA}COMMIT;")


# One write transaction: BEGIN IMMEDIATE takes the write lock up front, so read-check-write cannot interleave.
@contextmanager
def tx():
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:  # BEGIN itself may have failed (busy); don't mask that error
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# Run a read on the given connection, or on a short-lived one, and return plain dicts.
def _rows(conn, sql: str, params=()) -> list[dict]:
    if conn is not None:
        return [dict(r) for r in conn.execute(sql, params)]
    c = connect()
    try:
        return [dict(r) for r in c.execute(sql, params)]
    finally:
        c.close()


# Point the ledger and the fake PSP at a fresh temp dir and mark the env as test (for the self-check demos).
def use_temp_db() -> str:
    d = tempfile.mkdtemp(prefix="resolv-demo-")
    os.environ.update(RESOLV_ENV="test", RESOLV_DB=os.path.join(d, "resolv.db"),
                      RESOLV_PSP_DB=os.path.join(d, "psp.db"))
    return d


# ---- audit chain ---------------------------------------------------------------------------------

# The signing key and its id: AUDIT_HMAC_KEY is required in prod; dev and test fall back to a fixed dev key.
def signing_key() -> tuple[str, bytes]:
    key = os.environ.get("AUDIT_HMAC_KEY")
    if key:
        return os.environ.get("AUDIT_KEY_ID", "k1"), key.encode()
    if os.environ.get("RESOLV_ENV") == "prod":
        raise RuntimeError("AUDIT_HMAC_KEY must be set when RESOLV_ENV=prod")
    return _DEV_KEY


# Every key the verifier accepts: the signing key, retired keys in AUDIT_OLD_KEYS ("id:secret,..."), dev key outside prod.
def verify_keys() -> dict[str, bytes]:
    keys = {k: v.encode() for k, v in (p.split(":", 1) for p in os.environ.get("AUDIT_OLD_KEYS", "").split(",") if ":" in p)}
    if os.environ.get("RESOLV_ENV") != "prod":
        keys[_DEV_KEY[0]] = _DEV_KEY[1]
    kid, key = signing_key()
    keys[kid] = key
    return keys


# HMAC-SHA256 over the previous mac plus this record's canonical JSON.
def _mac(key: bytes, body: dict) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hmac.new(key, (body["prev_mac"] + canonical).encode(), hashlib.sha256).hexdigest()


# Any value as TEXT-safe: None and str pass through, anything else is stored as its repr (never raises).
def _text(v):
    return v if v is None or isinstance(v, str) else repr(v)


# A stored audit row as the record shape every reader uses (args parsed, ok as bool).
def _record(row) -> dict:
    r = dict(row)
    r["args"] = json.loads(r.pop("args_json"))
    r["ok"] = bool(r["ok"])
    return r


# Append one audit record inside the caller's transaction, chained to the last record, and return it.
def append_audit(conn, case_id, tool, order_id, args, decision, ok, result, caller_id=None, approver_id=None) -> dict:
    key_id, key = signing_key()
    last = conn.execute("SELECT mac FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
    body = {
        "record_id": uuid.uuid4().hex, "ts": now(), "case_id": _text(case_id), "tool": _text(tool),
        "order_id": _text(order_id), "args_json": json.dumps(args, sort_keys=True, default=repr),
        "caller_id": _text(caller_id), "approver_id": _text(approver_id), "action": decision.action,
        "rule_id": _text(decision.rule_id), "reason": _text(decision.reason), "ok": int(bool(ok)),
        "result": _text(result), "key_id": key_id, "prev_mac": last["mac"] if last else GENESIS,
    }
    cur = conn.execute(
        f"INSERT INTO audit({','.join(body)}, mac) VALUES ({','.join('?' * len(body))}, ?)",
        (*body.values(), _mac(key, body)),
    )
    return _record(conn.execute("SELECT * FROM audit WHERE seq=?", (cur.lastrowid,)).fetchone())


# Every audit record for a case, oldest first.
def case_trail(case_id: str, conn=None) -> list[dict]:
    return [_record(r) for r in _rows(conn, "SELECT * FROM audit WHERE case_id=? ORDER BY seq", (case_id,))]


# Check the whole chain (and, given an exported head, that nothing after it was cut): (True, None) or (False, bad seq).
def verify(head: dict | None = None) -> tuple[bool, int | None]:
    keys, prev, head_seen = verify_keys(), GENESIS, head is None
    for row in _rows(None, "SELECT * FROM audit ORDER BY seq"):
        mac, seq = row.pop("mac"), row.pop("seq")
        key = keys.get(row["key_id"])
        if row["prev_mac"] != prev or key is None or not hmac.compare_digest(_mac(key, row), mac):
            return False, seq
        if head and seq == head["seq"]:
            head_seen = mac == head["mac"]
        prev = mac
    return (True, None) if head_seen else (False, head["seq"])


# The newest record's {seq, mac, ts}, for an operator to store outside the database.
def head() -> dict | None:
    rows = _rows(None, "SELECT seq, mac, ts FROM audit ORDER BY seq DESC LIMIT 1")
    return rows[0] if rows else None


# ---- refunds -------------------------------------------------------------------------------------

# Live refunds on an order, shaped like successful issue_refund records so policy rule 5 can read them.
def order_refund_records(order_id: str, conn=None) -> list[dict]:
    rows = _rows(conn, "SELECT * FROM refunds WHERE order_id=? AND status IN ('pending','settled','unknown') "
                       "ORDER BY created_at", (_text(order_id),))
    return [{"tool": "issue_refund", "order_id": r["order_id"], "ok": True, "status": r["status"],
             "args": {"amount_usd": r["amount_cents"] / 100, "claim_type": r["claim_type"]}} for r in rows]


# Reserve a pending refund; raises sqlite3.IntegrityError on a reused key or a second live refund for the order.
def insert_refund(conn, idempotency_key, order_id, case_id, claim_type, amount_cents, audit_seq, approval_id=None) -> str:
    refund_id, t = uuid.uuid4().hex, now()
    conn.execute(
        "INSERT INTO refunds(refund_id, idempotency_key, order_id, case_id, claim_type, amount_cents, status, "
        "approval_id, audit_seq, created_at, updated_at) VALUES (?,?,?,?,?,?,'pending',?,?,?,?)",
        (refund_id, idempotency_key, order_id, case_id, claim_type, amount_cents, approval_id, audit_seq, t, t),
    )
    return refund_id


# Move a refund out of pending/unknown; False if it had already been resolved elsewhere.
def set_refund_status(conn, refund_id: str, status: str, psp_ref: str | None = None) -> bool:
    cur = conn.execute(
        "UPDATE refunds SET status=?, psp_ref=COALESCE(?, psp_ref), updated_at=? "
        "WHERE refund_id=? AND status IN ('pending','unknown')", (status, psp_ref, now(), refund_id))
    return cur.rowcount == 1


# Count one more failed reconcile attempt on a refund.
def bump_attempts(conn, refund_id: str) -> None:
    conn.execute("UPDATE refunds SET attempts=attempts+1, updated_at=? WHERE refund_id=?", (now(), refund_id))


# The refund created with this idempotency key, or None.
def refund_by_key(key: str, conn=None) -> dict | None:
    rows = _rows(conn, "SELECT * FROM refunds WHERE idempotency_key=?", (key,))
    return rows[0] if rows else None


# Refunds, optionally filtered by status, oldest first (reconcile and tests).
def refunds(statuses=None, older_than_s: float = 0) -> list[dict]:
    statuses = statuses or ("pending", "settled", "unknown", "failed", "reversed")
    marks = ",".join("?" * len(statuses))
    return _rows(None, f"SELECT * FROM refunds WHERE status IN ({marks}) AND updated_at <= ? ORDER BY created_at",
                 (*statuses, now(-older_than_s)))


# ---- escalations ---------------------------------------------------------------------------------

# File an escalation; returns its id, or None when the same action already has a pending one.
def create_escalation(conn, case_id, order_id, kind, reason, requested_by, action_json=None, action_hash=None,
                      rule_id=None, ttl_s: float = 86400) -> str | None:
    esc_id = f"ESC-{uuid.uuid4().hex[:12]}"
    cur = conn.execute(
        "INSERT OR IGNORE INTO escalations(id, case_id, order_id, kind, action_json, action_hash, policy_rule_id, "
        "reason, requested_by, status, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?)",
        (esc_id, _text(case_id), _text(order_id), kind, action_json, action_hash, rule_id, _text(reason),
         _text(requested_by), now(), now(ttl_s)),
    )
    return esc_id if cur.rowcount == 1 else None


# One escalation by id, or None.
def escalation(esc_id: str, conn=None) -> dict | None:
    rows = _rows(conn, "SELECT * FROM escalations WHERE id=?", (esc_id,))
    return rows[0] if rows else None


# Escalations with a status, newest first.
def escalations(status: str = "pending", limit: int = 50) -> list[dict]:
    return _rows(None, "SELECT * FROM escalations WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit))


# Mark pending escalations past their expiry as expired (auto-reject).
def expire_escalations() -> None:
    with tx() as conn:
        conn.execute("UPDATE escalations SET status='expired' WHERE status='pending' AND expires_at <= ?", (now(),))


# A human decision on a pending, unexpired escalation; False if it was not pending any more.
def decide_escalation(conn, esc_id: str, status: str, decided_by: str, note: str) -> bool:
    cur = conn.execute(
        "UPDATE escalations SET status=?, decided_by=?, decided_at=?, note=? "
        "WHERE id=? AND status='pending' AND expires_at > ?", (status, decided_by, now(), note, esc_id, now()))
    return cur.rowcount == 1


# Move an escalation from one status to another; False if it was not in from_status.
def move_escalation(conn, esc_id: str, from_status: str, to_status: str, result: str | None = None) -> bool:
    cur = conn.execute("UPDATE escalations SET status=?, result=COALESCE(?, result) WHERE id=? AND status=?",
                       (to_status, result, esc_id, from_status))
    return cur.rowcount == 1


# ---- request idempotency keys (Stripe convention, 24 h) ------------------------------------------

# Claim a key: ("new", None) to run, ("replay", saved), ("in_flight", None) or ("conflict", None) for another body.
def claim_key(key: str, scope: str, request_sha: str) -> tuple[str, dict | None]:
    with tx() as conn:
        conn.execute("DELETE FROM idempotency WHERE created_at < ?", (now(-86400),))
        row = conn.execute("SELECT * FROM idempotency WHERE key=?", (key,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO idempotency(key, scope, request_sha, created_at) VALUES (?,?,?,?)",
                         (key, scope, request_sha, now()))
            return "new", None
    if row["scope"] != scope or row["request_sha"] != request_sha:
        return "conflict", None
    if row["response_json"] is None:
        return "in_flight", None
    return "replay", json.loads(row["response_json"])


# Store the response for a claimed key, or release the key (None) so a failed request can be retried.
def finish_key(key: str, response: dict | None) -> None:
    with tx() as conn:
        if response is None:
            conn.execute("DELETE FROM idempotency WHERE key=?", (key,))
        else:
            conn.execute("UPDATE idempotency SET response_json=? WHERE key=?", (json.dumps(response, default=str), key))


# ---- outbox --------------------------------------------------------------------------------------

# Queue a message in the caller's transaction, so it is sent if and only if the change commits.
def add_outbox(conn, kind: str, payload: dict) -> None:
    conn.execute("INSERT INTO outbox(kind, payload_json, created_at) VALUES (?,?,?)",
                 (kind, json.dumps(payload, default=str), now()))


# Outbox rows not yet delivered, oldest first.
def unsent_outbox() -> list[dict]:
    return _rows(None, "SELECT * FROM outbox WHERE sent_at IS NULL ORDER BY id")


# Mark one outbox row delivered.
def mark_sent(outbox_id: int) -> None:
    with tx() as conn:
        conn.execute("UPDATE outbox SET sent_at=? WHERE id=?", (now(), outbox_id))


# ---- step journal (durable execution for resolv/agents/loop.py) -----------------------------------------

# Journal one completed step of a resolution; a step already journaled (a replayed write) is left as it was.
def add_step(case_id: str, step: int, kind: str, payload) -> None:
    with tx() as conn:
        conn.execute("INSERT OR IGNORE INTO steps(case_id, step, kind, payload_json, created_at) VALUES (?,?,?,?,?)",
                     (case_id, step, kind, json.dumps(payload, default=str), now()))


# A case's journaled step payloads, in step order ([] for a case that never started).
def steps(case_id: str) -> list:
    return [json.loads(r["payload_json"])
            for r in _rows(None, "SELECT payload_json FROM steps WHERE case_id=? ORDER BY step", (case_id,))]


# ---- test and eval resets (guarded in resolv/harness/audit.py) ------------------------------------------

# Delete a case's audit records, refunds, escalations and step journal. Breaks the global chain; test and eval only.
def clear_case(case_id: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM steps WHERE case_id=?", (case_id,))
        conn.execute("DELETE FROM refunds WHERE case_id=?", (case_id,))
        conn.execute("DELETE FROM escalations WHERE case_id=?", (case_id,))
        conn.execute("DELETE FROM audit WHERE case_id=?", (case_id,))


# Delete every refund on an order, so the next eval repeat starts with the order unrefunded.
def clear_order(order_id: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM refunds WHERE order_id=?", (order_id,))
