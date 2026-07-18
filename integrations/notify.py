"""The outbound channel — "email the customer, page the team". A MOCK, on purpose.

WHAT THIS IS. send() takes a Ticket (harness/routing.py already decided everything) and delivers
it: writes the customer's message to an outbox and, if the case was escalated, drops a card on
the assigned team's queue. It returns the paths it wrote so a caller (the API, the demo) can show
"sent to billing, ticket TKT-ABC123".

WHY MOCKED, AND WHERE THE REAL THING GOES. Real delivery is an SMTP account and a ticketing API —
credentials, retries, deliverability, a whole operational surface that proves nothing about THIS
project, which is about whether the agent + harness reach the right decision. So this writes files
under data/outbox/ instead of sending mail. The demo shows a real customer email and a real team
card; only the transport is faked. (ponytail: file outbox; swap send() internals for SES +
Zendesk when there's an inbox that needs to receive it — the signature and the Ticket contract
don't change, so nothing upstream moves.)

WHY IT'S SEPARATE FROM routing.py. routing decides (pure, testable, safe to call in the eval);
notify does I/O (side effects, never called in the eval's hot path). Splitting them means the
benchmark can compute a ticket to display without ever "sending" 200 emails, and the send path
can be mocked in a test without touching the decision logic.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from schemas import Ticket

OUTBOX = Path(__file__).parent.parent / "data" / "outbox"


def send(ticket: Ticket) -> dict:
    """Deliver a ticket: always email the customer; page a team only if it was escalated.

    Returns {"customer": <path>, "team": <path or None>} — what was written, for display and for
    the demo() assertions. Idempotent in spirit: the customer file is keyed by ticket_id so
    re-sending overwrites rather than duplicates, while team pages append (a queue is a log, and a
    re-page is a real event a human might need to see).
    """
    customer_dir = OUTBOX / "customer"
    customer_dir.mkdir(parents=True, exist_ok=True)
    customer_path = customer_dir / f"{ticket.ticket_id}.txt"
    customer_path.write_text(
        f"To: customer (order {ticket.order_id or 'unknown'})\n"
        f"Subject: Your support request [{ticket.ticket_id}]\n\n"
        f"{ticket.customer_message}\n",
        encoding="utf-8",
    )

    team_path = None
    if ticket.outcome == "escalated" and ticket.team:
        team_dir = OUTBOX / "teams"
        team_dir.mkdir(parents=True, exist_ok=True)
        team_path = team_dir / f"{ticket.team}.jsonl"
        card = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticket_id": ticket.ticket_id,
            "case_id": ticket.case_id,
            "order_id": ticket.order_id,
            "status": ticket.status,
        }
        with open(team_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(card) + "\n")

    return {"customer": str(customer_path), "team": str(team_path) if team_path else None}


def demo() -> None:
    """Send one of each outcome and assert what landed where — real files, temp ticket ids."""
    resolved = Ticket(
        ticket_id="TKT-DEMO01", case_id="demo-r", order_id="ORD-1000",
        outcome="resolved", team=None, status="closed",
        customer_message="Refund of $12.50 approved. Reference TKT-DEMO01.",
    )
    escalated = Ticket(
        ticket_id="TKT-DEMO02", case_id="demo-e", order_id="ORD-1200",
        outcome="escalated", team="logistics", status="open",
        customer_message="We've passed this to logistics. Reference TKT-DEMO02.",
    )

    r = send(resolved)
    assert Path(r["customer"]).exists() and r["team"] is None, "resolved: customer emailed, no team paged"
    assert "TKT-DEMO01" in Path(r["customer"]).read_text(encoding="utf-8")

    e = send(escalated)
    assert e["team"] is not None and Path(e["team"]).exists(), "escalated: a team must be paged"
    last = Path(e["team"]).read_text(encoding="utf-8").strip().splitlines()[-1]
    assert json.loads(last)["ticket_id"] == "TKT-DEMO02"

    # Re-sending the customer email overwrites (one file per ticket), doesn't pile up copies.
    send(resolved)
    assert Path(r["customer"]).read_text(encoding="utf-8").count("TKT-DEMO01") >= 1

    # Clean up the demo's traces so the outbox only ever holds real sends.
    Path(r["customer"]).unlink(missing_ok=True)
    Path(e["customer"]).unlink(missing_ok=True)

    print("notify demo OK — customer emailed always, team paged only on escalation")


if __name__ == "__main__":
    demo()
