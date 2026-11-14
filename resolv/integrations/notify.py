"""
# resolv/integrations/notify.py

**What:** the outbound channel, mocked: `send(ticket)` emails the customer and pages a team as files under
data/outbox/, and `drain_outbox()` delivers the ledger's outbox rows the same way.
**Why:** real delivery (SMTP, a ticketing API) proves nothing about whether the agent and harness decide correctly,
so only the transport is faked; swap these bodies for SES/Zendesk without changing callers. The ledger writes an
outbox row in the same transaction as the money change, so a message goes out if and only if the change committed;
the drain is at-least-once and keyed by outbox id, so a re-delivery overwrites instead of duplicating.
**Used by:** resolv/api/main.py, app.py, resolv/api/escalations.py.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from resolv.harness import ledger
from resolv.schemas import Ticket

OUTBOX = Path(__file__).resolve().parents[2] / "data" / "outbox"


# Deliver a ticket: always email the customer; page a team only if the case was escalated.
def send(ticket: Ticket) -> dict:
    customer_dir = OUTBOX / "customer"
    customer_dir.mkdir(parents=True, exist_ok=True)
    customer_path = customer_dir / f"{ticket.ticket_id}.txt"  # keyed by ticket, so a re-send overwrites
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
        team_path = team_dir / f"{ticket.team}.jsonl"  # a queue is a log: re-pages append
        card = {"ts": datetime.now(timezone.utc).isoformat(), "ticket_id": ticket.ticket_id,
                "case_id": ticket.case_id, "order_id": ticket.order_id, "status": ticket.status}
        with open(team_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(card) + "\n")

    return {"customer": str(customer_path), "team": str(team_path) if team_path else None}


# Deliver every unsent ledger outbox row as a file and mark it sent; returns how many went out.
def drain_outbox() -> int:
    rows = ledger.unsent_outbox()
    for row in rows:
        path = OUTBOX / "events" / f"{row['id']}-{row['kind']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(row["payload_json"], encoding="utf-8")
        ledger.mark_sent(row["id"])
    return len(rows)


# Send one ticket of each outcome to a temp outbox and assert what landed where.
def demo() -> None:
    import tempfile

    global OUTBOX
    OUTBOX = Path(tempfile.mkdtemp(prefix="resolv-outbox-"))
    resolved = Ticket(ticket_id="TKT-DEMO01", case_id="demo-r", order_id="ORD-1000", outcome="resolved",
                      team=None, status="closed", customer_message="Refund of $12.50 approved. Reference TKT-DEMO01.")
    escalated = Ticket(ticket_id="TKT-DEMO02", case_id="demo-e", order_id="ORD-1200", outcome="escalated",
                       team="logistics", status="open", customer_message="Passed to logistics. Reference TKT-DEMO02.")

    r = send(resolved)
    assert Path(r["customer"]).exists() and r["team"] is None, "resolved: customer emailed, no team paged"
    e = send(escalated)
    last = Path(e["team"]).read_text(encoding="utf-8").strip().splitlines()[-1]
    assert json.loads(last)["ticket_id"] == "TKT-DEMO02", "escalated: the team must be paged"
    print("notify demo OK: customer emailed always, team paged only on escalation")


if __name__ == "__main__":
    demo()
