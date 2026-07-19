"""The web API for Resolv — one HTTP endpoint that runs a customer complaint end to end.

Run it locally with:

    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser, or POST to /resolve.

WHAT THIS FILE IS. This is the thin "front door" of the system. It has NO business logic of its
own — it just wires four steps together in order:

    1. run_case()       the agent reads the complaint and calls tools   (agents/loop.py)
    2. routing.route()  turn the finished case into a ticket + team      (harness/routing.py)
    3. notify.send()    "email" the customer and page the team           (integrations/notify.py)
    4. return           hand the whole thing back as JSON

Every real decision (was the refund allowed? how much? which team?) is made and enforced in those
modules, not here. Keeping the endpoint dumb is the point: there is nothing here to get wrong, so
the API can never approve something the harness wouldn't.
"""
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from agents.loop import run_case
from harness import routing
from integrations import notify

# Create the app. The title/description show up on the auto-generated /docs page.
app = FastAPI(
    title="Resolv",
    description="A customer-refund agent that can only act inside a policy harness.",
)


# The shape of the request body. FastAPI reads the incoming JSON and checks it against this
# automatically — if "message" is missing, the caller gets a clear 422 error instead of a crash.
class Complaint(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    """A tiny endpoint to check the server is alive — handy for deploys and uptime checks."""
    return {"status": "ok"}


@app.post("/resolve")
async def resolve(complaint: Complaint) -> dict:
    """Resolve one complaint and return everything that happened.

    It's `async def` because run_case() is asynchronous (the agent awaits the language model).
    FastAPI handles the await for us, so a caller just POSTs and waits for the JSON back.
    """
    # A unique id for this case. It names the audit trail this run reads and writes, so two
    # customers never share history. Generated here, never taken from the caller.
    case_id = f"api-{uuid.uuid4().hex[:8]}"

    # Step 1 — run the agent. temperature=0.0 makes it deterministic: the same complaint gives the
    # same answer. (The eval uses 0.7 to get varied samples; a live customer wants the single most
    # reliable answer.)
    result = await run_case(case_id, complaint.message, temperature=0.0)

    # Step 2 — read the finished audit trail and decide the ticket (refund / escalate / deny) and,
    # if escalated, which team. Pure and deterministic — no second call to the model.
    ticket = routing.route(case_id, result["trail"], result["claim"])

    # Step 3 — deliver it: write the customer email and, if escalated, page the team.
    # (Mocked to files under data/outbox/ — see integrations/notify.py.)
    notify.send(ticket)

    # Step 4 — return everything. `trail` is included on purpose: it's the evidence for what the
    # agent actually did, so a caller can trust the record instead of the prose reply.
    return {
        "case_id": case_id,
        "reply": result["reply"],       # what to say to the customer, in plain language
        "claim": result["claim"],       # what intake read the message as (order id + claim type)
        "ticket": ticket.model_dump(),  # outcome, team, ticket id, customer message
        "trail": result["trail"],       # every tool call and the policy verdict on each
    }


def _selfcheck() -> None:
    """One no-network check: the app builds and /health answers. The /resolve path needs a live
    model, so it's exercised by the demo and the eval, not here. Run: python -m api.main
    """
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    print("api selfcheck OK — /health responds")


if __name__ == "__main__":
    _selfcheck()
