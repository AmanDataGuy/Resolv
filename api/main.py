"""FastAPI app — the HTTP entry point.

A single webhook receives a raw exception event and runs it straight through
agents.orchestrator.process_exception() in-process. This is synchronous by
design for a self-contained deployment: no message queue sits between the webhook
and the pipeline. A queue (Pub/Sub, SQS) would be the swap-in if webhook delivery
ever needs to be decoupled from processing — the handler body is the only thing
that changes. Run locally with:

    uvicorn api.main:app --reload
"""
from fastapi import FastAPI

from agents.orchestrator import process_exception
from db import get_exception, list_exceptions

app = FastAPI(title="Resolv API")


@app.post("/exceptions/webhook")
async def receive_exception(payload: dict):
    """Accepts a raw event dict and runs the full pipeline synchronously.

    Runs inline (no queue, no ack) so a single request drives detection ->
    harness -> drafting -> decision and returns the finished record. To decouple
    ingestion from processing, this handler's body would become a queue publish
    and a subscriber would call process_exception() out-of-band — nothing else moves.
    """
    record = await process_exception(payload)
    return {"ok": True, "data": record}


@app.get("/exceptions/{exception_id}")
async def get_exception_state(exception_id: str):
    record = await get_exception(exception_id)
    return {"ok": True, "data": record}


@app.get("/exceptions")
async def list_active_exceptions(status: str | None = None, supplier_id: str | None = None):
    records = await list_exceptions(status=status, supplier_id=supplier_id)
    return {"ok": True, "data": {"exceptions": records, "count": len(records)}}


@app.get("/health")
async def health():
    return {"ok": True}
