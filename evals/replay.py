"""
# evals/replay.py

**What:** replays a recorded trajectory ("cassette": the model's responses, the extractor's claim and the
customer's turns for one task) through the real agent loop, harness and grader, with no network and no key.
**Why:** the PR gate must prove, keylessly, that loop + harness + grader still behave identically on recorded model
outputs. A cassette stores the prompt hash it was recorded under; if the prompt changed, the replay refuses rather
than pass on stale responses.
**Used by:** tests/test_regression_replay.py, scripts/record_cassettes.py.
"""
import asyncio
import json
from pathlib import Path

CASSETTES = Path(__file__).parent / "regression" / "cassettes"
# fields of the graded row a cassette pins
PINNED = ("resolved", "unauthorized", "rules_hit", "paid")


# every cassette file, sorted
def cassettes() -> list[Path]:
    return sorted(CASSETTES.glob("*.json")) if CASSETTES.exists() else []


# run one cassette through run_case with the model, extractor and customer stubbed from the recording
def replay(cassette: dict, monkeypatch) -> dict:
    import litellm

    from resolv.agents import loop, llm
    from evals.runner import _grade
    from resolv.harness import audit

    responses = list(cassette["responses"])
    turns = list(cassette.get("customer_turns") or [])  # a single-turn recording stores None

    # the next recorded model response, shaped like litellm's ModelResponse
    async def fake_acomplete(**kwargs):
        if not responses:
            raise RuntimeError("cassette exhausted: the loop asked for more model calls than were recorded")
        return litellm.ModelResponse(**responses.pop(0))

    # the recorded extractor read
    async def fake_extract(message):
        return cassette.get("claim") or {}

    # the recorded customer turn, or None when the recording ended
    def fake_user(agent_said, history):
        return turns.pop(0) if turns else None

    for mod in (loop, llm):
        monkeypatch.setattr(mod, "acomplete", fake_acomplete, raising=False)
    monkeypatch.setattr(loop, "extract", fake_extract)

    task = cassette["task"]
    case_id = f"replay-{task['task_id']}"
    audit.clear(case_id)
    audit.clear_order(task["order_id"])
    user = fake_user if cassette.get("customer_turns") is not None else None
    result = asyncio.run(loop.run_case(case_id, cassette["opening"], temperature=0.0, user=user,
                                       caller_id=task["customer_id"]))
    return _grade(task, result["trail"], result["steps"], result["reply"])


# read a cassette file
def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
