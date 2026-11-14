"""
# scripts/freeze_openings.py

**What:** writes `evals/fixtures/openings.json`: the opening message of each of the 40 benchmark tasks, read from the
local simulator cache exactly as the old `simulator.opening()` looked it up. Calls no model.
**Why:** every adversarial opening was generated once (2026-07-17, by groq/llama-3.3-70b-versatile) and silently
reused by later sweeps. Freezing them makes that deliberate and versioned. If any opening is missing from the cache
the script stops: regenerating one is a decision for the owner, not a side effect.
**Used by:** run by hand: `python scripts/freeze_openings.py [--cache DIR]`.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RESOLV_ENV", "eval")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import simulator  # noqa: E402
from evals.tasks import build_tasks  # noqa: E402

ARCHIVE = Path("D:/projects/Resolv_archive/2026-11-02/data/cache/simulator")
GENERATED_BY = "groq/llama-3.3-70b-versatile"


# the cache key the simulator used when the openings were generated (messages only, no model)
def _legacy_key(task: dict) -> str:
    messages = [{"role": "system", "content": simulator._persona(task)},
                {"role": "user", "content": simulator.OPENING_PROMPT}]
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


# build the fixture from the cache, or stop listing the tasks whose opening is missing
def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the benchmark openings into a committed fixture.")
    ap.add_argument("--cache", default=str(simulator.CACHE if simulator.CACHE.exists() else ARCHIVE))
    ap.add_argument("--tasks", type=int, default=40)
    args = ap.parse_args()
    cache = Path(args.cache)

    fixture, missing = {}, []
    for task in build_tasks(args.tasks):
        if task["tactic"] == "honest":
            fixture[task["task_id"]] = {"text": task["message"], "tactic": "honest", "source": "dataset",
                                        "generated_by": None, "generated_on": None}
            continue
        path = cache / f"{_legacy_key(task)}.txt"
        if not path.exists():
            missing.append(task["task_id"])
            continue
        made = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat()
        fixture[task["task_id"]] = {"text": path.read_text(encoding="utf-8"), "tactic": task["tactic"],
                                    "source": "cache", "generated_by": GENERATED_BY, "generated_on": made}
    if missing:
        raise SystemExit(f"{len(missing)} openings are not in {cache}: {missing}. Ask the owner before regenerating.")

    simulator.FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    simulator.FIXTURE.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(fixture)} openings -> {simulator.FIXTURE}")


if __name__ == "__main__":
    main()
