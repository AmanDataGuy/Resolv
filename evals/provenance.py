"""
# evals/provenance.py

**What:** the run manifest (git sha, models per role, prompt/dataset/openings hashes, seed, timestamps) and the
per-row provenance fields (served model, tokens, provider-aware usd, error kind), plus atomic JSONL/JSON writes.
**Why:** a result row must say exactly what produced it. Without the served model and the prompt hash, two sweeps
that look comparable can silently differ (a provider alias routed to a new model, a prompt edit). Writes go to a
temp file and `os.replace`, so a crash mid-write never truncates rows that back a published number.
**Used by:** evals/runner.py, injection.py, ablation.py, extractor.py, cost.py, latency.py, reliability.py.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from resolv import config
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
DATASET = ROOT / "data" / "datasets" / "complaint_cases.json"

# error kinds that mean "the provider said no", not "the agent failed"
INFRA_ERRORS = ("quota", "rate_limit")


# short sha256 of a string, the hash length every row uses
def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# sha of a file's bytes, or None when the file is absent
def file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


# UTC timestamp in ISO form
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# the open weights behind a litellm model string ("groq/openai/gpt-oss-120b" -> "openai/gpt-oss-120b")
def weights_of(model: str) -> str:
    return (model or "").split("/", 1)[-1]


# provider name from a litellm model string ("groq/openai/gpt-oss-120b" -> "groq")
def provider_of(model: str) -> str:
    prefix = (model or "").split("/")[0]
    return "selfhosted" if prefix == "hosted_vllm" else prefix


# current git commit and whether the working tree has uncommitted changes
def _git() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, timeout=5).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


# hash of the agent's system prompt and tool schemas; changes whenever the agent's instructions change
def prompt_sha() -> str:
    from resolv.agents.loop import SYSTEM, TOOL_SCHEMAS
    return sha12(SYSTEM + json.dumps(TOOL_SCHEMAS, sort_keys=True))


# the manifest for one run; merged into every row so each row is self-describing after a resume
def manifest(name: str, temperature: float | None = None, seed: int = 0) -> dict:
    from evals.simulator import openings_sha
    sha, dirty = _git()
    started = now()
    return {
        "run_id": f"{started.replace(':', '-')}-{provider_of(config.AGENT_MODEL)}-{name}",
        "git_sha": sha,
        "git_dirty": dirty,
        "provider": provider_of(config.AGENT_MODEL),
        "agent_model": config.AGENT_MODEL,
        "extractor_model": config.EXTRACTOR_MODEL,
        "simulator_model": config.SIMULATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "prompt_sha": prompt_sha(),
        "dataset_sha": file_sha(DATASET),
        "openings_sha": openings_sha(),
        "temperature": temperature,
        "seed": seed,
        "started_at": started,
    }


# classify an exception so infra failures can be reported apart from agent failures
def error_kind(e: BaseException) -> str:
    text = f"{type(e).__name__} {e}".lower()
    if "quota" in text or "resource_exhausted" in text or "resourceexhausted" in text:
        return "quota"
    if "rate limit" in text or "ratelimit" in text or "429" in text or "too many requests" in text:
        return "rate_limit"
    if isinstance(e, TimeoutError) or "timeout" in text or "timed out" in text:
        return "timeout"
    if type(e).__module__.split(".")[0] in ("litellm", "openai", "httpx", "google"):
        return "provider_error"
    return "agent_exception"


# provenance fields for one finished run, from the context-local token meter
def usage_fields(model: str, prompt_tokens: int, completion_tokens: int, served: str | None) -> dict:
    priced = getattr(config, "is_priced", lambda m: True)(model)
    usd = config.price_usd(model, prompt_tokens, completion_tokens) if priced else None
    return {
        "usd_basis": model if priced else "unpriced",
        "requested_model": model,
        "served_model": served,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "tokens": prompt_tokens + completion_tokens,
        "usd": None if usd is None else round(usd, 6),
        "finished_at": now(),
    }


# read a JSONL file, or the "rows" of a pinned JSON baseline
def read_rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))["rows"]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# write text to a temp file, fsync it, then swap it in, so readers never see a half-written file
def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# atomically write rows as JSONL
def write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write(path, "".join(json.dumps(r) + "\n" for r in rows))


# atomically write one JSON document (cards, baselines)
def write_json(path: Path, obj) -> None:
    _atomic_write(path, json.dumps(obj, indent=2) + "\n")
