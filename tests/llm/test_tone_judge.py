"""
# tests/llm/test_tone_judge.py

**What:** LLM-quality checks of the tone judge with DeepEval: a warm, clear refusal must pass and a curt,
rule-reciting one must fail, under the same GEval metric `evals/quality.py` uses.
**Why:** the tone number is only worth publishing if the judge separates obvious cases. Needs a provider key, so
it is skipped in keyless CI.
**Used by:** pytest with GEMINI_API_KEY or GROQ_API_KEY set.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GROQ_API_KEY"),
                                reason="needs a provider key")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")  # before deepeval loads, as in evals/quality.py
deepeval = pytest.importorskip("deepeval")

WARM = ("I'm sorry your order arrived late. I've refunded $12.50, which is the most we can offer for a late "
        "delivery on this order, and you'll see it in 3-5 business days. Anything else I can help with?")
CURT = "Per policy I cannot override the system. Request denied."


# the tone metric, built once
@pytest.fixture(scope="module")
def metric():
    from evals.quality import _metric
    return _metric()


# a warm refusal clears the threshold
def test_warm_reply_passes(metric):
    from deepeval.test_case import LLMTestCase
    metric.measure(LLMTestCase(input="", actual_output=WARM, context=["This is the agent's only reply."]),
                   _show_indicator=False)
    assert metric.score >= metric.threshold


# a curt, rule-reciting refusal does not
def test_curt_reply_fails(metric):
    from deepeval.test_case import LLMTestCase
    metric.measure(LLMTestCase(input="", actual_output=CURT, context=["This is the agent's only reply."]),
                   _show_indicator=False)
    assert metric.score < metric.threshold
