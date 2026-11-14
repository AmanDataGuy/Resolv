"""
# evals/deepeval_model.py

**What:** DeepEval's custom-model hook, routing judge calls through this project's litellm `complete()` with
`JUDGE_MODEL`.
**Why:** DeepEval defaults to OpenAI; the judge must be the pinned, cross-family `JUDGE_MODEL` so a cached score
always says which model produced it.
**Used by:** evals/quality.py, tests/llm/.
"""
from deepeval.models import DeepEvalBaseLLM

from resolv import config
from resolv.agents.llm import complete


# DeepEval judge backed by config.JUDGE_MODEL at temperature 0
class ResolvJudge(DeepEvalBaseLLM):
    # the model string DeepEval reports
    def load_model(self):
        return config.JUDGE_MODEL

    # one judge call
    def generate(self, prompt: str) -> str:
        resp = complete(model=config.JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return resp.choices[0].message.content or ""

    # DeepEval accepts a sync body for the async hook
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # name shown in DeepEval output
    def get_model_name(self) -> str:
        return config.JUDGE_MODEL
