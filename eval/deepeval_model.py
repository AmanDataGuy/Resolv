"""
DeepEval's custom-model hook — lets eval/quality.py's GEval metric run its judge calls
through THIS project's own provider (Groq/Gemini via litellm), instead of DeepEval's
OpenAI default. Nothing else in eval/ imports this; it exists solely to plug
agents/runner_utils.py::complete() into DeepEval's DeepEvalBaseLLM interface.
"""
from deepeval.models import DeepEvalBaseLLM

from agents.runner_utils import complete
from config import MODEL


class ResolvJudge(DeepEvalBaseLLM):
    def load_model(self):
        return MODEL

    def generate(self, prompt: str) -> str:
        resp = complete(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)  # complete() is sync; DeepEval accepts a sync a_generate

    def get_model_name(self) -> str:
        return MODEL
