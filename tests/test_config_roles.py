"""
# tests/test_config_roles.py

**What:** resolv/config.py role models, the selfhosted provider, llm_kwargs, price_usd, and one key per provider.
**Why:** a role env var must move only its own role, self-hosted models must route to SELFHOSTED_BASE_URL at $0,
and multi-key rotation must stay gone.
**Used by:** pytest.
"""
import importlib

import pytest

from resolv import config
# Reload config with a clean set of provider/role env vars plus the given overrides.
@pytest.fixture
def load(monkeypatch):
    # Set env, reload, return the fresh module.
    def _load(**env):
        for name in ("LLM_PROVIDER", "LLM_MODEL", "AGENT_MODEL", "EXTRACTOR_MODEL", "SIMULATOR_MODEL",
                     "JUDGE_MODEL", "OPENROUTER_API_KEY", "SELFHOSTED_BASE_URL", "SELFHOSTED_MODEL"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        return importlib.reload(config)
    yield _load
    monkeypatch.undo()
    importlib.reload(config)


# All roles default to the provider default.
def test_all_roles_default_to_the_provider_default(load):
    c = load(LLM_PROVIDER="groq")
    assert c.AGENT_MODEL == c.EXTRACTOR_MODEL == c.SIMULATOR_MODEL == c.JUDGE_MODEL == "groq/openai/gpt-oss-120b"


# One role override changes only that role.
def test_one_role_override_changes_only_that_role(load):
    c = load(LLM_PROVIDER="groq", EXTRACTOR_MODEL="selfhosted:grpo")
    assert c.EXTRACTOR_MODEL == "hosted_vllm/grpo"
    assert c.AGENT_MODEL == c.SIMULATOR_MODEL == c.JUDGE_MODEL == "groq/openai/gpt-oss-120b"


# Selfhosted provider default.
def test_selfhosted_provider_default(load):
    c = load(LLM_PROVIDER="selfhosted", SELFHOSTED_MODEL="qwen-test", SELFHOSTED_BASE_URL="http://h:9/v1")
    assert c.AGENT_MODEL == "hosted_vllm/qwen-test"
    assert c.llm_kwargs(c.AGENT_MODEL) == {"api_base": "http://h:9/v1", "api_key": "local"}


# Hosted models get no extra kwargs.
def test_hosted_models_get_no_extra_kwargs():
    assert config.llm_kwargs("groq/openai/gpt-oss-120b") == {}


# price_usd for known models.
def test_price_usd_known_models():
    assert config.price_usd("groq/openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert config.price_usd("gemini/gemini-3.6-flash", 1_000_000, 0) == pytest.approx(0.75)
    assert config.price_usd("hosted_vllm/anything", 10**9, 10**9) == 0.0


# Unknown model is unpriced not gemini.
def test_unknown_model_is_unpriced_not_gemini():
    assert not config.is_priced("someprovider/unknown-model")
    assert config.price_usd("someprovider/unknown-model", 1_000_000, 1_000_000) == 0.0
    assert config.is_priced("hosted_vllm/x") and config.is_priced("openrouter/a/b:free")


# Single key config.
def test_single_key_config():
    assert not hasattr(config, "rotate_key") and not hasattr(config, "reset_key")
    assert not hasattr(config, "MODEL")


# Unknown provider rejected.
def test_unknown_provider_rejected(load):
    with pytest.raises(RuntimeError):
        load(LLM_PROVIDER="nope")


# Extractor get model routes selfhosted.
def test_extractor_get_model_routes_selfhosted(load, monkeypatch):
    c = load(LLM_PROVIDER="groq", EXTRACTOR_MODEL="selfhosted:grpo")
    monkeypatch.delenv("USE_GEMINI", raising=False)
    model = c.get_model("extractor")
    assert model.model == "hosted_vllm/grpo"
    assert model._additional_args["api_base"] == c.SELFHOSTED_BASE_URL
