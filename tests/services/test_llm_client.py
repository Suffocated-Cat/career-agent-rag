"""Tests for LLMClient — OpenAI-compatible chat wrapper (no network)."""
from app.services.llm_client import LLMClient


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _Usage:
    def __init__(self, prompt, completion, total):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class FakeOpenAI:
    """Stand-in for the OpenAI SDK client, recording the call."""

    def __init__(self, reply="hello", usage=None, **kwargs):
        self.reply = reply
        self.usage = usage
        self.init_kwargs = kwargs
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "Resp", (), {"choices": [_Msg(self.reply)], "usage": self.usage}
        )()


class TestIsConfigured:
    # Set attributes directly so the test is independent of the environment's
    # settings fallback (LLM_API_KEY / LLM_MODEL may be set in .env).
    def test_true_with_key_and_model(self):
        c = LLMClient(client=object())
        c.api_key, c.model = "k", "m"
        assert c.is_configured()

    def test_false_without_key(self):
        c = LLMClient(client=object())
        c.api_key, c.model = None, "m"
        assert not c.is_configured()

    def test_false_without_model(self):
        c = LLMClient(client=object())
        c.api_key, c.model = "k", None
        assert not c.is_configured()


class TestComplete:
    def test_returns_message_content(self):
        fake = FakeOpenAI(reply="the answer")
        llm = LLMClient(api_key="k", model="m", client=fake)
        assert llm.complete("q") == "the answer"

    def test_passes_model_and_messages(self):
        fake = FakeOpenAI()
        llm = LLMClient(api_key="k", model="my-model", client=fake)
        llm.complete("question", system="be terse")
        call = fake.calls[0]
        assert call["model"] == "my-model"
        assert call["messages"][0] == {"role": "system", "content": "be terse"}
        assert call["messages"][1] == {"role": "user", "content": "question"}

    def test_omits_system_when_absent(self):
        fake = FakeOpenAI()
        llm = LLMClient(api_key="k", model="m", client=fake)
        llm.complete("just user")
        assert fake.calls[0]["messages"] == [{"role": "user", "content": "just user"}]

    def test_default_temperature_zero(self):
        fake = FakeOpenAI()
        LLMClient(api_key="k", model="m", client=fake).complete("q")
        assert fake.calls[0]["temperature"] == 0.0


class TestUsageCapture:
    def test_records_last_usage(self):
        fake = FakeOpenAI(usage=_Usage(10, 5, 15))
        llm = LLMClient(api_key="k", model="m", client=fake)
        llm.complete("q")
        assert llm.last_usage.prompt_tokens == 10
        assert llm.last_usage.completion_tokens == 5
        assert llm.last_usage.total_tokens == 15

    def test_no_usage_leaves_last_usage_none(self):
        llm = LLMClient(api_key="k", model="m", client=FakeOpenAI(usage=None))
        llm.complete("q")
        assert llm.last_usage is None

    def test_feeds_usage_tracker(self):
        from app.services.usage import UsageTracker

        tracker = UsageTracker()
        fake = FakeOpenAI(usage=_Usage(10, 5, 15))
        llm = LLMClient(api_key="k", model="m", client=fake, usage_tracker=tracker)
        llm.complete("q")
        llm.complete("q")
        assert tracker.calls == 2
        assert tracker.total_tokens == 30


class TestLazyClient:
    def test_lazy_construct_uses_settings_and_caches(self, monkeypatch):
        import openai

        built = []

        def _factory(**kwargs):
            built.append(kwargs)
            return FakeOpenAI(**kwargs)

        monkeypatch.setattr(openai, "OpenAI", _factory)

        llm = LLMClient(api_key="key123", base_url="http://x/v1", model="m")
        c1 = llm._get_client()
        c2 = llm._get_client()
        assert c1 is c2  # cached
        assert len(built) == 1
        assert built[0] == {"api_key": "key123", "base_url": "http://x/v1"}
