"""Tests for LLM augmentation helpers with deterministic fallback."""
import pytest
from pydantic import BaseModel

from app.services.llm_support import extract_json, generate_model, generate_text


class FakeLLM:
    def __init__(self, reply="", configured=True, raises=False, fail_first=0, replies=None):
        self.reply = reply
        self.configured = configured
        self.raises = raises
        self.fail_first = fail_first  # first N calls raise, then succeed
        self.replies = replies  # optional sequence of successive replies
        self.calls = 0
        self.prompts = []
        self.systems = []

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        self.systems.append(system)
        if self.raises:
            raise RuntimeError("api down")
        if self.calls <= self.fail_first:
            raise RuntimeError("transient failure")
        if self.replies is not None:
            return self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return self.reply


class Person(BaseModel):
    name: str
    age: int


class TestGenerateText:
    def test_returns_llm_text(self):
        assert generate_text(FakeLLM(reply="hello"), "q", fallback="fb") == "hello"

    def test_strips_whitespace(self):
        assert generate_text(FakeLLM(reply="  hi  "), "q") == "hi"

    def test_fallback_when_unconfigured(self):
        assert generate_text(FakeLLM(configured=False), "q", fallback="fb") == "fb"

    def test_fallback_on_error(self):
        assert generate_text(FakeLLM(raises=True), "q", fallback="fb") == "fb"

    def test_fallback_on_empty_reply(self):
        assert generate_text(FakeLLM(reply="   "), "q", fallback="fb") == "fb"


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_array(self):
        assert extract_json("[1, 2]") == [1, 2]

    def test_fenced_block(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_without_lang(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_embedded_in_prose(self):
        assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_raises_when_no_json(self):
        with pytest.raises(ValueError):
            extract_json("no json here")


class TestGenerateModel:
    def test_valid_json_to_model(self):
        out = generate_model(
            FakeLLM(reply='{"name": "Ann", "age": 30}'), "q", Person,
            fallback=Person(name="x", age=0),
        )
        assert out == Person(name="Ann", age=30)

    def test_fenced_json(self):
        out = generate_model(
            FakeLLM(reply='```json\n{"name": "Bo", "age": 5}\n```'), "q", Person,
            fallback=Person(name="x", age=0),
        )
        assert out.name == "Bo"

    def test_fallback_when_unconfigured(self):
        fb = Person(name="fb", age=1)
        assert generate_model(FakeLLM(configured=False), "q", Person, fb) is fb

    def test_fallback_on_error(self):
        fb = Person(name="fb", age=1)
        assert generate_model(FakeLLM(raises=True), "q", Person, fb) is fb

    def test_fallback_on_invalid_json(self):
        fb = Person(name="fb", age=1)
        assert generate_model(FakeLLM(reply="not json"), "q", Person, fb) is fb

    def test_fallback_on_schema_mismatch(self):
        fb = Person(name="fb", age=1)
        # Missing required 'age' → validation error → fallback.
        out = generate_model(FakeLLM(reply='{"name": "Ann"}'), "q", Person, fb)
        assert out is fb

    def test_retries_once_then_succeeds(self):
        # First call raises, second returns valid JSON → retry recovers it.
        llm = FakeLLM(reply='{"name": "Ann", "age": 30}', fail_first=1)
        out = generate_model(llm, "q", Person, Person(name="fb", age=1))
        assert out == Person(name="Ann", age=30)
        assert llm.calls == 2

    def test_gives_up_after_retries(self):
        fb = Person(name="fb", age=1)
        llm = FakeLLM(reply='{"name": "Ann", "age": 30}', fail_first=5)
        out = generate_model(llm, "q", Person, fb)  # retries=1 → 2 attempts
        assert out is fb
        assert llm.calls == 2

    def test_retries_disabled(self):
        fb = Person(name="fb", age=1)
        llm = FakeLLM(reply='{"name": "Ann", "age": 30}', fail_first=1)
        out = generate_model(llm, "q", Person, fb, retries=0)
        assert out is fb
        assert llm.calls == 1

    def test_retry_is_corrective(self):
        # First reply is invalid (missing 'age'); the retry prompt must feed
        # back the bad output + error + schema so the model can fix it.
        llm = FakeLLM(
            replies=['{"name": "Ann"}', '{"name": "Ann", "age": 30}']
        )
        out = generate_model(llm, "extract the person", Person, Person(name="fb", age=1))
        assert out == Person(name="Ann", age=30)
        assert llm.calls == 2

        repair = llm.prompts[1]
        assert "extract the person" in repair          # original prompt retained
        assert '{"name": "Ann"}' in repair             # bad output fed back
        assert "Error:" in repair                       # the validation error
        assert "Required JSON schema:" in repair        # target schema

        # First attempt uses caller's system (None here); retry tightens it.
        assert llm.systems[0] is None
        assert "JSON" in llm.systems[1]
        assert "explanations" in llm.systems[1]

    def test_retry_system_keeps_caller_system(self):
        llm = FakeLLM(reply='{"name": "Ann", "age": 30}', fail_first=1)
        generate_model(
            llm, "extract", Person, Person(name="fb", age=1),
            system="You are an extractor.",
        )
        assert llm.systems[0] == "You are an extractor."
        # Retry system keeps the caller's instructions and adds JSON-only.
        assert "You are an extractor." in llm.systems[1]
        assert "JSON value only" in llm.systems[1]

    def test_repair_prompt_without_previous_on_call_error(self):
        # When the call itself raises, there's no previous reply to echo, but
        # the next attempt still proceeds (and here succeeds).
        llm = FakeLLM(reply='{"name": "Ann", "age": 30}', fail_first=1)
        out = generate_model(llm, "extract", Person, Person(name="fb", age=1))
        assert out == Person(name="Ann", age=30)
        repair = llm.prompts[1]
        assert "Previous response:" not in repair
