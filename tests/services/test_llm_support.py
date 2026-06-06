"""Tests for LLM augmentation helpers with deterministic fallback."""
import pytest
from pydantic import BaseModel

from app.services.llm_support import extract_json, generate_model, generate_text


class FakeLLM:
    def __init__(self, reply="", configured=True, raises=False):
        self.reply = reply
        self.configured = configured
        self.raises = raises

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        if self.raises:
            raise RuntimeError("api down")
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
