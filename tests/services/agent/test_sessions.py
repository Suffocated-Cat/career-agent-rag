"""Tests for chat sessions and rolling conversation memory."""
from app.services.agent.schemas import ReactState
from app.services.agent.sessions import (
    ChatMessage,
    ChatSession,
    SessionStore,
    conversation_context,
    fold_old_turns,
)


class FakeLLM:
    def __init__(self, reply="ROLLED"):
        self.reply = reply
        self.calls = 0

    def is_configured(self):
        return True

    def complete(self, prompt, system=None, **kwargs):
        self.calls += 1
        return self.reply


def _session():
    return ChatSession(session_id="s", state=ReactState())


def _round(s, u, a):
    s.history.append(ChatMessage("user", u))
    s.history.append(ChatMessage("assistant", a))


class TestSessionStore:
    def test_create_and_get(self):
        store = SessionStore()
        s = store.create(ReactState())
        assert store.get(s.session_id) is s

    def test_get_missing(self):
        store = SessionStore()
        assert store.get(None) is None
        assert store.get("nope") is None


class TestConversationContext:
    def test_empty_on_first_message(self):
        s = _session()
        s.history.append(ChatMessage("user", "u1"))  # only the current message
        assert conversation_context(s) == ""

    def test_recent_verbatim_excludes_current(self):
        s = _session()
        _round(s, "u1", "a1")
        s.history.append(ChatMessage("user", "u2"))  # current message
        ctx = conversation_context(s)
        assert "User: u1" in ctx and "Assistant: a1" in ctx
        assert "u2" not in ctx

    def test_includes_summary(self):
        s = _session()
        s.summary = "earlier stuff"
        s.summarized = 0
        s.history.append(ChatMessage("user", "now"))
        assert "Summary of earlier conversation: earlier stuff" in conversation_context(s)


class TestFoldOldTurns:
    def test_no_fold_within_window(self):
        s = _session()
        _round(s, "u0", "a0")
        _round(s, "u1", "a1")  # 4 messages ≤ window
        fold_old_turns(s, FakeLLM())
        assert s.summary == "" and s.summarized == 0

    def test_folds_beyond_window_via_llm(self):
        s = _session()
        for i in range(5):
            _round(s, f"u{i}", f"a{i}")  # 10 messages
        llm = FakeLLM("ROLLED")
        fold_old_turns(s, llm)
        assert s.summarized == 4  # 10 - window(6)
        assert s.summary == "ROLLED"
        assert llm.calls == 1
        # The recent window stays available verbatim.
        assert len(s.history) - s.summarized == 6

    def test_incremental_fold(self):
        s = _session()
        for i in range(5):
            _round(s, f"u{i}", f"a{i}")
        llm = FakeLLM()
        fold_old_turns(s, llm)
        _round(s, "u5", "a5")  # 12 messages
        fold_old_turns(s, llm)
        assert s.summarized == 6
        assert llm.calls == 2  # folded twice, incrementally

    def test_fallback_without_llm(self):
        s = _session()
        for i in range(5):
            _round(s, f"u{i}", f"a{i}")
        fold_old_turns(s, None)
        assert s.summarized == 4
        assert "u0" in s.summary  # bounded plain-text trail
