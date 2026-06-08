import json


class _UnconfiguredLLM:
    def is_configured(self):
        return False

    def complete(self, *args, **kwargs):
        return ""


class _ScriptedLLM:
    """Configured LLM that scripts the *agent's* decisions only.

    Decision prompts (identified by the ReAct system prompt) are served from the
    scripted sequence; every other call — tool-internal extraction and the
    final-answer composition — returns ``answer``. Tool extraction treats that
    as non-JSON and falls back deterministically, while the composition step
    yields it as the final reply (also streamed by ``stream``).
    """

    def __init__(self, replies, answer="Final answer."):
        self.replies = list(replies)
        self.answer = answer
        self.calls = 0

    def is_configured(self):
        return True

    def complete(self, prompt, system=None, **kwargs):
        if not system or "ReAct agent" not in system:
            return self.answer  # compose / tool-internal
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply

    def stream(self, prompt, system=None, **kwargs):
        yield self.answer


def test_career_match_returns_200(client, monkeypatch):
    """POST /api/v1/career-match runs the full pipeline (offline path)."""
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)

    response = client.post(
        "/api/v1/career-match",
        json={
            "jd_text": "ML Engineer at Acme\n\nRequirements:\n- Python\n- Docker",
            "resume_text": "Skills: Python\n\nExperience:\nML Engineer at Beta\n- Built models in Python",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["report"]["full_report"]
    assert 0.0 <= data["match"]["overall_score"] <= 1.0
    assert "project_audit" in data["match"]
    assert "project_relevance" in data["match"]


def test_career_match_empty_rejected(client):
    """Empty inputs return the consistent 422 error envelope."""
    response = client.post(
        "/api/v1/career-match", json={"jd_text": "", "resume_text": "x"}
    )
    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_career_ask_drives_agent(client, monkeypatch):
    """POST /api/v1/career/ask runs the ReAct agent and returns answer + trace."""
    scripted = _ScriptedLLM([
        json.dumps({"thought": "parse first", "action": "parse_jd", "action_input": {}}),
        json.dumps({"thought": "done", "action": "finish"}),
    ], answer="You match well on Python.")
    monkeypatch.setattr("app.api.deps.get_llm", lambda: scripted)
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr("app.api.deps.get_kb_retriever", lambda: None)

    response = client.post(
        "/api/v1/career/ask",
        json={
            "question": "Why isn't my match higher?",
            "jd_text": "ML Engineer at Acme\n\nRequirements:\n- Python",
            "resume_text": "Skills: Python\n\nExperience:\nML Engineer at Beta",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["completed"] is True
    assert body["answer"] == "You match well on Python."  # composed after finish
    assert body["steps"][0]["action"] == "parse_jd"


def test_career_ask_empty_rejected(client):
    """Empty question returns the consistent 422 error envelope."""
    response = client.post(
        "/api/v1/career/ask",
        json={"question": "", "jd_text": "x", "resume_text": "y"},
    )
    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_career_ask_requires_a_jd(client):
    """Neither jd_text nor jds → 422 from the model validator."""
    response = client.post(
        "/api/v1/career/ask",
        json={"question": "which?", "resume_text": "y"},
    )
    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_career_ask_requires_configured_llm(client, monkeypatch):
    """An unconfigured LLM yields a clean 503, not a raw 500."""
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    response = client.post(
        "/api/v1/career/ask",
        json={"question": "why?", "jd_text": "JD", "resume_text": "resume"},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert "configured LLM" in body["error"]["message"]


def test_career_ask_compares_multiple_jds(client, monkeypatch):
    """Multiple JDs drive the compare_jds tool and return a ranked answer."""
    scripted = _ScriptedLLM([
        json.dumps({"thought": "need resume", "action": "parse_resume", "action_input": {}}),
        json.dumps({"thought": "rank them", "action": "compare_jds", "action_input": {}}),
        json.dumps({"thought": "done", "action": "finish"}),
    ], answer="Apply to Job A first.")
    monkeypatch.setattr("app.api.deps.get_llm", lambda: scripted)
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr("app.api.deps.get_kb_retriever", lambda: None)

    response = client.post(
        "/api/v1/career/ask",
        json={
            "question": "Which of these should I apply to?",
            "resume_text": "Skills: Python\n\nExperience:\nML Engineer at Beta",
            "jds": [
                {"text": "ML Engineer\n\nRequirements:\n- Python", "label": "Job A"},
                {"text": "Frontend\n\nRequirements:\n- React"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["completed"] is True
    actions = [s["action"] for s in body["steps"]]
    assert "compare_jds" in actions
    assert body["answer"] == "Apply to Job A first."


# ── /career/chat/stream (multi-turn chat over SSE) ─────────────────────────

JD = "ML Engineer at Acme\n\nRequirements:\n- Python\n- Docker"
RESUME = "Skills: Python\n\nExperience:\nML Engineer at Beta\n- Built models in Python"


def _done_event(text):
    """Parse the JSON payload of the final `event: done` SSE message."""
    for block in text.split("\n\n"):
        if "event: done" in block:
            for line in block.split("\n"):
                if line.startswith("data:"):
                    return json.loads(line[len("data:"):].strip())
    return None


def test_chat_stream_slash_emits_only_done(client, monkeypatch):
    """Slash commands run the deterministic pipeline — no LLM, no steps/tokens."""
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)

    res = client.post("/api/v1/career/chat/stream", json={
        "message": "/match", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 200
    assert "event: step" not in res.text and "event: token" not in res.text
    done = _done_event(res.text)
    assert done["state"] == "answered"
    assert done["session_id"]
    assert "Match score" in done["reply"]


def test_chat_stream_help(client, monkeypatch):
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    res = client.post("/api/v1/career/chat/stream", json={
        "message": "/help", "resume_text": "x", "jd_text": "y",
    })
    assert res.status_code == 200
    assert "/compare" in _done_event(res.text)["reply"]


def test_chat_stream_free_text_streams_steps_then_answer(client, monkeypatch):
    scripted = _ScriptedLLM([
        json.dumps({"thought": "parse", "action": "parse_jd", "action_input": {}}),
        json.dumps({"thought": "done", "action": "finish"}),
    ], answer="You're a solid match.")
    monkeypatch.setattr("app.api.deps.get_llm", lambda: scripted)
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr("app.api.deps.get_kb_retriever", lambda: None)

    res = client.post("/api/v1/career/chat/stream", json={
        "message": "how do I look?", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 200
    text = res.text
    assert "event: step" in text and "parse_jd" in text  # reasoning streamed
    assert "event: token" in text                        # answer streamed
    done = _done_event(text)
    assert done["state"] == "answered"
    assert done["reply"] == "You're a solid match."


def test_chat_stream_ask_user_pause_and_resume(client, monkeypatch):
    scripted = _ScriptedLLM([
        json.dumps({"thought": "need info", "action": "ask_user",
                    "action_input": {"question": "What was the measurable impact?"}}),
        json.dumps({"thought": "done", "action": "finish"}),
    ], answer="Rewritten: cut latency 40%.")
    monkeypatch.setattr("app.api.deps.get_llm", lambda: scripted)
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr("app.api.deps.get_kb_retriever", lambda: None)

    # Turn 1: the agent pauses to ask the user (no answer tokens yet).
    res1 = client.post("/api/v1/career/chat/stream", json={
        "message": "Improve my bullet 'built models'", "jd_text": JD, "resume_text": RESUME,
    })
    done1 = _done_event(res1.text)
    sid = done1["session_id"]
    assert done1["state"] == "awaiting_user"
    assert done1["reply"] == "What was the measurable impact?"
    assert "event: token" not in res1.text

    # Turn 2: the reply resumes the same run; the answer is then streamed.
    res2 = client.post("/api/v1/career/chat/stream", json={
        "session_id": sid, "message": "It cut latency 40%",
    })
    done2 = _done_event(res2.text)
    assert done2["session_id"] == sid
    assert done2["state"] == "answered"
    assert done2["reply"] == "Rewritten: cut latency 40%."
    assert "event: token" in res2.text
    assert len(done2["history"]) == 4  # 2 user + 2 assistant


def test_chat_stream_requires_llm_for_agent(client, monkeypatch):
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    res = client.post("/api/v1/career/chat/stream", json={
        "message": "free text question", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 503
