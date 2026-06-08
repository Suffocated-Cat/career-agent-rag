import json


class _UnconfiguredLLM:
    def is_configured(self):
        return False

    def complete(self, *args, **kwargs):
        return ""


class _ScriptedLLM:
    """Configured LLM that scripts the *agent's* decisions only.

    The same client is shared by the agent loop and the tools' internal LLM
    augmentation (e.g. ``parse_jd`` extraction). To keep the scripted decision
    sequence from being consumed by those internal calls, replies are served
    only for ReAct decision prompts (identified by the agent system prompt);
    every other call returns "" so the deterministic fallback path is used.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def is_configured(self):
        return True

    def complete(self, prompt, system=None, **kwargs):
        if not system or "ReAct agent" not in system:
            return ""  # tool-internal call → deterministic fallback
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


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
        json.dumps({"thought": "done", "final_answer": "You match well on Python."}),
    ])
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
    assert body["answer"] == "You match well on Python."
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
        json.dumps({"thought": "done", "final_answer": "Apply to Job A first."}),
    ])
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


# ── /career/chat (multi-turn, slash commands, ask_user) ────────────────────

JD = "ML Engineer at Acme\n\nRequirements:\n- Python\n- Docker"
RESUME = "Skills: Python\n\nExperience:\nML Engineer at Beta\n- Built models in Python"


def test_chat_slash_works_without_llm(client, monkeypatch):
    """Slash commands run the deterministic pipeline — no LLM needed."""
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)

    res = client.post("/api/v1/career/chat", json={
        "message": "/match", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"]
    assert body["state"] == "answered"
    assert "Match score" in body["reply"]


def test_chat_help_is_first_turn(client, monkeypatch):
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    res = client.post("/api/v1/career/chat", json={"message": "/help", "resume_text": "x", "jd_text": "y"})
    assert res.status_code == 200
    assert "/compare" in res.json()["reply"]


def test_chat_free_text_runs_agent(client, monkeypatch):
    scripted = _ScriptedLLM([
        json.dumps({"thought": "parse", "action": "parse_jd", "action_input": {}}),
        json.dumps({"thought": "done", "final_answer": "You're a solid match."}),
    ])
    monkeypatch.setattr("app.api.deps.get_llm", lambda: scripted)
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr("app.api.deps.get_kb_retriever", lambda: None)

    res = client.post("/api/v1/career/chat", json={
        "message": "How do I look for this role?", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["state"] == "answered"
    assert body["reply"] == "You're a solid match."
    assert [s["action"] for s in body["steps"]] == ["parse_jd"]


def test_chat_ask_user_pause_and_resume(client, monkeypatch):
    scripted = _ScriptedLLM([
        json.dumps({"thought": "need info", "action": "ask_user",
                    "action_input": {"question": "What was the measurable impact?"}}),
        json.dumps({"thought": "done", "final_answer": "Rewritten: cut latency 40%."}),
    ])
    monkeypatch.setattr("app.api.deps.get_llm", lambda: scripted)
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr("app.api.deps.get_kb_retriever", lambda: None)

    # Turn 1: the agent pauses to ask the user.
    res1 = client.post("/api/v1/career/chat", json={
        "message": "Improve my bullet 'built models'", "jd_text": JD, "resume_text": RESUME,
    })
    body1 = res1.json()
    sid = body1["session_id"]
    assert body1["state"] == "awaiting_user"
    assert body1["reply"] == "What was the measurable impact?"

    # Turn 2: the user's reply resumes the same run to completion.
    res2 = client.post("/api/v1/career/chat", json={
        "session_id": sid, "message": "It cut latency 40%",
    })
    body2 = res2.json()
    assert body2["session_id"] == sid
    assert body2["state"] == "answered"
    assert body2["reply"] == "Rewritten: cut latency 40%."
    # History spans both turns (2 user + 2 assistant).
    assert len(body2["history"]) == 4


def test_chat_agent_requires_llm(client, monkeypatch):
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    res = client.post("/api/v1/career/chat", json={
        "message": "free text question", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 503
    assert res.json()["status"] == "error"


# ── /career/chat/stream (SSE) ──────────────────────────────────────────────

def test_chat_stream_emits_steps_and_done(client, monkeypatch):
    scripted = _ScriptedLLM([
        json.dumps({"thought": "parse", "action": "parse_jd", "action_input": {}}),
        json.dumps({"thought": "done", "final_answer": "Looks good."}),
    ])
    monkeypatch.setattr("app.api.deps.get_llm", lambda: scripted)
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr("app.api.deps.get_kb_retriever", lambda: None)

    res = client.post("/api/v1/career/chat/stream", json={
        "message": "how do I look?", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 200
    text = res.text
    assert "event: step" in text   # at least one streamed step (parse_jd)
    assert "event: done" in text
    assert "Looks good." in text


def test_chat_stream_slash_emits_only_done(client, monkeypatch):
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)

    res = client.post("/api/v1/career/chat/stream", json={
        "message": "/match", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 200
    assert "event: step" not in res.text
    assert "event: done" in res.text
    assert "Match score" in res.text


def test_chat_stream_requires_llm_for_agent(client, monkeypatch):
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    res = client.post("/api/v1/career/chat/stream", json={
        "message": "free text question", "jd_text": JD, "resume_text": RESUME,
    })
    assert res.status_code == 503
