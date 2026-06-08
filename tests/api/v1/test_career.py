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
