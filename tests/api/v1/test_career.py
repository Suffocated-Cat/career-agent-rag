import json


class _UnconfiguredLLM:
    def is_configured(self):
        return False

    def complete(self, *args, **kwargs):
        return ""


class _ScriptedLLM:
    """Configured LLM returning a fixed sequence of replies (last repeats)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def is_configured(self):
        return True

    def complete(self, *args, **kwargs):
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
