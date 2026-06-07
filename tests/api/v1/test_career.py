class _UnconfiguredLLM:
    def is_configured(self):
        return False

    def complete(self, *args, **kwargs):
        return ""


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
