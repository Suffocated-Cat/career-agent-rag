class _UnconfiguredLLM:
    def is_configured(self):
        return False

    def complete(self, *args, **kwargs):
        return ""


def _offline(monkeypatch):
    from app.services.knowledge import build_inmemory_kb_retriever

    monkeypatch.setattr("app.api.deps.get_llm", lambda: _UnconfiguredLLM())
    monkeypatch.setattr("app.api.deps.get_embedding_service", lambda: None)
    monkeypatch.setattr(
        "app.api.deps.get_kb_retriever", lambda: build_inmemory_kb_retriever()
    )


def test_interview_prep_returns_200(client, monkeypatch):
    """RAG interview-prep endpoint runs offline (BM25 KB + deterministic guide)."""
    _offline(monkeypatch)
    response = client.post(
        "/api/v1/interview-prep",
        json={
            "jd_text": "Role\n\nRequirements:\n- Python\n- Docker",
            "resume_text": "Skills: Python",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert "docker" in [g.lower() for g in data["gaps"]]
    assert isinstance(data["questions"], list)
    assert data["guide"]


def test_interview_prep_empty_rejected(client):
    response = client.post(
        "/api/v1/interview-prep", json={"jd_text": "", "resume_text": "x"}
    )
    assert response.status_code == 422
    assert response.json()["status"] == "error"
