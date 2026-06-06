class _FakeLLM:
    def __init__(self, reply="", configured=True):
        self.reply = reply
        self.configured = configured

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        return self.reply


def test_parse_jd_returns_200(client, monkeypatch):
    """POST /api/v1/jd/parse should return 200 with valid response schema."""
    # Unconfigured LLM → deterministic rule-based parsing (offline).
    monkeypatch.setattr("app.api.v1.jd._get_llm", lambda: _FakeLLM(configured=False))
    response = client.post(
        "/api/v1/jd/parse",
        json={"raw_text": "Looking for a Python developer with 3+ years experience."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["raw_text"] == "Looking for a Python developer with 3+ years experience."
    assert isinstance(data["data"]["skills"], list)


def test_parse_jd_uses_llm_when_configured(client, monkeypatch):
    """When the LLM is configured, the endpoint returns LLM-extracted fields."""
    reply = '{"title": "Staff Engineer", "company": "Acme", "skills": ["Go"], "responsibilities": [], "nice_to_haves": []}'
    monkeypatch.setattr("app.api.v1.jd._get_llm", lambda: _FakeLLM(reply=reply))
    response = client.post("/api/v1/jd/parse", json={"raw_text": "some jd text"})

    assert response.status_code == 200
    jd = response.json()["data"]
    assert jd["title"] == "Staff Engineer"
    assert jd["skills"] == ["go"]  # lowercased
    assert jd["raw_text"] == "some jd text"


def test_parse_jd_empty_text_rejected(client):
    """POST /api/v1/jd/parse with empty raw_text should return 422."""
    response = client.post("/api/v1/jd/parse", json={"raw_text": ""})

    assert response.status_code == 422


def test_get_llm_builds_client():
    """_get_llm constructs a real LLMClient (offline; no network on build)."""
    from app.api.v1.jd import _get_llm
    from app.services.llm_client import LLMClient

    _get_llm.cache_clear()
    assert isinstance(_get_llm(), LLMClient)
    _get_llm.cache_clear()
