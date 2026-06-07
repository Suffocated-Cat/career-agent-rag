class _FakeLLM:
    def __init__(self, reply="", configured=True):
        self.reply = reply
        self.configured = configured

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        return self.reply


def test_parse_resume_returns_200(client, monkeypatch):
    """POST /api/v1/resume/parse should return 200 with valid response schema."""
    # Unconfigured LLM → deterministic rule-based parsing (offline).
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _FakeLLM(configured=False))
    response = client.post(
        "/api/v1/resume/parse",
        json={"raw_text": "Python developer with 5 years experience at Google."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert isinstance(data["data"]["skills"], list)
    assert isinstance(data["data"]["projects"], list)
    assert isinstance(data["data"]["education"], list)
    assert isinstance(data["data"]["experience"], list)


def test_parse_resume_uses_llm_when_configured(client, monkeypatch):
    """When the LLM is configured, the endpoint returns LLM-extracted fields."""
    reply = (
        '{"skills": ["Rust"], "experience": [], "education": [], '
        '"projects": [{"name": "CLI", "description": "a tool", "technologies": ["rust"]}]}'
    )
    monkeypatch.setattr("app.api.deps.get_llm", lambda: _FakeLLM(reply=reply))
    response = client.post("/api/v1/resume/parse", json={"raw_text": "some resume text"})

    assert response.status_code == 200
    resume = response.json()["data"]
    assert resume["skills"] == ["rust"]  # lowercased
    assert resume["projects"][0]["name"] == "CLI"
    assert resume["raw_text"] == "some resume text"


def test_parse_resume_empty_text_rejected(client):
    """POST /api/v1/resume/parse with empty raw_text should return 422."""
    response = client.post("/api/v1/resume/parse", json={"raw_text": ""})

    assert response.status_code == 422
