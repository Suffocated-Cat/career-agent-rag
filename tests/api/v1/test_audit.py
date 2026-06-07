class _FakeAuditLLM:
    def __init__(self, reply="", configured=True):
        self.reply = reply
        self.configured = configured

    def is_configured(self):
        return self.configured

    def complete(self, prompt, system=None, **kwargs):
        return self.reply


def test_audit_returns_200(client, monkeypatch):
    """POST /api/v1/audit should return a risk report."""
    # Unconfigured LLM → no advice, offline.
    monkeypatch.setattr(
        "app.api.deps.get_llm", lambda: _FakeAuditLLM(configured=False)
    )
    response = client.post(
        "/api/v1/audit",
        json={
            "resume": {
                "raw_text": "...",
                "skills": ["python", "rag", "agent", "mcp"],
                "experience": [
                    {
                        "title": "AI Intern",
                        "company": "BrightApps",
                        "highlights": ["Worked on several AI features and helped improve UX"],
                    }
                ],
                "projects": [
                    {
                        "name": "LLM Assistant",
                        "description": "A chatbot.",
                        "technologies": ["llm", "rag", "agent", "mcp"],
                    }
                ],
            }
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    report = data["data"]

    assert "findings" in report
    assert isinstance(report["findings"], list)
    assert len(report["findings"]) >= 1
    assert 0.0 <= report["risk_score"] <= 1.0
    categories = {f["category"] for f in report["findings"]}
    assert "unsupported_skill" in categories

    finding = report["findings"][0]
    assert {"category", "severity", "subject", "detail"} <= finding.keys()


def test_audit_clean_resume_low_risk(client):
    """A substantiated resume should yield no findings and zero risk."""
    response = client.post(
        "/api/v1/audit",
        json={"resume": {"raw_text": "nothing structured"}},
    )

    assert response.status_code == 200
    report = response.json()["data"]
    assert report["findings"] == []
    assert report["risk_score"] == 0.0


def test_audit_includes_advice_when_llm_configured(client, monkeypatch):
    """With an LLM configured, the audit returns how-to-fix advice."""
    monkeypatch.setattr(
        "app.api.deps.get_llm",
        lambda: _FakeAuditLLM(reply="Add metrics and link a repo."),
    )
    response = client.post(
        "/api/v1/audit",
        json={
            "resume": {
                "raw_text": "...",
                "skills": ["python", "rag", "agent", "mcp"],
                "projects": [
                    {"name": "LLM Assistant", "description": "A chatbot.",
                     "technologies": ["rag", "agent"]}
                ],
            }
        },
    )

    assert response.status_code == 200
    report = response.json()["data"]
    assert report["findings"]  # deterministic findings present
    assert report["advice"] == "Add metrics and link a repo."
