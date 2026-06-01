def test_parse_resume_returns_200(client):
    """POST /api/v1/resume/parse should return 200 with valid response schema."""
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


def test_parse_resume_empty_text_rejected(client):
    """POST /api/v1/resume/parse with empty raw_text should return 422."""
    response = client.post("/api/v1/resume/parse", json={"raw_text": ""})

    assert response.status_code == 422
