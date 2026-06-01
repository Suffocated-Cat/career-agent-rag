def test_parse_jd_returns_200(client):
    """POST /api/v1/jd/parse should return 200 with valid response schema."""
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


def test_parse_jd_empty_text_rejected(client):
    """POST /api/v1/jd/parse with empty raw_text should return 422."""
    response = client.post("/api/v1/jd/parse", json={"raw_text": ""})

    assert response.status_code == 422
