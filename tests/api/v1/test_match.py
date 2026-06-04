def test_match_returns_200(client):
    """POST /api/v1/match should return 200 with valid response schema."""
    response = client.post(
        "/api/v1/match",
        json={
            "jd": {"raw_text": "Looking for Python dev with Docker skills."},
            "resume": {"raw_text": "I know Python and Docker."},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    result = data["data"]
    assert isinstance(result["matched_skills"], list)
    assert isinstance(result["missing_skills"], list)
    assert 0.0 <= result["overall_score"] <= 1.0
    assert 0.0 <= result["skill_match_rate"] <= 1.0
    assert "semantic_similarity" in result
