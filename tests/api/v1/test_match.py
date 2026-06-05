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


def test_match_response_includes_semantic_fields(client):
    """POST /api/v1/match response should include vector match fields."""
    response = client.post(
        "/api/v1/match",
        json={
            "jd": {
                "raw_text": "Looking for ML engineer with PyTorch experience.",
                "skills": ["pytorch", "docker"],
                "responsibilities": ["Build ML models"],
            },
            "resume": {
                "raw_text": "Deep learning engineer with TensorFlow and Kubernetes.",
                "skills": ["tensorflow", "kubernetes"],
                "experience": [
                    {
                        "title": "ML Engineer",
                        "company": "AI Corp",
                        "duration": "2022–Present",
                        "highlights": ["Built recommendation system"],
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    result = data["data"]

    # New vector match fields should be present in response
    assert "semantic_skill_matches" in result
    assert isinstance(result["semantic_skill_matches"], list)
    assert "semantic_skill_match_rate" in result
    assert "experience_matches" in result
    assert isinstance(result["experience_matches"], list)
    assert "experience_match_rate" in result
