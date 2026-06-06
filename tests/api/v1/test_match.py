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


def test_match_response_includes_project_relevance(client):
    """POST /api/v1/match should rank resume items by relevance to the JD."""
    response = client.post(
        "/api/v1/match",
        json={
            "jd": {
                "raw_text": "Looking for ML engineer with PyTorch.",
                "skills": ["pytorch", "recommendation", "ranking"],
                "responsibilities": ["Build recommendation models"],
            },
            "resume": {
                "raw_text": "ML and frontend experience.",
                "skills": ["pytorch", "react"],
                "experience": [
                    {
                        "title": "ML Engineer",
                        "company": "Acme",
                        "highlights": ["Built recommendation ranking models in pytorch"],
                    },
                    {
                        "title": "Frontend Dev",
                        "company": "WebCo",
                        "highlights": ["Built a react dashboard"],
                    },
                ],
            },
        },
    )

    assert response.status_code == 200
    result = response.json()["data"]

    assert "project_relevance" in result
    relevance = result["project_relevance"]
    assert isinstance(relevance, list)
    assert len(relevance) >= 1

    top = relevance[0]
    assert {"doc_id", "source_type", "label", "score", "normalized_score"} <= top.keys()
    assert top["label"] == "ML Engineer at Acme"  # most relevant to the JD
    assert top["normalized_score"] == 1.0


def test_generate_report_returns_200(client):
    """POST /api/v1/match/report should return 200 with valid report schema."""
    response = client.post(
        "/api/v1/match/report",
        json={
            "jd": {
                "raw_text": "Looking for ML Engineer with PyTorch.",
                "title": "ML Engineer",
                "skills": ["pytorch", "docker"],
                "responsibilities": ["Build ML models"],
            },
            "resume": {
                "raw_text": "Deep learning engineer.",
                "skills": ["tensorflow", "kubernetes"],
                "experience": [
                    {
                        "title": "ML Engineer",
                        "company": "AI Corp",
                        "highlights": ["Built recommendation system"],
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    report = data["data"]

    # Check report structure
    assert "job_title" in report
    assert report["job_title"] == "ML Engineer"
    assert "overall_score" in report
    assert "overall_rating" in report
    assert report["overall_rating"] in ["Excellent", "Good", "Fair", "Low"]
    assert "skill_summary" in report
    assert "matched_skills" in report
    assert "missing_skills" in report
    assert "skill_gap_analysis" in report
    assert "recommendations" in report
    assert "full_report" in report
    assert len(report["full_report"]) > 0
    assert 0.0 <= report["overall_score"] <= 1.0
