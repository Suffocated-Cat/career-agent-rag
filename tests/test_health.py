def test_health_check_returns_200(client):
    """The /health endpoint should return 200 OK with status information."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "CareerAgent"
    assert "version" in data
