"""Tests for the consistent error envelope and root endpoint."""
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.errors import register_exception_handlers


def _error_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    class Body(BaseModel):
        x: int

    @app.post("/validate")
    async def validate(body: Body):
        return {"ok": True}

    @app.get("/http")
    async def http_err():
        raise HTTPException(status_code=404, detail="not found")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    return app


class TestErrorEnvelope:
    def test_validation_error(self):
        client = TestClient(_error_app())
        resp = client.post("/validate", json={"x": "not-an-int"})
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"]["type"] == "validation_error"
        assert isinstance(body["error"]["detail"], list)

    def test_http_exception(self):
        client = TestClient(_error_app())
        resp = client.get("/http")
        assert resp.status_code == 404
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"]["type"] == "http_error"
        assert body["error"]["message"] == "not found"

    def test_unhandled_exception(self):
        # raise_server_exceptions=False so the 500 handler's response is returned.
        client = TestClient(_error_app(), raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"]["type"] == "internal_error"
        # Internals are not leaked.
        assert "kaboom" not in body["error"]["message"]


class TestRootAndHealth:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["app_name"] == "CareerAgent"
        assert data["docs"] == "/docs"
        assert data["ui"] == "/ui/"

    def test_ui_served(self, client):
        resp = client.get("/ui/")
        assert resp.status_code == 200
        assert "CareerAgent" in resp.text

    def test_real_app_validation_uses_envelope(self, client):
        # The real app's parse endpoint rejects empty text with the envelope.
        resp = client.post("/api/v1/jd/parse", json={"raw_text": ""})
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"]["type"] == "validation_error"
