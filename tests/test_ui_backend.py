"""
tests/test_ui_backend.py
-------------------------
Unit tests verifying FastAPI backend application status and Benchmark All endpoint.
"""
from fastapi.testclient import TestClient
from ui.backend.app import app

client = TestClient(app)


def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "LexGLUE" in data["dataset"]


def test_api_benchmark_all():
    response = client.post("/api/benchmark-all")
    assert response.status_code == 200
    data = response.json()
    assert "completed successfully" in data["message"]
    assert len(data["summary_results"]) >= 3
    assert "report_artifacts" in data
