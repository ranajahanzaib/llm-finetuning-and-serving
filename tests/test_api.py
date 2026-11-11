from fastapi.testclient import TestClient

from llm_finetuning_and_serving.api import app


def test_api_health():
    client = TestClient(app)
    assert client.get("/api/v1/health").status_code == 200


def test_api_limits_input():
    client = TestClient(app)
    assert client.post("/api/v1/predict", json={"text": ""}).status_code == 422
