import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "BERT Sentiment Analysis" in response.text

def test_analyze_positive():
    response = client.post("/api/analyze", json={"text": "I love this product, it works amazingly well!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert "score" in data
    assert data["score"] > 0.5

def test_analyze_negative():
    response = client.post("/api/analyze", json={"text": "This service is terrible and disappointing."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert "score" in data
    assert data["score"] > 0.5

def test_analyze_empty():
    response = client.post("/api/analyze", json={"text": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Text cannot be empty."
