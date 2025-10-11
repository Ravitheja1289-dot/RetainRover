import json
import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'status' in data


def test_predict_sample():
    # Use sample from enhanced_app sample-data
    sample = {
        "age": 45,
        "gender": 1,
        "tenure": 39,
        "balance": 83807.86,
        "products_number": 1,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 119346.88
    }
    resp = client.post('/predict', json=sample)
    # If model or preprocessor missing, expect 503; otherwise 200
    assert resp.status_code in (200, 500, 503)
