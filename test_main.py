import httpx
import pytest

def test_real_ad_prediction():
    # Local FastAPI endpoint
    url = "http://localhost:8000/smart-ads"
    
    # Real-world test data (Numerical indices)
    payload = {
        "gender_idx": 0.0,
        "device_type_idx": 1.0,
        "browsing_history_idx": 2.0,
        "time_of_day_idx": 1.0,
        "age": 28
    }
    
    with httpx.Client() as client:
        response = client.post(url, json=payload, timeout=60.0)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "click_probability" in data or "reason" in data
    
    if "click_probability" in data:
        assert 0 <= data["click_probability"] <= 1.0
