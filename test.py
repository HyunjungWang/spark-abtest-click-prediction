from fastapi.testclient import TestClient
from main import app 

client = TestClient(app)

def test_smart_ads_endpoint():
    payload = {
        "gender": "Female", "device_type": "Mobile",
        "browsing_history": "Shopping", "time_of_day": "Afternoon", "age": 25
    }
    response = client.post("/smart-ads", json=payload)
    assert response.status_code == 200
    assert "decision" in response.json()