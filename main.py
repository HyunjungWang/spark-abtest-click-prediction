from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import Row
from pyspark.sql import SparkSession
import mlflow
import os

DATABRICKS_SERVING_URL="https://dbc-89b96fbc-5a71.cloud.databricks.com/serving-endpoints/sk/invocations"
if "DATABRICKS_HOST" in os.environ:
    os.environ['DATABRICKS_HOST'] = os.getenv("DATABRICKS_HOST")
    os.environ['DATABRICKS_TOKEN'] = os.getenv("DATABRICKS_TOKEN")

    
app = FastAPI(title="Ad Click Prediction Service")
# Business Logic: Minimum probability to justify showing an ad
THRESHOLD = 0.4 

class UserData(BaseModel):
    """
    Schema for incoming user data.
    Note: Categorical features must be pre-encoded as indices (float)
    as the Scikit-Learn model expects numerical input.
    """
    gender_idx: float
    device_type_idx: float
    browsing_history_idx: float
    time_of_day_idx: float
    age: int

@app.post("/smart-ads")
def get_ad_decision(data: UserData):
    """
    Endpoint that simulates 3 ad positions (Top, Side, Bottom)
    and returns the position with the highest predicted click probability.
    """
    
    # 2. Simulate 3 potential ad positions (Top=0.0, Side=1.0, Bottom=2.0)
    positions = [0.0, 1.0, 2.0] 
    candidates = []
    
    for pos in positions:
        # Construct feature row in the EXACT order used during model training
        row = [
            data.gender_idx,
            data.device_type_idx,
            pos, # ad_position_idx
            data.browsing_history_idx,
            data.time_of_day_idx,
            data.age
        ]
        candidates.append(row)

    # 3. Requesting Databricks Model Serving
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # "dataframe_records" is the standard format for Scikit-Learn models in MLflow serving
    payload = {"dataframe_records": candidates}

    try:
        response = requests.post(DATABRICKS_SERVING_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise exception for 4xx/5xx errors
        
        # 4. Parsing the probability results
        # Our custom Wrapper returns a list of probabilities (float)
        probs = response.json().get("predictions", [])
        
        if not probs:
            return {"error": "No predictions returned from the model."}

        # Find the highest probability and its corresponding position
        max_prob = max(probs)
        best_idx = probs.index(max_prob)
        
        pos_map = {0: "Top", 1: "Side", 2: "Bottom"}
        best_pos_name = pos_map.get(best_idx)

        # 5. Business Rule Check
        if max_prob < THRESHOLD:
            return {
                "decision": "NO_AD",
                "reason": f"Maximum probability {max_prob:.4f} is below the threshold of {THRESHOLD}",
                "best_candidate_position": best_pos_name
            }

        return {
            "decision": "SHOW_AD",
            "assigned_position": best_pos_name,
            "click_probability": round(max_prob, 4)
        }

    except requests.exceptions.RequestException as e:
        return {"error": "Databricks Serving Request Failed", "details": str(e)}
    except Exception as e:
        return {"error": "An unexpected error occurred", "details": str(e)}
