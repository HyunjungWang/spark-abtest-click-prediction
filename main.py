from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import Row
from pyspark.sql import SparkSession
import mlflow
import os
import requests
import streamlit as st
#DATABRICKS_SERVING_URL = os.getenv("DATABRICKS_SERVING_URL", "https://dbc-89b96fbc-5a71.cloud.databricks.com/serving-endpoints/sk/invocations")
#DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_TOKEN= st.secrets["DATABRICKS_TOKEN"].strip()
DATABRICKS_SERVING_URL = st.secrets["DATABRICKS_SERVING_URL"].strip()
app = FastAPI(title="Ad Click Prediction Service")
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
    age: float

@app.post("/smart-ads")
def get_ad_decision(data: UserData):
    """
    Endpoint that simulates 3 ad positions (Top, Side, Bottom)
    and returns the position with the highest predicted click probability.
    """
    if not DATABRICKS_TOKEN:
        return {"error": "DATABRICKS_TOKEN is not set in environment variables."}
    
    # 2. Simulate 3 potential ad positions (Top=0.0, Side=1.0, Bottom=2.0)
    positions = [0.0, 1.0, 2.0] 
    payload = {
        "dataframe_split": {
            "columns": [
                "gender_idx", 
                "device_type_idx", 
                "ad_position_idx", 
                "browsing_history_idx", 
                "time_of_day_idx", 
                "age"
            ],
            "data": [
                [float(data.gender_idx), float(data.device_type_idx), 0.0, float(data.browsing_history_idx), float(data.time_of_day_idx), float(data.age)],
                [float(data.gender_idx), float(data.device_type_idx), 1.0, float(data.browsing_history_idx), float(data.time_of_day_idx), float(data.age)],
                [float(data.gender_idx), float(data.device_type_idx), 2.0, float(data.browsing_history_idx), float(data.time_of_day_idx), float(data.age)]
            ]
        }
    }

    # 3. Requesting Databricks Model Serving
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(DATABRICKS_SERVING_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise exception for 4xx/5xx errors
        
        # 4. Parsing the probability results

        predictions= response.json().get("predictions", [])
        if len(predictions) > 0 and isinstance(predictions[0], list):
            probs = predictions[0] 
        else:
            probs = predictions    
        
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
