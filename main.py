from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import Row
from pyspark.sql import SparkSession
import mlflow
import os

spark = SparkSession.builder.getOrCreate()

mlflow.set_tracking_uri("databricks")
if "DATABRICKS_HOST" in os.environ:
    os.environ['DATABRICKS_HOST'] = os.getenv("DATABRICKS_HOST")
    os.environ['DATABRICKS_TOKEN'] = os.getenv("DATABRICKS_TOKEN")

    
app = FastAPI(title="Ad Click Prediction Service")

UC_VOLUME_PATH = "/Volumes/workspace/default/model_tmp"
model_uri = "models:/workspace.default.gbt@champion"
model = mlflow.spark.load_model(model_uri, dfs_tmpdir=UC_VOLUME_PATH)

class UserData(BaseModel):
    gender: str
    device_type: str
    browsing_history: str
    time_of_day: str
    age: int



@app.post("/smart-ads")
def get_ad_decision(data: UserData):
    positions = ["Top", "Side", "Bottom"]
    THRESHOLD = 0.4  
    rows = []
    for pos in positions:
        row_dict = data.dict()
        row_dict['ad_position'] = pos
        rows.append(Row(**row_dict))
    
    simulation_df = spark.createDataFrame(rows)
    
    predictions = model.transform(simulation_df)
    
    results = predictions.select("ad_position", "probability").collect()
    
    best_pos = None
    max_prob = 0
    
    for res in results:
        click_prob = float(res.probability[1]) 
        if click_prob > max_prob:
            max_prob = click_prob
            best_pos = res.ad_position
            
    if max_prob < THRESHOLD:
        return {
            "decision": "NO_AD",
            "reason": f"Maximum probability {max_prob:.4f} is below the required threshold of {THRESHOLD}",          
            "best_candidate": best_pos
        }
    
    return {
        "decision": "SHOW_AD",
        "assigned_position": best_pos,
        "click_probability": round(max_prob, 4)
    }
