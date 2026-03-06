import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(page_title="Ad Click Predictor", page_icon="🎯")

st.title("🎯 Smart Ad Click Predictor")
st.write("Enter user data to predict the real-time probability of an ad click.")
API_URL = "https://dbc-89b96fbc-5a71.cloud.databricks.com/serving-endpoints/gbt/invocations" 
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

# 1. Input Form Setup
# 2. Input Form (Sidebar)
with st.sidebar:
    st.header("👤 User Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=90, value=25)
    device = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
    
    st.header("🌐 Online Behavior")
    # Added your specific browsing history categories
    browsing_history = st.selectbox(
        "Browsing History", 
        ["Shopping", "News", "Entertainment", "Education", "Social Media", "Unknown"]
    )
    
    st.header("⏰ Context")
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    
# 2. Prediction Button
if st.button("Run Prediction"):
    # API Endpoint (Databricks Serving or Local FastAPI)
    # Update this URL with your actual deployment endpoint
    
    data = {
        "gender": gender,
        "age": age,
        "device_type": device,
        "browsing_history": "Shopping", # Sample Data
        "time_of_day": "Afternoon"
    }

    try:
        response = requests.post(API_URL, json=data)
        result = response.json()
        
        st.subheader("📊 Analysis Result")
        
        # Checking the model's decision
        if result.get("decision") == "YES":
            st.balloons()
            st.success(f"High conversion potential! This user is likely to click the ad. (Decision: {result['decision']})")
        else:
            st.warning("Low conversion potential. This user is unlikely to click the ad.")
            
    except Exception as e:
        st.error(f"Error: {e}\n(Please ensure the API server is running!)")