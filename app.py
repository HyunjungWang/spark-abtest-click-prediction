import streamlit as st
import requests
import json
import os

# Page Configuration
st.set_page_config(page_title="Ad Click Predictor", page_icon="🎯")

st.title("🎯 Smart Ad Click Predictor")
st.write("Enter user data to predict the real-time probability of an ad click.")
API_URL = "https://dbc-89b96fbc-5a71.cloud.databricks.com/serving-endpoints/sk/invocations" 
try:
    dbx_token = st.secrets["DATABRICKS_TOKEN"].strip()
    dbx_url = st.secrets["DATABRICKS_SERVING_URL"].strip()
   
    
   # os.environ["DATABRICKS_TOKEN"] = dbx_token
   # os.environ["DATABRICKS_SERVING_URL"] = dbx_url
    
    st.success("✅ Credentials loaded successfully!")
except KeyError as e:
    st.error(f"❌ Secret missing: {e}. Check your secrets.toml file.")
    st.stop() 
    
 headers = {
    "Authorization": f"Bearer {dbx_token}",  # Bearer와 토큰 사이 공백 확인!
    "Content-Type": "application/json"
}
# 1. Input Form Setup
gender_map = {"Male": 0.0, "Female": 1.0}
device_map = {"Mobile": 0.0, "Desktop": 1.0, "Tablet": 2.0}
history_map = {"Shopping": 0.0, "News": 1.0, "Entertainment": 2.0, "Education": 3.0, "Social Media": 4.0, "Unknown": 5.0}
time_map = {"Morning": 0.0, "Afternoon": 1.0, "Evening": 2.0, "Night": 3.0}
# 2. Input Form (Sidebar)with st.sidebar:
st.header("👤 User Profile")
gender = st.selectbox("Gender", list(gender_map.keys()))
age = st.number_input("Age", min_value=10, max_value=90, value=25)
device = st.selectbox("Device Type", list(device_map.keys()))
    
st.header("🌐 Online Behavior")
browsing_history = st.selectbox("Browsing History", list(history_map.keys()))
    
st.header("⏰ Context")
time_of_day = st.selectbox("Time of Day", list(time_map.keys()))

# 2. Prediction Button
if st.button("Run Prediction"):
    # Convert input to numerical indices for the model
    payload = {
        "gender_idx": gender_map[gender],
        "device_type_idx": device_map[device],
        "browsing_history_idx": history_map[browsing_history],
        "time_of_day_idx": time_map[time_of_day],
        "age": age
    }

    try:
        with st.spinner("Analyzing optimal ad placement..."):
            # Send request to our FastAPI (main.py)
            response = requests.post(dbx_url,headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            decision = result.get('decision', 'UNKNOWN')
            position = result.get('assigned_position', 'None')
            prob = result.get('click_probability', 0.0)
            
            st.subheader("📊 Recommendation Results")
            
            if decision == "SHOW_AD":
                st.balloons()
                st.success(f"✅ **Decision: {decision}**")
                
                col1, col2 = st.columns(2)
                col1.metric("Best Position", position)
                col2.metric("Click Probability", f"{prob * 100:.2f}%")
                
                st.progress(prob)
                st.info(f"Target this user at the **{position}** of the page for maximum engagement.")
            else:
                st.warning(f"⚠️ **Decision: {decision}**")
                st.write(f"Reason: {result.get('reason', 'Low probability')}")

    except Exception as e:
        st.error(f"Failed to connect to the prediction server: {e}")

