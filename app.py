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
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "dataframe_records": [
            {
                "gender": gender,
                "age": age,
                "device_type": device,
                "browsing_history": browsing_history, 
                "time_of_day": time_of_day          
            }
        ]
    }

    try:
        with st.spinner("Calculating probability..."):
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            decision = result.get('decision', 'UNKNOWN')
            position = result.get('assigned_position', 'None')
            yes_probability = result.get('click_probability', 0.0) 
            
            st.subheader("📊 Real-time Prediction Analysis")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(yes_probability)
            with col2:
                st.write(f"**{yes_probability * 100:.2f}%**")

            st.write(f"**Decision:** `{decision}` | **Position:** `{position}`")

            if yes_probability > 0.7:
                st.balloons()
                st.success(f"🔥 **High Potential!** The model predicts a {yes_probability * 100:.1f}% chance of a click.")
            elif yes_probability > 0.4:
                st.info(f"⚡ **Moderate Interest.** Probability: {yes_probability * 100:.1f}%")
            else:
                st.warning(f"❄️ **Low Engagement.** Probability: {yes_probability * 100:.1f}%")