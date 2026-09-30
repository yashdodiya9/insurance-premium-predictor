import os
import requests
import streamlit as st

def get_api_base_url() -> str:
    """API address: Streamlit secret `API_URL`, else env var `API_URL`, else local default."""
    try:
        return st.secrets["API_URL"].rstrip("/")
    except Exception:  # no secrets configured (e.g. running locally)
        return os.getenv("API_URL", "http://13.60.16.138:8000").rstrip("/")


API_URL = get_api_base_url() + "/predict"

st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job']
)

if st.button("Predict Premium Category"):
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:
        # Generous timeout: a free-tier API may need a while to wake up after being idle.
        with st.spinner("Contacting the API (it may take up to a minute if it was asleep)..."):
            response = requests.post(API_URL, json=input_data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Insurance Premium Category: **{result['predicted_category']}**")
            st.metric("Confidence", f"{result['confidence']:.0%}")
            st.bar_chart(result["class_probabilities"])
        elif response.status_code == 422:
            st.error("Some inputs were invalid. Please check your values.")
            st.json(response.json())
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.json())

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
    except requests.exceptions.Timeout:
        st.error("⏱️ The API took too long to respond. Please try again.")
