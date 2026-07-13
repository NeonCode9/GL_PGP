import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Visit with Us: Premium Package AI Analytics", layout="wide", page_icon="✈️")
st.title("✈️ Wellness Tourism Package Purchase Predictor")

@st.cache_resource
def load_production_model():
    # Target model artifact stored locally within the root container context
    local_path = "best_model.pkl"
    if os.path.exists(local_path):
        return joblib.load(local_path)
    return None

model = load_production_model()

if model is not None:
    st.success("⚡ Production machine learning pipeline active within container space.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Profile Demographics**")
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Gross Monthly Income (INR)", min_value=0, value=25000)
    with col2:
        st.info("**Logistics & Companions**")
        city_tier = st.selectbox("City Tier Classification", [1, 2, 3])
        num_person = st.number_input("Total Companions Traveling", min_value=1, max_value=15, value=2)
        children = st.number_input("Accompanying Children (< Age 5)", min_value=0, max_value=10, value=0)
        num_trips = st.number_input("Average Annual Trips Target", min_value=0, max_value=25, value=2)
        passport = st.selectbox("Valid Passport Holder?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        own_car = st.selectbox("Owns Personal Vehicle?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    with col3:
        st.info("**Sales Outreach Telemetry**")
        type_of_contact = st.selectbox("Lead Sourcing Stream", ["Company Invited", "Self Inquiry", "Unknown"])
        product_pitched = st.selectbox("Product Tier Originally Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        duration_pitch = st.number_input("Sales Pitch Duration (Minutes)", min_value=0, max_value=180, value=15)
        followups = st.number_input("Total Campaign Follow-ups", min_value=0, max_value=20, value=3)
        property_star = st.slider("Preferred Hotel Rating Selection", 3, 5, 3)
        pitch_satisfaction = st.slider("Outreach Pitch Satisfaction Score", 1, 5, 3)

    if st.button("🔮 Score Target Purchase Probability"):
        payload = pd.DataFrame([{
            'Age': float(age), 'TypeofContact': type_of_contact, 'CityTier': int(city_tier),
            'Occupation': occupation, 'Gender': gender, 'NumberOfPersonVisiting': float(num_person),
            'PreferredPropertyStar': float(property_star), 'MaritalStatus': marital_status,
            'NumberOfTrips': float(num_trips), 'Passport': int(passport), 'OwnCar': int(own_car),
            'NumberOfChildrenVisiting': float(children), 'Designation': designation,
            'MonthlyIncome': float(monthly_income), 'PitchSatisfactionScore': int(pitch_satisfaction),
            'ProductPitched': product_pitched, 'NumberOfFollowups': float(followups),
            'DurationOfPitch': float(duration_pitch)
        }])
        
        prediction = model.predict(payload)[0]
        probability = model.predict_proba(payload)[0][1]
        
        st.markdown("---")
        if prediction == 1:
            st.success(f"🎯 **High Conversion Likelihood! Target Probability: {probability*100:.2f}%**")
        else:
            st.warning(f"💤 **Low Conversion Candidate. Target Probability: {probability*100:.2f}%**")
else:
    st.error("❌ Application Error: System could not locate local container model weights files.")
