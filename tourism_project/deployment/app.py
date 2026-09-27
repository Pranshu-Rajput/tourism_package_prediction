import os
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------- Load model
# The model is committed by the CI/CD pipeline and sits next to this file.
# abspath(__file__) makes the path work regardless of where Streamlit is launched from.
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# ---------------------------------------------------------------- Page config
st.set_page_config(page_title="Wellness Tourism Package Prediction", page_icon="🧳")

st.title("Wellness Tourism Package — Purchase Prediction")
st.write(
    """
    This application predicts whether a customer is likely to purchase the new
    **Wellness Tourism Package**, so the sales team can prioritise outreach
    *before* contacting them.

    Enter the customer details below and click **Predict**.
    """
)

# ---------------------------------------------------------------- Inputs: customer details
st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
    # Continuous inputs -> number boxes with sensible bounds
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    monthly_income = st.number_input(
        "Monthly Income", min_value=1000, max_value=100000, value=23000, step=500
    )
    # Text categories -> dropdowns (labels must match the training data exactly)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status", ["Married", "Single", "Divorced", "Unmarried"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

with col2:
    # Discrete numeric columns -> dropdowns with the fixed levels seen in training
    citytier = st.selectbox("City Tier", [1, 2, 3])
    num_persons = st.selectbox("Number of Persons Visiting", [1, 2, 3, 4, 5], index=2)
    num_children = st.selectbox("Number of Children Visiting (below 5)", [0, 1, 2, 3], index=1)
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    passport = st.selectbox("Holds a Valid Passport", ["No", "Yes"])
    own_car = st.selectbox("Owns a Car", ["No", "Yes"])
    num_trips = st.number_input(
        "Number of Trips per Year", min_value=0, max_value=30, value=3, step=1
    )

# ---------------------------------------------------------------- Inputs: interaction details
st.subheader("Interaction Details")
col3, col4 = st.columns(2)

with col3:
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    duration_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=60, value=15, step=1
    )

with col4:
    num_followups = st.selectbox("Number of Follow-ups", [1, 2, 3, 4, 5, 6], index=2)
    pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)

# ---------------------------------------------------------------- Assemble dataframe
# Single row with the same 18 column names used in training
# (order does not matter: ColumnTransformer selects columns by name)
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,     # Yes/No -> 0/1 coding used in training
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

# Collapsible panel showing exactly what is sent to the model
with st.expander("Review the input sent to the model"):
    st.dataframe(input_data)

# ---------------------------------------------------------------- Predict
if st.button("Predict"):
    prediction = int(model.predict(input_data)[0])                 # 0 or 1
    probability = float(model.predict_proba(input_data)[0][1])     # P(buyer)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"**Likely to purchase** the Wellness Tourism Package "
                   f"(probability: {probability:.2%})")
        st.write("Recommended action: prioritise this customer for outreach.")
    else:
        st.warning(f"**Unlikely to purchase** the Wellness Tourism Package "
                   f"(probability: {probability:.2%})")
        st.write("Recommended action: deprioritise, or target with a different offer.")

    # Clamp to [0, 1] as required by st.progress
    st.progress(min(max(probability, 0.0), 1.0))
