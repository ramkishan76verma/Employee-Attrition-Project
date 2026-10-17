import streamlit as st
import pandas as pd
import joblib

model = joblib.load("attrition_model.pkl")

st.title("Employee Attrition Prediction System")
st.write("Predict whether an employee is at risk of leaving the company.")

st.sidebar.header("Employee Details")

age = st.sidebar.number_input("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.number_input("Distance From Home", 1, 30, 5)

job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)

overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
business_travel = st.sidebar.selectbox(
    "Business Travel",
    ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
)

job_role = st.sidebar.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]
)

if st.button("Predict Attrition Risk"):

    data = pd.read_csv("dataset.csv")

    employee = data.iloc[[0]].copy()

    employee["Age"] = age
    employee["MonthlyIncome"] = monthly_income
    employee["DistanceFromHome"] = distance
    employee["JobSatisfaction"] = job_satisfaction
    employee["EnvironmentSatisfaction"] = environment_satisfaction
    employee["OverTime"] = overtime
    employee["BusinessTravel"] = business_travel
    employee["JobRole"] = job_role

    employee = employee.drop("Attrition", axis=1)

    prediction = model.predict(employee)[0]
    probability = model.predict_proba(employee)[0][1]

    if prediction == 1:
        st.error("⚠️ High Attrition Risk")
    else:
        st.success("✅ Low Attrition Risk")

    st.write("Attrition Probability:", round(probability * 100, 2), "%")