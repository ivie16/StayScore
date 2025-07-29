import streamlit as st
import pickle
import pandas as pd
#from sklearn.utils import column_or_1d

#import helper

# Load encoders, scaler, and trained model
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("Logistic Regression_model.pkl", "rb"))

attrition_encoders = pickle.load(open("attrition_encoder.pkl", "rb"))
attrition_scaler = pickle.load(open("attrition_scaler.pkl", "rb"))
attrition_model = pickle.load(open("attrition_model.pkl", "rb"))
categorical_cols = pickle.load(open("categorical_cols.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Sidebar Navigation with reduced width
st.sidebar.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            width: 300px !important;
            min-width: 200px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Customer Churn", "Employee Attrition"])

if page == "Customer Churn":
    # Custom styles
    st.markdown(
        """
        <style>
            .main {
                background-color: #f5f5f5;
            }
            .stButton>button {
                background-color: #FF4B4B;
                color: white;
                font-size: 18px;
                border-radius: 10px;
                padding: 10px;
                display: block;
                margin: 0 auto;
            }
            .stButton>button:hover {
                background-color: #FF0000;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title with styling
    st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Customer Churn Prediction</h1>",
                unsafe_allow_html=True)

    # User input fields in columns
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], index=0)
        Partner = st.selectbox("Partner", ["Yes", "No"], index=1)
        Dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
        tenure = st.number_input("Tenure", min_value=0, step=1)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"], index=0)
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=1)
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=0)
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"], index=1)
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"], index=1)

    with col2:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"], index=1)
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"], index=1)
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"], index=1)
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"], index=1)
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                        "Credit card (automatic)"], index=0)
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    st.markdown("---")

    # Prediction button
    if st.button("Predict Churn"):
        input_data = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        input_df = pd.DataFrame([input_data])

        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[[column]])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        pred_prob = model.predict_proba(input_scaled)[0][1]

        result = "Churn" if prediction == 1 else "No Churn"

        # Display results with styling
        st.markdown(
            f"<h2 style='text-align: center; color: {'#FF0000' if prediction == 1 else '#2E8B57'};'>Prediction: {result}</h2>",
            unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Probability: {round(pred_prob * 100, 2)}%</h3>",
                    unsafe_allow_html=True)

elif page == "Employee Attrition":
    st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Employee Attrition Prediction</h1>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=60, value=30)
        BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        DailyRate = st.number_input("Daily Rate", min_value=0, value=800)
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        DistanceFromHome = st.number_input("Distance From Home", min_value=0, value=5)
        Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        EducationField = st.selectbox("Education Field",
                                      ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources",
                                       "Other"])
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                            "Manufacturing Director", "Healthcare Representative", "Manager",
                                            "Sales Representative", "Research Director", "Human Resources"])
        JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)

    with col2:
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=5000)
        NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0, value=1)
        OverTime = st.selectbox("OverTime", ["Yes", "No"])
        PercentSalaryHike = st.number_input("Percent Salary Hike", min_value=0, value=15)
        PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        TotalWorkingYears = st.number_input("Total Working Years", min_value=0, value=5)
        TrainingTimesLastYear = st.number_input("Training Times Last Year", min_value=0, value=2)
        WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        YearsAtCompany = st.number_input("Years at Company", min_value=0, value=3)
        YearsInCurrentRole = st.number_input("Years in Current Role", min_value=0, value=2)
        YearsWithCurrManager = st.number_input("Years with Current Manager", min_value=0, value=2)
    st.markdown("---")

    if st.button("Predict Attrition"):
        input_data = {
            'Age': Age,
            'BusinessTravel': BusinessTravel,
            'DailyRate': DailyRate,
            'Department': Department,
            'DistanceFromHome': DistanceFromHome,
            'Education': Education,
            'EducationField': EducationField,
            'EnvironmentSatisfaction': EnvironmentSatisfaction,
            'Gender': Gender,
            'JobInvolvement': JobInvolvement,
            'JobLevel': JobLevel,
            'JobRole': JobRole,
            'JobSatisfaction': JobSatisfaction,
            'MaritalStatus': MaritalStatus,
            'MonthlyIncome': MonthlyIncome,
            'NumCompaniesWorked': NumCompaniesWorked,
            'OverTime': OverTime,
            'PercentSalaryHike': PercentSalaryHike,
            'PerformanceRating': PerformanceRating,
            'RelationshipSatisfaction': RelationshipSatisfaction,
            'StockOptionLevel': StockOptionLevel,
            'TotalWorkingYears': TotalWorkingYears,
            'TrainingTimesLastYear': TrainingTimesLastYear,
            'WorkLifeBalance': WorkLifeBalance,
            'YearsAtCompany': YearsAtCompany,
            'YearsInCurrentRole': YearsInCurrentRole,
            'YearsSinceLastPromotion': YearsSinceLastPromotion,
            'YearsWithCurrManager': YearsWithCurrManager
        }

        input_df = pd.DataFrame([input_data])

        # Separate categorical and numerical
        cat_df = input_df[categorical_cols]
        num_df = input_df.drop(columns=categorical_cols)

        # Encode categorical
        cat_encoded = attrition_encoders.transform(cat_df)  # No .toarray()

        # Create DataFrame with encoded feature names
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=attrition_encoders.get_feature_names_out(categorical_cols))

        # Combine
        final_df = pd.concat([num_df.reset_index(drop=True), cat_encoded_df], axis=1)

        # Reorder columns
        final_df = final_df.reindex(columns=feature_columns, fill_value=0)

        # Scale
        final_scaled = attrition_scaler.transform(final_df)

        # Predict
        prediction = attrition_model.predict(final_scaled)[0]
        pred_prob = attrition_model.predict_proba(final_scaled)[0][1]

        result = "Yes (Will Leave)" if prediction == 1 else "No (Will Stay)"

        st.markdown(
            f"<h2 style='text-align: center; color: {'#FF0000' if prediction == 1 else '#2E8B57'};'>Prediction: {result}</h2>",
            unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Probability: {round(pred_prob * 100, 2)}%</h3>",
                    unsafe_allow_html=True)