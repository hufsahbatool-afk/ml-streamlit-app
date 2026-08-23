import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Multi-Dataset ML Prediction Dashboard")

# Dataset selector
dataset = st.selectbox(
    "Choose a dataset:",
    ["Cervical Cancer", "IoTNet24 IDS", "Heart Disease", "FDA Adverse Events"]
)

# Model options (same 8 algorithms across all datasets)
model_options = [
    "Logistic Regression", "KNN", "Naive Bayes", "Decision Tree",
    "Random Forest", "SVM", "Gradient Boosting", "AdaBoost"
]
model_choice = st.selectbox("Choose a model:", model_options)

# Maps display name -> filename slug used when we saved the .pkl files
model_file_map = {
    "Logistic Regression": "logistic_regression",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "SVM": "svm",
    "Gradient Boosting": "gradient_boosting",
    "AdaBoost": "adaboost"
}

st.divider()

# ============================================
# CERVICAL CANCER DATASET
# ============================================
if dataset == "Cervical Cancer":
    st.header("Cervical Cancer Risk Prediction")
    st.write("Enter patient information below:")

    # Numeric inputs
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    partners = st.number_input("Number of sexual partners", min_value=0, max_value=30, value=2)
    first_intercourse = st.number_input("First sexual intercourse (age)", min_value=8, max_value=40, value=17)
    pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=15, value=1)
    smokes_years = st.number_input("Smokes (years)", min_value=0.0, max_value=50.0, value=0.0)
    smokes_packs = st.number_input("Smokes (packs/year)", min_value=0.0, max_value=50.0, value=0.0)
    hc_years = st.number_input("Hormonal Contraceptives (years)", min_value=0.0, max_value=40.0, value=0.0)
    iud_years = st.number_input("IUD (years)", min_value=0.0, max_value=30.0, value=0.0)
    stds_number = st.number_input("STDs (number)", min_value=0, max_value=10, value=0)

    # Binary (Yes/No) inputs -- converted to 1/0
    hormonal_contra = st.selectbox("Uses Hormonal Contraceptives?", ["No", "Yes"])
    iud = st.selectbox("Uses IUD?", ["No", "Yes"])
    std_condy = st.selectbox("STDs: condylomatosis?", ["No", "Yes"])
    std_vaginal = st.selectbox("STDs: vaginal condylomatosis?", ["No", "Yes"])
    std_syphilis = st.selectbox("STDs: syphilis?", ["No", "Yes"])
    std_hiv = st.selectbox("STDs: HIV?", ["No", "Yes"])
    std_hpv = st.selectbox("STDs: HPV?", ["No", "Yes"])

    if st.button("Predict"):
        # Build the feature array in the EXACT order used during training
        input_data = np.array([[
            age, partners, first_intercourse, pregnancies,
            smokes_years, smokes_packs,
            1 if hormonal_contra == "Yes" else 0, hc_years,
            1 if iud == "Yes" else 0, iud_years,
            stds_number,
            1 if std_condy == "Yes" else 0,
            1 if std_vaginal == "Yes" else 0,
            1 if std_syphilis == "Yes" else 0,
            1 if std_hiv == "Yes" else 0,
            1 if std_hpv == "Yes" else 0
        ]])

        # Load the scaler and the selected model
        scaler = joblib.load("notebook1/cervical_scaler.pkl")
        model_filename = f"notebook1/cervical_{model_file_map[model_choice]}.pkl"
        model = joblib.load(model_filename)

        # Scale the input the same way training data was scaled
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"Prediction: **Positive** (Biopsy likely positive)")
        else:
            st.success(f"Prediction: **Negative** (Biopsy likely negative)")
        st.write(f"Probability of positive result: **{probability*100:.1f}%**")
        # ============================================
# IOTNET24 IDS DATASET
# ============================================
elif dataset == "IoTNet24 IDS":
    st.header("IoT Network Intrusion Detection")
    st.write("Enter connection details below:")

    resp_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=80)
    proto = st.selectbox("Protocol", ["tcp", "udp"])
    duration = st.number_input("Duration", min_value=0.0, value=1.0)
    orig_bytes = st.number_input("Originator Bytes", min_value=0.0, value=0.0)
    resp_bytes = st.number_input("Responder Bytes", min_value=0.0, value=0.0)
    missed_bytes = st.number_input("Missed Bytes", min_value=0, value=0)
    orig_pkts = st.number_input("Originator Packets", min_value=0, value=1)
    orig_ip_bytes = st.number_input("Originator IP Bytes", min_value=0, value=0)
    resp_pkts = st.number_input("Responder Packets", min_value=0, value=0)
    resp_ip_bytes = st.number_input("Responder IP Bytes", min_value=0, value=0)
    conn_state = st.selectbox("Connection State", ["OTH", "RSTR", "S0", "S1", "S3", "SF"])

    if st.button("Predict"):
        # Encode proto using the saved encoder
        proto_encoder = joblib.load("notebook2/iotnet24_proto_encoder.pkl")
        proto_encoded = proto_encoder.transform([proto])[0]

        # One-hot encode conn_state manually (matches training format)
        conn_states = ["OTH", "RSTR", "S0", "S1", "S3", "SF"]
        conn_state_encoded = [1 if conn_state == cs else 0 for cs in conn_states]

        input_data = np.array([[
            resp_port, proto_encoded, duration, orig_bytes, resp_bytes,
            missed_bytes, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes,
            *conn_state_encoded
        ]])

        scaler = joblib.load("notebook2/iotnet24_scaler.pkl")
        label_encoder = joblib.load("notebook2/iotnet24_label_encoder.pkl")
        model_filename = f"notebook2/iotnet24_{model_file_map[model_choice]}.pkl"
        model = joblib.load(model_filename)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        predicted_label = label_encoder.inverse_transform([prediction])[0]

        st.divider()
        if predicted_label == "Malicious":
            st.error(f"Prediction: **{predicted_label}**")
        else:
            st.success(f"Prediction: **{predicted_label}**")
        st.write(f"Probability of Malicious: **{probability*100:.1f}%**")

# ============================================
# HEART DISEASE DATASET
# ============================================
elif dataset == "Heart Disease":
    st.header("Heart Disease Prediction")
    st.write("Enter patient information below:")

    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina?", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0)
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("Predict"):
        sex_encoder = joblib.load("notebook3/heart_sex_encoder.pkl")
        angina_encoder = joblib.load("notebook3/heart_angina_encoder.pkl")

        sex_encoded = sex_encoder.transform([sex])[0]
        angina_encoded = angina_encoder.transform([exercise_angina])[0]

        cp_options = ["ASY", "ATA", "NAP", "TA"]
        cp_encoded = [1 if chest_pain == c else 0 for c in cp_options]

        ecg_options = ["LVH", "Normal", "ST"]
        ecg_encoded = [1 if resting_ecg == e else 0 for e in ecg_options]

        slope_options = ["Down", "Flat", "Up"]
        slope_encoded = [1 if st_slope == s else 0 for s in slope_options]

        input_data = np.array([[
            age, sex_encoded, resting_bp, cholesterol,
            1 if fasting_bs == "Yes" else 0, max_hr, angina_encoded, oldpeak,
            *cp_encoded, *ecg_encoded, *slope_encoded
        ]])

        scaler = joblib.load("notebook3/heart_scaler.pkl")
        model_filename = f"notebook3/heart_{model_file_map[model_choice]}.pkl"
        model = joblib.load(model_filename)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"Prediction: **Heart Disease Likely**")
        else:
            st.success(f"Prediction: **Normal**")
        st.write(f"Probability of Heart Disease: **{probability*100:.1f}%**")

# ============================================
# FDA ADVERSE EVENTS DATASET
# ============================================
elif dataset == "FDA Adverse Events":
    st.header("FDA Adverse Drug Event - Fatality Prediction")
    st.write("Enter report information below:")

    patient_sex = st.selectbox("Patient Sex", ["Male", "Female"])
    patient_age = st.number_input("Patient Age", min_value=0, max_value=100, value=60)
    reaction_count = st.number_input("Number of Reactions Reported", min_value=1, max_value=100, value=3)

    drugs = ["AMLODIPINE", "ATENOLOL", "AVASTIN", "DIOVAN", "ERBITUX", "IBRANCE",
             "LETAIRIS", "LISINOPRIL", "NEXAVAR", "NORVASC", "OPSUMIT", "Other",
             "REMODULIN", "TRACLEER", "TYKERB"]
    drug_choice = st.selectbox("Drug Name", drugs)

    indications = ["BREAST CANCER", "BREAST CANCER FEMALE", "BREAST CANCER METASTATIC",
                   "CARDIAC DISORDER", "CARDIAC FAILURE", "COLON CANCER", "COLORECTAL CANCER",
                   "HEPATIC NEOPLASM MALIGNANT", "HYPERTENSION", "LUNG NEOPLASM MALIGNANT",
                   "NON-SMALL CELL LUNG CANCER", "OVARIAN CANCER", "Other", "PROSTATE CANCER",
                   "PULMONARY ARTERIAL HYPERTENSION", "PULMONARY HYPERTENSION"]
    indication_choice = st.selectbox("Drug Indication", indications)

    if st.button("Predict"):
        sex_encoder = joblib.load("notebook4/fda_sex_encoder.pkl")
        fatal_encoder = joblib.load("notebook4/fda_fatal_encoder.pkl")

        sex_encoded = sex_encoder.transform([patient_sex])[0]

        drug_encoded = [1 if drug_choice == d else 0 for d in drugs]
        indication_encoded = [1 if indication_choice == i else 0 for i in indications]

        input_data = np.array([[
            sex_encoded, patient_age, reaction_count,
            *drug_encoded, *indication_encoded
        ]])

        scaler = joblib.load("notebook4/fda_scaler.pkl")
        model_filename = f"notebook4//fda_{model_file_map[model_choice]}.pkl"
        model = joblib.load(model_filename)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        predicted_label = fatal_encoder.inverse_transform([prediction])[0]

        st.divider()
        if predicted_label == "Yes":
            st.error(f"Prediction: **Fatal**")
        else:
            st.success(f"Prediction: **Non-Fatal**")
        st.write(f"Probability of Fatal outcome: **{probability*100:.1f}%**")
        st.caption("⚠️ Note: this model's predictive performance is limited (F1 ≈ 0.41) — treat results as indicative only, not a reliable diagnostic tool.")