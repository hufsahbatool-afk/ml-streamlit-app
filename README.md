
# 🧠 DiagnoSense

### A Multi-Domain Diagnostic & Detection ML Classifier

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ml-app-app-frnkfbb2bucpfnt838esck.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

**DiagnoSense** is a multi-domain machine learning application that brings together four independent classification projects into a single interactive Streamlit platform.

The project demonstrates how the same 8 machine learning algorithms, evaluated through a consistent preprocessing, training, and hyperparameter-tuning methodology, perform across genuinely different real-world domains — from medical risk prediction to network intrusion detection.

> **Note:** DiagnoSense is an educational and research-oriented machine learning project. Its predictions should not be interpreted as professional medical, security, or clinical advice.

---

## 🚀 Live Application

### [Open DiagnoSense →](https://ml-app-app-frnkfbb2bucpfnt838esck.streamlit.app/)

The deployed application provides an interactive interface for all four datasets, letting users pick a dataset, select a trained model, enter input values, and receive a live prediction with confidence probability.

---

## 📌 Project Overview

DiagnoSense contains four binary classification case studies:

| # | Case Study | Domain | Target |
|---|---|---|---|
| 1 | 🩺 **Cervical Cancer Risk Prediction** | Healthcare | Biopsy result (Positive / Negative) |
| 2 | 🌐 **IoTNet24 Intrusion Detection** | Cybersecurity | Network traffic (Malicious / Benign) |
| 3 | ❤️ **Heart Disease Prediction** | Healthcare | Diagnosis (Disease / Normal) |
| 4 | 💊 **FDA Adverse Drug Event Prediction** | Healthcare | Outcome (Fatal / Non-Fatal) |

All four are binary classification tasks, evaluated with the same 8 algorithms and the same rigorous before/after tuning methodology — allowing a genuine comparison of how model performance depends on the underlying data, not just the algorithm.

---

# 🩺 1. Cervical Cancer Risk Prediction

### Problem
Predict whether a patient's biopsy will confirm cervical cancer, based on demographic, lifestyle, and medical history risk factors.

### Dataset
1,670 patient records · 16 input features after cleaning · Kaggle: Cervical Cancer Risk Factors Combined Dataset

### Target Classes
- 🟢 **Negative** (93.5%)
- 🔴 **Positive** (6.5%)

### Models Compared
Logistic Regression · KNN · Naive Bayes · Decision Tree · Random Forest · SVM · Gradient Boosting · AdaBoost

### Best Model
**Random Forest**
- F1-Score: **0.952**
- Recall: **0.909**
- ROC-AUC: **0.996**

---

# 🌐 2. IoTNet24 Intrusion Detection

### Problem
Classify IoT network connections as malicious or benign based on connection statistics (port, protocol, connection state, packet/byte counts).

### Dataset
23,145 raw records, 7,821 after deduplication · 16 input features · Kaggle: IoTNet24 Dataset for IDS

### Target Classes
- 🔴 **Malicious** (91.7%)
- 🟢 **Benign** (8.3%)

### Models Compared
Logistic Regression · KNN · Naive Bayes · Decision Tree · Random Forest · SVM · Gradient Boosting · AdaBoost

### Best Model
**Gradient Boosting**
- F1-Score: **0.999**
- Recall: **0.999**
- ROC-AUC: **1.000**

---

# ❤️ 3. Heart Disease Prediction

### Problem
Predict the presence of heart disease from patient clinical data and exercise stress test results.

### Dataset
918 patient records, combining 5 clinical sources (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog) · 18 input features after encoding · Kaggle/UCI: Heart Failure Prediction Dataset

### Target Classes
- 🔴 **Heart Disease** (55.3%)
- 🟢 **Normal** (44.7%)

### Models Compared
Logistic Regression · KNN · Naive Bayes · Decision Tree · Random Forest · SVM · Gradient Boosting · AdaBoost

### Best Model
**Random Forest**
- F1-Score: **0.903**
- Recall: **0.912**
- ROC-AUC: **0.928**

---

# 💊 4. FDA Adverse Drug Event Prediction

### Problem
Predict whether an FDA-reported adverse drug event (for oncology and cardiology medications) resulted in patient death.

### Dataset
13,955 raw reports, 6,130 after deduplication · 34 input features after encoding · Kaggle: FDA Adverse Drug Events (Oncology & Cardiology)

### Target Classes
- 🟢 **Non-Fatal** (78.6%)
- 🔴 **Fatal** (21.4%)

### Models Compared
Logistic Regression · KNN · Naive Bayes · Decision Tree · Random Forest · SVM · Gradient Boosting · AdaBoost

### Best Model
**Logistic Regression**
- F1-Score: **0.414**
- Recall: **0.626**
- ROC-AUC: **0.670**

> This is the weakest-performing dataset in the project, and deliberately reported as-is: feature-target correlations were markedly weaker here than in the other three datasets, indicating that basic demographic and drug-category data alone are insufficient to reliably predict fatality.

---

# 🔬 Machine Learning Workflow

Across all four case studies, the same pipeline was followed:

```
Raw Dataset
     ↓
Data Loading
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning (missing values, disguised zeros, unit errors, duplicates)
     ↓
Feature Engineering (leakage removal, redundancy removal)
     ↓
Encoding (Label / One-Hot)
     ↓
Feature Scaling (RobustScaler)
     ↓
Stratified Train-Test Split
     ↓
Baseline Model Training (8 algorithms)
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning (GridSearchCV)
     ↓
Final Model Selection (baseline retained if tuning underperformed)
     ↓
Model Serialization (joblib)
     ↓
Streamlit Deployment
```

Trained models, scalers, and encoders are stored in the `notebook1/` – `notebook4/` directories and loaded directly by the Streamlit application.

---

# 📊 Evaluation

Classification performance was assessed using multiple metrics rather than Accuracy alone, since 3 of the 4 datasets are meaningfully imbalanced:

- Accuracy
- Precision
- Recall
- **F1-Score** (primary selection metric)
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

Using multiple metrics — rather than Accuracy alone — was essential: on the Cervical Cancer dataset, AdaBoost achieved 93% Accuracy while never once correctly predicting a positive case (F1 = 0), which only F1-Score and Recall exposed.

---

# 🖥️ Streamlit Application

DiagnoSense converts four independent notebook-based ML projects into a single interactive web application.

### Application Flow

```
DiagnoSense
│
├── Select Dataset   → Cervical Cancer / IoTNet24 / Heart Disease / FDA Adverse Events
│
├── Select Model     → Choose from 8 tuned algorithms
│
├── Enter Inputs     → Patient / connection data via form fields
│
└── Get Prediction   → Predicted class + probability
```

---

# 📁 Repository Structure

```
DiagnoSense/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── notebook1/              # Cervical Cancer models, scaler, feature info
├── notebook2/              # IoTNet24 models, scaler, encoders, feature info
├── notebook3/              # Heart Disease models, scaler, encoders, feature info
├── notebook4/              # FDA Adverse Events models, scaler, encoders, feature info
│
├── notebooks/
│   ├── Cervical_Cancer_Classification.ipynb
│   ├── IoTNet24_Classification.ipynb
│   ├── Heart_Disease_Classification.ipynb
│   └── FDA_Adverse_Events_Classification.ipynb
│
├── documentation/
│   └── ML_Project_Documentation.docx
│
└── presentation/
    └── ML_Project_Presentation.pptx
```

---

# 📓 Project Notebooks

The complete experimental workflows are available in the [`notebooks/`](./notebooks) directory. Click a badge below to open any notebook directly in Google Colab, or use the GitHub link to view it in-browser.

| Notebook | View on GitHub | Open in Colab |
|---|---|---|
| Cervical Cancer Risk Classification | [View](./notebooks/Cervical_Cancer_Classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hufsahbatool-afk/ml-streamlit-app/blob/main/notebooks/Cervical_Cancer_Classification.ipynb) |
| IoTNet24 Intrusion Detection | [View](./notebooks/IoTNet24_Classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hufsahbatool-afk/ml-streamlit-app/blob/main/notebooks/IoTNet24_Classification.ipynb) |
| Heart Disease Classification | [View](./notebooks/Heart_Disease_Classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hufsahbatool-afk/ml-streamlit-app/blob/main/notebooks/Heart_Disease_Classification.ipynb) |
| FDA Adverse Drug Event Classification | [View](./notebooks/FDA_Adverse_Events_Classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hufsahbatool-afk/ml-streamlit-app/blob/main/notebooks/FDA_Adverse_Events_Classification.ipynb) |

Each notebook contains the full data preparation, exploratory analysis, cleaning, feature engineering, model training, evaluation, and hyperparameter tuning performed for that dataset.

---

# 📄 Documentation

The complete project documentation is available here:

📘 [DiagnoSense Project Documentation](./documentation/ML_Project_Documentation.docx)

The documentation provides a detailed explanation of the datasets, preprocessing methodology, machine learning methods, and complete before/after tuning results — including individual confusion matrices, ROC curves, and Precision-Recall curves for every model.

---

# 🎤 Presentation

The project presentation is available here:

📊 [DiagnoSense Presentation](./presentation/ML_Project_Presentation.pptx)

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/hufsahbatool-afk/ml-streamlit-app.git
cd ml-streamlit-app
```

Create a virtual environment:

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 🛠️ Technologies Used

**Programming:** Python

**Data Science:** NumPy, Pandas

**Machine Learning:** Scikit-learn, Joblib

**Visualization:** Matplotlib, Seaborn

**Web Application:** Streamlit

**Development:** Google Colab, VS Code, Git, GitHub

**Deployment:** Streamlit Community Cloud

---

# 💡 Key Learning Outcomes

DiagnoSense demonstrates practical, hands-on experience with:

- End-to-end machine learning pipelines across multiple real-world domains
- Exploratory data analysis and data quality investigation
- Data cleaning: disguised missing values, unit-inconsistency errors, duplicate investigation
- Feature engineering and data-leakage prevention
- Categorical encoding (Label and One-Hot)
- Feature scaling for skewed, outlier-containing data
- Training and comparing 8 classification algorithms per dataset
- Hyperparameter tuning with GridSearchCV, including recognizing when tuning underperforms baseline
- Classification evaluation beyond Accuracy (F1, Recall, ROC-AUC, Precision-Recall)
- Model serialization and deployment
- Building and deploying a multi-page ML prediction web application

---

# ⚠️ Disclaimer

DiagnoSense is developed for **educational and academic purposes**.

The predictions generated by this application are based on machine learning models trained on historical, publicly available datasets. They are **not validated clinical or security tools** and should not be used for medical diagnosis, treatment decisions, or real-world intrusion detection.

---

# 👩‍💻 Author

**Hafsa Batool**

Computer Science Student | Machine Learning Project

GitHub: @hufsahbatool(https://github.com/hufsahbatool)

---

### 🚀 Try the Application

**[Launch DiagnoSense →](https://ml-app-app-frnkfbb2bucpfnt838esck.streamlit.app/)**
