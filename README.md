# 🏦 Bank Term Deposit Subscription Prediction

## 📌 Project Overview

This project predicts whether a customer will subscribe to a term deposit using Machine Learning techniques. It helps banks identify high-potential customers and improve marketing efficiency.

---

## 🎯 Objective

* Predict customer subscription behavior
* Reduce marketing cost
* Improve campaign success rate
* Support business decision-making

---

## 🤖 Models Used

* XGBoost (Final Best Model)
* Feature Engineering applied
* Threshold tuning (0.36 instead of default 0.5)

---

## 📊 Results

* High Accuracy
* Balanced F1 Score
* Business-friendly output:

  * High Potential
  * Medium Potential
  * Low Potential

---

## 📂 Dataset

* Bank Marketing Dataset
* Features used:

  * Age
  * Job
  * Marital Status
  * Education
  * Balance
  * Campaign
  * Previous Outcome
  * Loan / Housing
  * Contact Type

---

## 🛠 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit (Frontend Dashboard)
* FastAPI (Backend API)
* Plotly (Visualization)
* Joblib (Model Saving)

---

## 🚀 How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run FastAPI backend

```bash
uvicorn api:app --reload
```

### Step 3: Run Streamlit app

```bash
streamlit run app.py
```

---

## 🌐 Access Links

* Streamlit App → http://localhost:8501
* API Docs → http://127.0.0.1:8000/docs

---

## 🔥 Features

* 📊 Interactive Dashboard
* 🎯 Live Prediction using API
* 📈 Advanced Charts & Insights
* 🤖 AI-style Assistant
* 🔄 What-if Scenario Simulation
* 🔊 Voice Output for Prediction
* 📋 Downloadable Data

---

## 📊 Sample Prediction

Input:

```json
{
  "age": 35,
  "job": "management",
  "marital": "single",
  "education": "tertiary",
  "default": "no",
  "balance": 5000,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "sep",
  "campaign": 1,
  "pdays": 5,
  "previous": 2,
  "poutcome": "success"
}
```

Output:

* Probability: 72%
* Prediction: Subscribe
* Segment: High Potential

---

## ⚠️ Important Note

* Model works best with realistic input data
* Short or incomplete inputs may give neutral or low-confidence predictions
* Threshold is tuned for business performance

---

## 📁 Project Structure

```
Bank-Term-Deposit-App/
│
├── app.py
├── api.py
├── bank_model.pkl
├── preprocessor.pkl
├── bank-full.csv
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 👩‍💻 Author

**Priyadharshini**
Data Science & Machine Learning Developer

---

## 💡 Future Improvements

* Deploy to cloud (Render / AWS)
* Add user authentication
* Real-time data integration
* Advanced explainable AI (SHAP)
