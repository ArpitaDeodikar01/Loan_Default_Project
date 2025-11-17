AIML MINI PROJECT GROUP 8


👥 Project Contributors

This project was collaboratively developed by the following team of students:

Advika Khorgade (UCE2023502)

Arpita Deodikar (UCE2023515)

Jui Kulkarni (UCE2023532)

PPT LINK - https://drive.google.com/file/d/1MPl6ut3UILGd-30BEGkkeAdSN1T-y2J6/view?usp=sharing

[Uploading LoanLens A Multi-Stage Machine Learning Framework for Credit Risk Assessment and Personalized Financial Recommendation.pdf…]()


---

# **LoanLens – Loan Default Prediction & Financial Recommendation System**

LoanLens is an end-to-end **FinTech Machine Learning application** designed to predict loan default risk, segment customers based on financial behavior, recommend suitable banking products, and support users through an integrated **Gemini-powered financial chatbot**.
The system combines **predictive**, **prescriptive**, and **descriptive analytics** into a unified, user-friendly platform built with **Flask**, **HTML/CSS**, **JavaScript**, and
production-ready ML models.

---

# **📌 1. Project Overview**

LoanLens solves key challenges in digital lending:

* **Predicting the probability of loan default** using a Random Forest model
* **Segmenting borrowers into risk groups** using K-Means clustering
* **Recommending suitable bank/loan products** via an XGBoost classifier
* **Providing financial guidance** through a FinTech-specific Gemini API chatbot

The objective is to deliver a reliable, interpretable, and scalable solution that modern lenders can integrate into real-world credit workflows.

---

# **📌 2. Features Implemented**

### **✔ Loan Default Prediction (Random Forest)**

* Predicts probability of a borrower defaulting
* Produces risk tiers (Low / Medium / High)
* Based on financial, demographic, and credit-related attributes

### **✔ Customer Segmentation (K-Means)**

* Computes a custom **RiskScore** using normalized financial variables
* Groups borrowers into 6 clusters
* Helps understand borrower risk distribution

### **✔ Bank/Product Recommendation (XGBoost)**

* Uses engineered ratios such as

  * Loan-to-Income Ratio
  * Credit-to-Income Ratio
  * Affordability Score
* Provides the most suitable bank recommendation
* Includes fallback logic for unmatched profiles

### **✔ Gemini API Chatbot**

* Answers **FinTech-related queries only**
* Helps users understand credit terms, LoanLens outputs, model logic, etc.
* Supports real-time interaction on the UI

### **✔ Flask-Based Web Application**

* Interactive, responsive UI
* Modular backend with reusable model pipelines
* Real-time inference

---

# **📌 3. Technology Stack**

### **Frontend**

* HTML5
* CSS3
* JavaScript

### **Backend**

* Python Flask
* REST API architecture

### **Machine Learning**

* **Random Forest Classifier**
* **K-Means Clustering**
* **XGBoost Classifier**
* Scikit-Learn
* Joblib for model serialization

### **Chatbot**

* Gemini API (Google Generative AI)

---

# **📌 4. Project Structure**

```
LoanLens/
│
├── app.py                     # Flask backend + API routes
├── Train_models.py            # Training RF, XGB, KMeans models
├── clustering.py              # RiskScore & K-Means logic
├── Inspect_columns.py         # Dataset validation utility
│
├── models/                    # Saved trained models (.pkl/.joblib)
│     ├── random_forest.pkl
│     ├── xgboost_model.pkl
│     └── kmeans_model.pkl
│
├── data/                      # Datasets (loan_data.csv, bank_data.csv)
│
├── templates/                 # HTML templates for UI
│     ├── index.html
│     ├── result.html
│     ├── chatbot.html
│
├── static/                    # CSS, JS, media
│     ├── css/
│     └── js/
│
├── .env                       # Contains GEMINI_API_KEY
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

# **📌 5. How to Run LoanLens on Any PC**

### **Step 1 — Clone the Repository**

```bash
git clone https://github.com/ArpitaDeodikar01/Loan_Default_Project.git
cd Loan_Default_Project
```

### **Step 2 — Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3 — Add Your Gemini API Key**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

### **Step 4 — Train the Models (Optional)**

Models are pre-trained, but you can retrain:

```bash
python Train_models.py
```

### **Step 5 — Run the Flask Application**

```bash
python app.py
```

### **Step 6 — Open in Browser**

Go to:

```
http://127.0.0.1:5000/
```

You can now:

* Enter borrower details
* View default probability
* Check risk cluster and recommendation
* Chat using the Gemini-powered chatbot

---

# **📌 6. Future Enhancements**

* Add SHAP-based explainability
* Include time-series risk modeling
* Deploy to cloud (AWS / Render / Railway)
* Add user authentication for enterprise use

---
