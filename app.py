import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
































# ===========================================================
# ✅ Environment Setup
# ===========================================================
















# Will be set after BASE is defined
































#print("🔑 Loaded GEMINI_API_KEY:", "✅ Found" if GEMINI_API_KEY else "❌ Missing")
































app = Flask(__name__)
# NOTE: The secret key provided by the user is a generic API key structure. Using a placeholder for security.
app.secret_key = "super-secret-key-for-session-management"
































BASE = os.path.dirname(os.path.abspath(__file__))




# Load .env file from the project root directory explicitly
env_path = os.path.join(BASE, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env file from: {env_path}")
else:
    load_dotenv()  # Fallback to default location
    print(f"⚠️ .env file not found at {env_path}, using default location")




# Set Gemini API variables after ensuring .env is loaded
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCkQLhMPMTdF8dmmFfOUceFT8mVhahl7ls")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")




# Debug: Print what we loaded
print(f"🔑 GEMINI_API_KEY: {'✅ Found' if GEMINI_API_KEY else '❌ Missing'} (first 10 chars: {GEMINI_API_KEY[:10] if GEMINI_API_KEY else 'N/A'}...)")
print(f"🤖 GEMINI_MODEL: {GEMINI_MODEL}")
































# ===========================================================
# ✅ Data and Model Loading
# ===========================================================
















MODELS_DIR = os.path.join(BASE, "models")
















DATA_PATH = os.path.join(BASE, "data", "loan_data.csv")
BANK_DATA_PATH = os.path.join(BASE, "data", "bank_data.csv")
USERS_FILE = os.path.join(BASE, "users.json")
PORTFOLIO_FILE = os.path.join(BASE, "user_portfolio.json")
















if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Data file not found at {DATA_PATH}")
















try:
    df = pd.read_csv(DATA_PATH, engine='python', encoding='utf-8', on_bad_lines='skip')
    print(f"✅ Loaded {len(df)} rows from loan_data.csv")
except Exception as e:
    print(f"❌ Error loading loan_data.csv: {e}")
    try:
        df = pd.read_csv(DATA_PATH, engine='python', encoding='latin-1', on_bad_lines='skip')
        print(f"✅ Loaded {len(df)} rows with latin-1 encoding")
    except Exception as e2:
        raise SystemExit(f"Failed to load data file: {e2}")
































CAT_COLS = ['Education', 'EmploymentType', 'MaritalStatus',
            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
NUM_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore',
            'MonthsEmployed', 'NumCreditLines', 'InterestRate',
            'LoanTerm', 'DTIRatio']
































ALL_FEATURES = NUM_COLS + CAT_COLS
































label_encoders = {}
category_options = {}
for c in CAT_COLS:
    le = LabelEncoder()
    df[c] = df[c].astype(str).str.strip() # Strip whitespace from data frame values
    # Ensure categories cover all unique values present, even NaNs as strings
    df[c + "_enc"] = le.fit_transform(df[c])
    label_encoders[c] = le
    # Clean up the options list for robust frontend use
    options = [opt.strip() for opt in le.classes_.astype(str)]
    category_options[c] = list(set(options))
































def try_load(name):
    """Try loading a model file with different extensions"""
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None
































# Load primary prediction models
rf_model = try_load("random_forest_model.joblib")
scaler = try_load("scaler.joblib")
label_encoders_loaded = try_load("encoders.joblib")
















# Load clustering models and new metadata
kmeans_clustering = try_load("kmeans_clustering.joblib")
cluster_preprocessor = try_load("cluster_preprocessor.joblib")
cluster_risk_labels = try_load("cluster_risk_labels.joblib")
cluster_colors_map = try_load("cluster_colors.joblib")
risk_scaler_params = try_load("risk_scaler_params.joblib") # NEW
cluster_feature_names = try_load("cluster_feature_names.joblib") # NEW








# Load XGBoost bank recommendation model
xgb_bank_model = try_load("xgb_bank_model.joblib")
bank_label_encoders = try_load("bank_label_encoders.joblib")
xgb_feature_columns = try_load("xgb_feature_columns.joblib")








# Load bank data if it exists
bank_df = None
if os.path.exists(BANK_DATA_PATH):
    try:
        bank_df = pd.read_csv(BANK_DATA_PATH, engine='python', encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Loaded {len(bank_df)} rows from bank_data.csv")
    except Exception as e:
        print(f"⚠️ Could not load bank_data.csv: {e}")
        bank_df = None
















print("🔹 RF Model:", "✅ Loaded" if rf_model else "❌ Not found")
print("🔹 Scaler:", "✅ Loaded" if scaler else "❌ Not found")
print("🔹 KMeans Clustering:", "✅ Loaded" if kmeans_clustering else "❌ Not found")
print("🔹 Cluster Preprocessor:", "✅ Loaded" if cluster_preprocessor else "❌ Not found")
print("🔹 Risk Scaler Params:", "✅ Loaded" if risk_scaler_params else "❌ Not found")
print("🔹 XGBoost Bank Model:", "✅ Loaded" if xgb_bank_model else "❌ Not found")
print("🔹 Bank Label Encoders:", "✅ Loaded" if bank_label_encoders else "❌ Not found")
print("🔹 XGBoost Feature Columns:", "✅ Loaded" if xgb_feature_columns else "❌ Not found")
































# ===========================================================
# ✅ User Management (UNCHANGED)
# ===========================================================
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)
































def save_users(u):
    with open(USERS_FILE, 'w') as f:
        json.dump(u, f)




# ===========================================================
# ✅ Portfolio Storage (Per-User Prediction History)
# ===========================================================
def load_portfolio_history():
    """Load per-user prediction history from disk."""
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
            # Ensure structure is always a dict
            return data if isinstance(data, dict) else {}
    except Exception:
        # On any error, fall back to empty to avoid breaking the app
        return {}




def save_portfolio_history(history):
    """Persist portfolio history to disk."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(history, f)
    except Exception as e:
        print(f"⚠️ Could not save portfolio history: {e}")




# In-memory cache of portfolio; persisted on each update for simplicity
portfolio_history = load_portfolio_history()
































# ===========================================================
# ✅ Routes (UNCHANGED LOGIN/LOGOUT)
# ===========================================================
@app.route("/")
def index():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))
































@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        users = load_users()
        if username in users and users[username]["password"] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")
































@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        users = load_users()
        if username in users:
            return render_template("register.html", error="User already exists")
        users[username] = {"password": password}
        save_users(users)
        session["user"] = username
        return redirect(url_for("dashboard"))
    return render_template("register.html")
































@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))
































@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect(url_for("login"))
   
    # Get all categories for dropdowns in dashboard.html
    # This is critical for the Find Banks tab dropdowns
    bank_categories = {
        # Using the unique values from the loan data to populate dashboard dropdowns
        'Education': list(df['Education'].unique().astype(str)),
        'EmploymentType': list(df['EmploymentType'].unique().astype(str)),
        # Add other categories needed for bank filtering if necessary
    }
   
    return render_template("dashboard.html", categories=category_options, features=ALL_FEATURES, bank_categories=bank_categories)
































# ===========================================================
# ✅ Helper Functions (UNCHANGED)
# ===========================================================
def encode_input(data):
    row = []
    for col in NUM_COLS:
        val = data.get(col)
        row.append(float(val) if val not in [None, ""] else np.nan)
    for col in CAT_COLS:
        # Use .strip() for robust handling of user input categories
        v = str(data.get(col, "")).strip()
        le = label_encoders.get(col)
        try:
            # Transform the stripped user input
            row.append(int(le.transform([v])[0]))
        except Exception:
            # Fallback to the first category (index 0) if the value is unknown
            # This is a common practice for handling OOV/unknown categories.
            row.append(0)
    return np.array([row])
































# ===========================================================
# ✅ Prediction Endpoint (UNCHANGED)
# ===========================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not rf_model or not scaler:
        return jsonify({"error": "Models not found. Train first."}), 500
















    payload = request.get_json() or request.form.to_dict()
    X_raw = encode_input(payload)
    X_scaled = scaler.transform(X_raw)
















    # Use rf_model for prediction
    prob = float(rf_model.predict_proba(X_scaled)[0][1])
    loan_amount = float(payload.get("LoanAmount", 0))
    income = float(payload.get("Income", 1))
    monthly_capacity = max(1, income * 0.20) / 12
    est_months = int(max(6, min(loan_amount / monthly_capacity, 360)))
















    monthly_rate = float(payload.get("InterestRate", 0)) / 100 / 12
    if monthly_rate > 0:
        denom = 1 - (1 + monthly_rate) ** (-est_months)
        monthly_payment = loan_amount * (monthly_rate / denom)
    else:
        monthly_payment = loan_amount / est_months
















    schedule, outstanding = [], loan_amount
    for m in range(1, est_months + 1):
        interest = outstanding * monthly_rate
        principal = monthly_payment - interest
        outstanding = max(0, outstanding - principal)
        schedule.append({"month": m, "payment": round(monthly_payment, 2), "outstanding": round(outstanding, 2)})
















    # Determine risk level (Updated thresholds)
    # Low risk: 0-20%, Medium risk: 20-30%, High risk: >= 30%
    if prob < 0.20:
        risk_level = "Low"
        suggestion = "Low risk — standard repayment"
    elif prob < 0.30:
        risk_level = "Medium"
        suggestion = "Medium risk — consider consolidation"
    else:
        risk_level = "High"
        suggestion = "High risk — consider refinancing or restructuring"








    result_payload = {
        "probability": round(prob, 4),
        "risk_level": risk_level,
        "estimated_months": est_months,
        "monthly_payment": round(monthly_payment, 2),
        "schedule": schedule,
        "suggestion": suggestion
    }


    # Store this prediction in the per-user portfolio history (not visible elsewhere)
    try:
        username = session.get("user")
        if username:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input": payload,
                "prediction": {
                    "probability": result_payload["probability"],
                    "risk_level": result_payload["risk_level"],
                    "estimated_months": result_payload["estimated_months"],
                    "monthly_payment": result_payload["monthly_payment"],
                    "suggestion": result_payload["suggestion"],
                },
            }
            user_history = portfolio_history.get(username, [])
            user_history.append(entry)
            portfolio_history[username] = user_history
            save_portfolio_history(portfolio_history)
    except Exception as e:
        # Do not break prediction flow if history logging fails
        print(f"⚠️ Failed to log portfolio entry: {e}")


    return jsonify(result_payload)
































# ===========================================================
# ✅ Visuals Endpoint (UNCHANGED)
# ===========================================================
@app.route("/api/visuals", methods=["GET"])
def api_visuals():
    df_local = df.copy()
    default_by_edu = df_local.groupby("Education")["Default"].mean().reset_index().to_dict(orient="records")
    loan_vals = df_local["LoanAmount"].dropna().astype(float).tolist()
    corr = df_local.select_dtypes(include=[np.number]).corr()["Default"].sort_values(ascending=False).head(10).to_dict()
    return jsonify({
        "default_by_education": default_by_edu,
        "loan_amount_values": loan_vals,
        "correlations": corr
    })
 


@app.route("/api/occupation_approval", methods=["GET"])
def api_occupation_approval():
    """
    Occupation-wise approval rates.
    NOTE:
    - In the dataset, `Default` = 1 means the loan defaulted (i.e., NOT approved in the long run).
    - Approval rate is therefore computed as 1 - default_rate, as requested.
    - We treat `EmploymentType` as the occupation dimension for this visualization.
    """
    df_local = df.copy()


    # Basic safety check to avoid breaking if schema changes
    if "EmploymentType" not in df_local.columns or "Default" not in df_local.columns:
        return jsonify({"error": "Required columns not found"}), 500


    # Ensure Default is numeric (0/1); coerce errors to NaN then drop
    df_local["Default"] = pd.to_numeric(df_local["Default"], errors="coerce")
    df_local = df_local.dropna(subset=["Default"])


    grouped = (
        df_local.groupby("EmploymentType")["Default"]
        .mean()
        .reset_index()
        .rename(columns={"Default": "default_rate"})
    )
    grouped["approval_rate"] = 1.0 - grouped["default_rate"]


    return jsonify({
        "occupation_approval": grouped[["EmploymentType", "approval_rate"]].to_dict(orient="records")
    })




@app.route("/api/portfolio", methods=["GET"])
def api_portfolio():
    """
    Return the logged prediction history for the currently logged-in user.
    Data is stored server-side and only exposed through this endpoint.
    """
    username = session.get("user")
    if not username:
        return jsonify({"entries": []})
    entries = portfolio_history.get(username, [])
    # Return in reverse chronological order (latest first) for display
    return jsonify({"entries": list(reversed(entries))})


















# ===========================================================
# ✅ Bank Suggestion Endpoint (USING XGBOOST MODEL)
# ===========================================================
@app.route('/api/recommend_bank', methods=['POST'])
def recommend_bank():
    if bank_df is None:
        return jsonify({"error": "Bank data not found"}), 500
   
    if not xgb_bank_model or not bank_label_encoders or not xgb_feature_columns:
        return jsonify({"error": "XGBoost bank model not found. Please train the model first."}), 500
   
    data = request.get_json() or {}
   
    try:
        credit_score = float(data.get('CreditScore', 0))
        income = float(data.get('Income', 0))
        age = float(data.get('Age', 0))
        loan_amount = float(data.get('LoanAmount', 0))
        education = str(data.get('Education', '')).strip()
        employment = str(data.get('EmploymentType', '')).strip()
       
        # Encode categorical features using the trained encoders
        le_edu = bank_label_encoders.get('Education')
        le_emp = bank_label_encoders.get('EmploymentType')
       
        if le_edu is None or le_emp is None:
            return jsonify({"error": "Bank label encoders not properly loaded"}), 500
       
        # Handle unknown categories - try to find closest match
        try:
            education_enc = le_edu.transform([education])[0]
        except (ValueError, KeyError):
            # If education not found, use the first available
            if len(le_edu.classes_) > 0:
                education_enc = 0
            else:
                return jsonify({"error": "Education encoder has no classes"}), 500
       
        try:
            employment_enc = le_emp.transform([employment])[0]
        except (ValueError, KeyError):
            # If employment not found, use the first available
            if len(le_emp.classes_) > 0:
                employment_enc = 0
            else:
                return jsonify({"error": "Employment encoder has no classes"}), 500
       
        # Create feature engineering (same as training)
        # Calculate engineered features
        credit_income_ratio = credit_score / (income / 1000 + 1)
        loan_income_ratio = loan_amount / (income + 1)
        loan_age_ratio = loan_amount / (age + 1)
        credit_age_score = (credit_score * age) / 100
        affordability_score = (income * 12) / (loan_amount + 1)
        risk_score = (loan_amount / income) / (credit_score / 100 + 1)
        income_per_month = income / 12
        loan_to_annual_income = loan_amount / (income + 1)
       
        # Prepare features in the exact order as training
        # Feature order: ['CreditScore', 'Income', 'Age', 'LoanAmount',
        #                 'Education_enc', 'EmploymentType_enc',
        #                 'Credit_Income_Ratio', 'Loan_Income_Ratio', 'Loan_Age_Ratio',
        #                 'Credit_Age_Score', 'Affordability_Score', 'Risk_Score',
        #                 'Income_Per_Month', 'Loan_To_Annual_Income']
        X_user = np.array([[
            credit_score,
            income,
            age,
            loan_amount,
            education_enc,
            employment_enc,
            credit_income_ratio,
            loan_income_ratio,
            loan_age_ratio,
            credit_age_score,
            affordability_score,
            risk_score,
            income_per_month,
            loan_to_annual_income
        ]])
       
        # Predict using XGBoost model (returns probabilities)
        predicted_probs = xgb_bank_model.predict_proba(X_user)[0]
        predicted_label = xgb_bank_model.predict(X_user)[0]
       
        # Get top 3 predictions with probabilities
        top_3_indices = np.argsort(predicted_probs)[-3:][::-1]
       
        # Decode the predicted bank name
        le_bank = bank_label_encoders.get('BankName')
        if le_bank is None:
            return jsonify({"error": "Bank name encoder not found"}), 500
       
        predicted_bank_name = le_bank.inverse_transform([predicted_label])[0]
       
        # If model predicts "No_Match", try to find alternative banks
        if predicted_bank_name == 'No_Match':
            # Try to get the next best prediction
            for idx in top_3_indices:
                alt_bank = le_bank.inverse_transform([idx])[0]
                if alt_bank != 'No_Match':
                    predicted_bank_name = alt_bank
                    break
           
            if predicted_bank_name == 'No_Match':
                return jsonify({
                    "message": "No matching banks found based on your profile. Try adjusting your criteria.",
                    "banks": []
                }), 404
       
        # Get the predicted bank details from bank_df
        # Filter by education and employment as well for better matching
        recommended = bank_df[
            (bank_df['Bank_Name'] == predicted_bank_name) &
            (bank_df['Education_Required'] == education) &
            (bank_df['Employment_Type'] == employment)
        ].copy()
       
        # If no exact match, try without education/employment filter
        if recommended.empty:
            recommended = bank_df[bank_df['Bank_Name'] == predicted_bank_name].copy()
       
        # If still empty, use fallback matching
        if recommended.empty:
            recommended = bank_df.copy()
            if 'Min_Credit_Score' in bank_df.columns:
                recommended = recommended[recommended['Min_Credit_Score'] <= credit_score]
            if 'Min_Income' in bank_df.columns:
                recommended = recommended[recommended['Min_Income'] <= income]
            if 'Min_Age' in bank_df.columns:
                recommended = recommended[recommended['Min_Age'] <= age]
            if 'Max_Age' in bank_df.columns:
                recommended = recommended[recommended['Max_Age'] >= age]
           
            if recommended.empty:
                return jsonify({
                    "message": "No matching banks found. Try relaxing the criteria.",
                    "banks": []
                }), 404
       
        # Sort by best rate/fee if multiple matches
        if len(recommended) > 1:
            if 'Interest_Rate' in recommended.columns:
                recommended = recommended.sort_values(by='Interest_Rate', ascending=True)
            elif 'Processing_Fee' in recommended.columns:
                recommended = recommended.sort_values(by='Processing_Fee', ascending=True)
       
        # Return top 5 matches
        return jsonify(recommended.head(5).to_dict(orient='records'))
















    except Exception as e:
        print(f"❌ Bank recommendation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during bank recommendation: {str(e)}"}), 500
































# ===========================================================
# ✅ Clustering Endpoint (UNCHANGED)
# ===========================================================
@app.route("/api/clusters", methods=["GET"])
def api_clusters():
    # ... (No changes needed, this loads the viz data generated by clustering.py)
    try:
        # Check for new clustering system
        if kmeans_clustering and cluster_preprocessor and cluster_risk_labels and cluster_colors_map:
            print("📊 Using new risk-based clustering system...")
           
            # Load cluster visualization data
            cluster_data_path = os.path.join(BASE, "data", "cluster_data.csv")
            if os.path.exists(cluster_data_path):
                df_viz = pd.read_csv(cluster_data_path)
            else:
                return jsonify({"error": "Cluster data not found. Run clustering.py first."}), 500
           
            # Format data for frontend
            formatted_clusters = []
            for cluster_id in sorted(df_viz['Cluster'].unique()):
                cluster_data = df_viz[df_viz['Cluster'] == cluster_id]
                risk_label = cluster_data.iloc[0]['Risk']
                color = cluster_data.iloc[0]['Color']
               
                # Sample points for visualization
                sample_size = min(500, len(cluster_data))
                cluster_sample = cluster_data.sample(n=sample_size, random_state=42) if len(cluster_data) > 500 else cluster_data
               
                points = []
                for _, row in cluster_sample.iterrows():
                    points.append({
                        'income': float(row['Income']),
                        'credit_score': float(row['CreditScore']),
                        'default_prob': float(row['Default'])
                    })
               
                formatted_clusters.append({
                    'cluster_id': int(cluster_id),
                    'risk_label': risk_label,
                    'points': points,
                    'size': len(cluster_data),
                    'color': color,
                    'avg_income': float(cluster_data['Income'].mean()),
                    'avg_credit': float(cluster_data['CreditScore'].mean()),
                    'default_rate': float(cluster_data['Default'].mean()),
                    # NEW: average loan amount per cluster (for visualization only)
                    'avg_loan_amount': float(cluster_data['LoanAmount'].mean()) if 'LoanAmount' in cluster_data.columns else None
                })
           
            print(f"✅ Loaded {len(formatted_clusters)} clusters")
            return jsonify({'clusters': formatted_clusters})
       
        # Fallback to old clustering system
        elif kmeans_clustering:
             return jsonify({"error": "No cluster visualization data found. Run clustering.py first."}), 500
       
        else:
            return jsonify({"error": "No clustering model found. Run clustering.py first."}), 500
   
    except Exception as e:
        print(f"❌ Cluster error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Clustering failed: {str(e)}"}), 500
































# ===========================================================
# ✅ User Cluster Assignment Endpoint (UNCHANGED)
# ===========================================================
@app.route("/api/test_cluster", methods=["POST"]) # Changed route name to match frontend
def api_test_cluster():
    """Assign a user to a risk cluster based on their details, including RiskScore calculation."""
    try:
        if not kmeans_clustering or not cluster_preprocessor or not cluster_risk_labels or not risk_scaler_params:
            return jsonify({"error": "Clustering models or metadata not found. Run clustering.py first."}), 500
       
        data = request.get_json() or {}
       
        # 1. Extract and Validate Input
        income = float(data.get("Income", 0))
        credit_score = float(data.get("CreditScore", 0))
        age = float(data.get("Age", 0))
        loan_amount = float(data.get("LoanAmount", 0))
        education = str(data.get("Education", "")).strip() # Apply strip for consistency
        marital_status = str(data.get("MaritalStatus", "")).strip() # Apply strip for consistency
        employment_type = str(data.get("EmploymentType", "")).strip() # Apply strip for consistency
       
        # Features required for clustering preprocessor (order must match clustering.py: numeric + categorical)
        numeric_features = ['Income', 'CreditScore', 'Age', 'LoanAmount']
        categorical_features = ['Education', 'MaritalStatus', 'EmploymentType']
       
        user_data = pd.DataFrame([{
            'Income': income,
            'CreditScore': credit_score,
            'Age': age,
            'LoanAmount': loan_amount,
            'Education': education,
            'MaritalStatus': marital_status,
            'EmploymentType': employment_type
        }])
       
        # 2. Calculate RiskScore (Must match clustering.py logic)
        mins = risk_scaler_params['min']
        maxs = risk_scaler_params['max']
        calc_features = risk_scaler_params['features'] # ['Income', 'CreditScore', 'LoanAmount']
       
        # Apply min-max normalization using TRAINED min/max values
        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0
       
        income_norm = normalize(income, mins[calc_features.index('Income')], maxs[calc_features.index('Income')])
        credit_norm = normalize(credit_score, mins[calc_features.index('CreditScore')], maxs[calc_features.index('CreditScore')])
        loan_norm = normalize(loan_amount, mins[calc_features.index('LoanAmount')], maxs[calc_features.index('LoanAmount')])
















        # RiskScore formula (Default=0 for new prediction)
        risk_score = (
            (1 - credit_norm) * 3.0 +
            loan_norm * 2.0 +
            (1 - income_norm) * 2.0 +
            0.0 # Default history (weight 5.0) is 0 for new input
        ) / 7.0 # New total weight is 3+2+2 = 7.0 (since 5.0 is excluded)
       
        risk_score = max(0, min(risk_score, 1)) # Ensure 0-1 range
















        # 3. Preprocess and Combine
        X_user_processed = cluster_preprocessor.transform(user_data[numeric_features + categorical_features])
       
        # Combine the processed features with the custom RiskScore
        X_user_combined = np.hstack([X_user_processed, np.array([[risk_score]])])
















        # 4. Predict cluster
        user_cluster = int(kmeans_clustering.predict(X_user_combined)[0])
        risk_label = cluster_risk_labels.get(user_cluster, "Unknown")
        color = cluster_colors_map.get(user_cluster, "#6c757d")
       
        # Recommendation logic based on risk label
        recommendation = "Review eligibility based on profile."
        if "Low" in risk_label:
            recommendation = "High likelihood of approval. Offer competitive rates."
        elif "High" in risk_label:
            recommendation = "Proceed with caution. Requires management review or collateral."
       
        return jsonify({
            "cluster": user_cluster,
            "risk_label": risk_label,
            "color": color,
            "recommendation": recommendation,
            "message": f"Customer assigned to Cluster {user_cluster} with {risk_label}"
        })
   
    except Exception as e:
        print(f"❌ Cluster assignment error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
































# ===========================================================
# ✅ Chatbot Endpoint (Gemini API Integration)
# ===========================================================
@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    """Handle chatbot requests using Gemini API"""
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
       
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
       
        # Use the API key (with safe fallback)
        try:
            # Try to get from global variables first
            api_key = globals().get('GEMINI_API_KEY', "AIzaSyCkQLhMPMTdF8dmmFfOUceFT8mVhahl7ls")
            model_name = globals().get('GEMINI_MODEL', "gemini-2.5-flash")
        except:
            # If that fails, use hardcoded defaults
            api_key = "AIzaSyCkQLhMPMTdF8dmmFfOUceFT8mVhahl7ls"
            model_name = "gemini-2.5-flash"
       
        # Ensure we have values
        if not api_key:
            api_key = "AIzaSyCkQLhMPMTdF8dmmFfOUceFT8mVhahl7ls"
        if not model_name:
            model_name = "gemini-2.5-flash"
       
        if not api_key:
            return jsonify({"error": "Gemini API key not configured"}), 500
       
        print(f"🔑 Using Gemini API Key: {api_key[:10]}...")  # Debug: show first 10 chars
        print(f"🤖 Using Gemini Model: {model_name}")  # Debug: show model name
       
        # Prepare the request to Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
       
        # System prompt to make the chatbot focused on loan and financial topics
        system_prompt = """You are a helpful AI assistant for a loan default prediction system called LoanLens.
You help users with questions about:
- Loan applications and eligibility
- Credit scores and financial health
- Loan default predictions and risk assessment
- Financial advice related to loans
- Understanding loan terms and conditions
- Bank recommendations and loan options




Be friendly, professional, and provide accurate information. If asked about something outside your expertise, politely redirect to loan-related topics."""
       
        # Correct Gemini API payload structure
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
       
        headers = {
            "Content-Type": "application/json"
        }
       
        print(f"📤 Sending request to Gemini API: {url[:80]}...")  # Debug
        response = requests.post(url, json=payload, headers=headers, timeout=30)
       
        print(f"📥 Response status: {response.status_code}")  # Debug
       
        if response.status_code == 200:
            result = response.json()
            print(f"📥 Response keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")  # Debug
           
            # Check for errors in response
            if "error" in result:
                error_msg = result["error"].get("message", "Unknown error")
                print(f"❌ Gemini API returned error: {error_msg}")
                return jsonify({"error": f"Gemini API error: {error_msg}"}), 500
           
            # Extract response text
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
               
                # Check if content was blocked
                if "finishReason" in candidate and candidate["finishReason"] != "STOP":
                    finish_reason = candidate.get("finishReason", "UNKNOWN")
                    print(f"⚠️ Content blocked or incomplete. Finish reason: {finish_reason}")
                    return jsonify({"error": f"Response blocked: {finish_reason}"}), 500
               
                if "content" in candidate and "parts" in candidate["content"]:
                    if len(candidate["content"]["parts"]) > 0:
                        bot_response = candidate["content"]["parts"][0].get("text", "Sorry, I couldn't generate a response.")
                        print(f"✅ Successfully got response: {bot_response[:50]}...")  # Debug
                        return jsonify({"response": bot_response})
           
            print(f"❌ Invalid response structure: {result}")  # Debug
            return jsonify({"error": "Invalid response from Gemini API"}), 500
        else:
            error_msg = response.text
            print(f"❌ Gemini API error: {response.status_code} - {error_msg}")
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_msg = error_json["error"].get("message", error_msg)
            except:
                pass
            return jsonify({"error": f"API request failed ({response.status_code}): {error_msg}"}), response.status_code
           
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return jsonify({"error": "Request timeout. Please try again."}), 504
    except requests.exceptions.RequestException as e:
        print(f"❌ Chatbot request error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Network error: {str(e)}"}), 500
    except Exception as e:
        print(f"❌ Chatbot error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500








# ===========================================================
# ✅ Repayment Timeline & Strategy Endpoint (UNCHANGED)
# ===========================================================
@app.route("/api/repayment", methods=["POST"])
def api_repayment():
    # ... (Unchanged logic)
    data = request.get_json() or {}
    income = float(data.get("Income", 0))
    loan_amount = float(data.get("LoanAmount", 0))
    interest_rate = float(data.get("InterestRate", 0)) / 100 / 12
    risk = float(data.get("probability", 0))
















    if income <= 0 or loan_amount <= 0:
        return jsonify({"error": "Invalid income or loan amount."}), 400
















    monthly_capacity = max(1, income * 0.20) / 12
    est_months = int(max(6, min(loan_amount / monthly_capacity, 360)))
















    if interest_rate > 0:
        denom = 1 - (1 + interest_rate) ** (-est_months)
        monthly_payment = loan_amount * (interest_rate / denom)
    else:
        monthly_payment = loan_amount / est_months
















    schedule, outstanding = [], loan_amount
    for m in range(1, est_months + 1):
        interest = outstanding * interest_rate
        principal = monthly_payment - interest
        outstanding = max(0, outstanding - principal)
        schedule.append({
            "month": m,
            "payment": round(monthly_payment, 2),
            "outstanding": round(outstanding, 2)
        })
















    strategy = (
        "Low risk — maintain regular EMI payments."
        if risk < 0.05 else
        "Medium risk — opt for shorter loan tenure and avoid new credit."
        if risk < 0.1 else
        "High risk — restructure loan or seek financial counselling."
    )
















    return jsonify({
        "strategy": strategy,
        "timeline": schedule[:12],  # first 12 months
        "monthly_payment": round(monthly_payment, 2),
        "total_months": est_months
    })
































# ===========================================================
# ✅ Run Server
# ===========================================================
if __name__ == "__main__":
    # Ensure all models are loaded or the app will crash later
    if not rf_model or not scaler or not kmeans_clustering or not cluster_preprocessor:
        print("CRITICAL: Some models failed to load. Please run all training scripts.")
    app.run(debug=True)
