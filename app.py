import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv


# ===========================================================
# ✅ Environment Setup
# ===========================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


print("🔑 Loaded GEMINI_API_KEY:", "✅ Found" if GEMINI_API_KEY else "❌ Missing")


app = Flask(__name__)
app.secret_key = "replace_this_with_a_real_secret"


BASE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================
# ✅ Data and Model Loading
# ===========================================================
"""
DATA_PATH = os.path.join(BASE, "data", "loan_data.csv")
BANK_DATA_PATH = os.path.join(BASE, "data", "bank_data.csv")
MODELS_DIR = os.path.join(BASE, "models")
USERS_FILE = os.path.join(BASE, "users.json")
"""

BASE = os.path.dirname(os.path.abspath(__file__))

# If your models are directly in root
MODELS_DIR = os.path.join(BASE, "models")

DATA_PATH = os.path.join(BASE, "data", "loan_data.csv")
BANK_DATA_PATH = os.path.join(BASE, "data", "bank_data.csv")
USERS_FILE = os.path.join(BASE, "users.json")

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
    df[c] = df[c].astype(str)
    df[c + "_enc"] = le.fit_transform(df[c])
    label_encoders[c] = le
    category_options[c] = list(le.classes_.astype(str))


def try_load(name):
    """Try loading a model file with different extensions"""
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# Load models with correct filenames
rf_model = try_load("random_forest_model.joblib")
scaler = try_load("scaler.joblib")
kmeans_clustering = try_load("kmeans_model.joblib")  # or "kmeans_clustering.joblib" - check which one you want
label_encoders_loaded = try_load("encoders.joblib")

print("🔹 RF Model:", "✅ Loaded" if rf_model else "❌ Not found")
print("🔹 Scaler:", "✅ Loaded" if scaler else "❌ Not found")
print("🔹 KMeans:", "✅ Loaded" if kmeans_clustering else "❌ Not found")
print("🔹 Encoders:", "✅ Loaded" if label_encoders_loaded else "❌ Not found")

# Load clustering models
kmeans_clustering = try_load("kmeans_clustering.joblib")
cluster_preprocessor = try_load("cluster_preprocessor.joblib")
cluster_risk_labels = try_load("cluster_risk_labels.joblib")
cluster_colors_map = try_load("cluster_colors.joblib")

# Fallback to old kmeans if new one doesn't exist
kmeans = kmeans_clustering or try_load("kmeans_model.joblib")
label_encoders_loaded = try_load("label_encoders.joblib")

# Load bank data if it exists
bank_df = None
if os.path.exists(BANK_DATA_PATH):
    try:
        bank_df = pd.read_csv(BANK_DATA_PATH, engine='python', encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Loaded {len(bank_df)} rows from bank_data.csv")
    except Exception as e:
        print(f"⚠️ Could not load bank_data.csv: {e}")
        bank_df = None


# ===========================================================
# ✅ User Management
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
# ✅ Routes
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
    return render_template("dashboard.html", categories=category_options, features=ALL_FEATURES)


# ===========================================================
# ✅ Helper Functions
# ===========================================================
def encode_input(data):
    row = []
    for col in NUM_COLS:
        val = data.get(col)
        row.append(float(val) if val not in [None, ""] else np.nan)
    for col in CAT_COLS:
        v = str(data.get(col, ""))
        le = label_encoders.get(col)
        try:
            row.append(int(le.transform([v])[0]))
        except Exception:
            row.append(0)
    return np.array([row])


# ===========================================================
# ✅ Prediction Endpoint
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

    suggestion = (
        "Low risk — standard repayment"
        if prob < 0.4
        else ("Medium risk — consider consolidation"
              if prob < 0.7
              else "High risk — consider refinancing or restructuring")
    )

    return jsonify({
        "probability": round(prob, 4),
        "estimated_months": est_months,
        "monthly_payment": round(monthly_payment, 2),
        "schedule": schedule,
        "suggestion": suggestion
    })


# ===========================================================
# ✅ Visuals Endpoint
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


# ===========================================================
# ✅ Bank Suggestion Endpoint
# ===========================================================
@app.route("/api/bank_suggestion", methods=["POST"])
def api_bank_suggestion():
    if not xgb_bank_model:
        return jsonify({"error": "Bank model not found"}), 500
    
    if bank_df is None:
        return jsonify({"error": "Bank data not found"}), 500

    data = request.get_json() or {}
    try:
        credit = float(data.get("CreditScore", 0))
        income = float(data.get("Income", 0))
        age = float(data.get("Age", 0))
        loan_amt = float(data.get("LoanAmount", 0))
        education = data.get("Education", "12th Pass")
        emp_type = data.get("EmploymentType", "Salaried")

        # Encode to match model
        edu_code = 0
        emp_code = 0
        
        if 'Education_Required' in bank_df.columns:
            edu_cats = bank_df['Education_Required'].astype('category').cat.categories
            edu_code = edu_cats.get_loc(education) if education in edu_cats else 0
        
        if 'Employment_Type' in bank_df.columns:
            emp_cats = bank_df['Employment_Type'].astype('category').cat.categories
            emp_code = emp_cats.get_loc(emp_type) if emp_type in emp_cats else 0

        input_data = np.array([[credit, loan_amt, income, age, age, edu_code, emp_code]])
        best_bank = xgb_bank_model.predict(input_data)[0]

        match = bank_df[bank_df["Bank_Name"] == best_bank].iloc[0].to_dict()

        return jsonify({
            "best_bank": best_bank,
            "interest_rate": match.get("Interest_Rate", "N/A"),
            "processing_fee": match.get("Processing_Fee", "N/A"),
            "loan_type": match.get("Loan_Type", "N/A")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/recommend_bank', methods=['POST'])
def recommend_bank():
    if bank_df is None:
        return jsonify({"error": "Bank data not found"}), 500
    
    data = request.get_json() or {}
    credit_score = float(data.get('CreditScore', 0))
    income = float(data.get('Income', 0))
    age = float(data.get('Age', 0))
    education = data.get('Education', '')
    employment = data.get('EmploymentType', '')

    # Simple filter based on conditions
    recommended = bank_df.copy()
    
    if 'Min_Credit_Score' in bank_df.columns:
        recommended = recommended[recommended['Min_Credit_Score'] <= credit_score]
    if 'Min_Income' in bank_df.columns:
        recommended = recommended[recommended['Min_Income'] <= income]
    if 'Min_Age' in bank_df.columns:
        recommended = recommended[recommended['Min_Age'] <= age]
    if 'Max_Age' in bank_df.columns:
        recommended = recommended[recommended['Max_Age'] >= age]
    if 'Education_Required' in bank_df.columns and education:
        recommended = recommended[recommended['Education_Required'] == education]
    if 'Employment_Type' in bank_df.columns and employment:
        recommended = recommended[recommended['Employment_Type'] == employment]

    if recommended.empty:
        return jsonify({"message": "No matching banks found."}), 404

    return jsonify(recommended.to_dict(orient='records'))


# ===========================================================
# ✅ Clustering Endpoint
# ===========================================================
@app.route("/api/clusters", methods=["GET"])
def api_clusters():
    try:
        # Check for new clustering system
        if kmeans_clustering and cluster_preprocessor and cluster_risk_labels and cluster_colors_map:
            print("📊 Using new risk-based clustering system...")
            
            # Load cluster visualization data
            cluster_data_path = os.path.join(BASE, "data", "cluster_data.csv")
            if os.path.exists(cluster_data_path):
                df_viz = pd.read_csv(cluster_data_path)
            else:
                return jsonify({"error": "Cluster data not found. Run train_clustering.py first."}), 500
            
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
                    'default_rate': float(cluster_data['Default'].mean())
                })
            
            print(f"✅ Loaded {len(formatted_clusters)} clusters")
            return jsonify({'clusters': formatted_clusters})
        
        # Fallback to old clustering system
        elif kmeans:
            print("⚠️ Using legacy clustering system...")
            
            if 'Income' not in df.columns or 'Default' not in df.columns:
                return jsonify({"error": "Required columns not found"}), 500
            
            required_cols = NUM_COLS
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify({"error": f"Missing columns: {missing_cols}"}), 500
            
            df_cluster = df[required_cols + ['Default']].copy()
            for col in required_cols + ['Default']:
                df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
            df_cluster = df_cluster.dropna()
            
            if len(df_cluster) == 0:
                return jsonify({"error": "No valid data"}), 500
            
            X_cluster = df_cluster[required_cols].values
            clusters = kmeans.predict(X_cluster)
            df_cluster['Cluster'] = clusters
            
            cluster_colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6c757d']
            unique_clusters = sorted(df_cluster['Cluster'].unique())
            
            formatted_clusters = []
            for cluster_id in unique_clusters:
                cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
                sample_size = min(1000, len(cluster_data))
                cluster_sample = cluster_data.sample(n=sample_size, random_state=42) if len(cluster_data) > 1000 else cluster_data
                
                points = []
                for _, row in cluster_sample.iterrows():
                    points.append({
                        'income': float(row['Income']),
                        'default_prob': float(row['Default'])
                    })
                
                formatted_clusters.append({
                    'points': points,
                    'size': len(cluster_data),
                    'color': cluster_colors[int(cluster_id) % len(cluster_colors)]
                })
            
            return jsonify({'clusters': formatted_clusters})
        
        else:
            return jsonify({"error": "No clustering model found. Run train_clustering.py first."}), 500
    
    except Exception as e:
        print(f"❌ Cluster error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Clustering failed: {str(e)}"}), 500


# ===========================================================
# ✅ User Cluster Assignment Endpoint
# ===========================================================
@app.route("/api/assign_cluster", methods=["POST"])
def api_assign_cluster():
    """Assign a user to a risk cluster based on their details."""
    try:
        if not kmeans_clustering or not cluster_preprocessor or not cluster_risk_labels:
            return jsonify({"error": "Clustering models not found. Run train_clustering.py first."}), 500
        
        data = request.get_json() or {}
        
        # Extract required features (matching training script)
        income = float(data.get("Income", 0))
        credit_score = float(data.get("CreditScore", 0))
        age = float(data.get("Age", 0))
        loan_amount = float(data.get("LoanAmount", 0))
        education = str(data.get("Education", ""))
        marital_status = str(data.get("MaritalStatus", ""))
        employment_type = str(data.get("EmploymentType", ""))
        
        if income <= 0 or credit_score <= 0:
            return jsonify({"error": "Invalid income or credit score"}), 400
        
        # Preprocess user input (order must match training)
        user_data = pd.DataFrame([{
            'Income': income,
            'CreditScore': credit_score,
            'Age': age,
            'LoanAmount': loan_amount,
            'Education': education,
            'MaritalStatus': marital_status,
            'EmploymentType': employment_type
        }])
        
        X_user = cluster_preprocessor.transform(user_data)
        
        # Predict cluster
        user_cluster = int(kmeans_clustering.predict(X_user)[0])
        risk_label = cluster_risk_labels.get(user_cluster, "Unknown")
        color = cluster_colors_map.get(user_cluster, "#6c757d")
        
        return jsonify({
            "cluster": user_cluster,
            "risk_label": risk_label,
            "color": color,
            "message": f"Customer assigned to Cluster {user_cluster} with {risk_label}"
        })
    
    except Exception as e:
        print(f"❌ Cluster assignment error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ===========================================================
# ✅ Repayment Timeline & Strategy Endpoint
# ===========================================================
@app.route("/api/repayment", methods=["POST"])
def api_repayment():
    """Generate repayment strategy and timeline."""
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
        if risk < 0.4 else
        "Medium risk — opt for shorter loan tenure and avoid new credit."
        if risk < 0.7 else
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
    app.run(debug=True)