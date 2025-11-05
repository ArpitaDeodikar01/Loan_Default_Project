# app.py (corrected & robust wiring for RF, KMeans, XGBoost)
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
# ✅ Data and Model Loading (paths)
# ===========================================================
DATA_PATH = os.path.join(BASE, "data", "loan_data.csv")
BANK_DATA_PATH = os.path.join(BASE, "data", "bank_data.csv")
MODELS_DIR = os.path.join(BASE, "models")
USERS_FILE = os.path.join(BASE, "users.json")

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Data file not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ===========================================================
# ✅ Column definitions (used for encoding & prediction)
# Keep these consistent with training
# ===========================================================
CAT_COLS = ['Education', 'EmploymentType', 'MaritalStatus',
            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
NUM_COLS = ['Age', 'Income', 'LoanAmount', 'CreditScore',
            'MonthsEmployed', 'NumCreditLines', 'InterestRate',
            'LoanTerm', 'DTIRatio']

ALL_FEATURES = NUM_COLS + CAT_COLS

# Build label encoders from the loan_data.csv (so encodings match training data)
label_encoders = {}
category_options = {}
for c in CAT_COLS:
    le = LabelEncoder()
    # ensure strings (avoid NaN issues)
    df[c] = df[c].astype(str)
    df[c + "_enc"] = le.fit_transform(df[c])
    label_encoders[c] = le
    # frontend wants category lists
    category_options[c] = list(le.classes_.astype(str))

# ===========================================================
# Helper: robust model loader (accepts several naming variants)
# ===========================================================
def try_load(name):
    path = os.path.join(MODELS_DIR, name)
    return joblib.load(path) if os.path.exists(path) else None

def try_load_any(names):
    for n in names:
        obj = try_load(n)
        if obj is not None:
            return obj, n
    return None, None

# ===========================================================
# ✅ Load models (robust names)
# - Random Forest for loan default
# - Scaler for numeric features
# - KMeans for segmentation
# - XGBoost for bank recommendation
# - Optional label encoders saved separately for bank model
# ===========================================================
rf_model, rf_loaded_name = try_load_any([
    "random_forest_model.joblib", "random_forest.joblib", "random_forest_model.pkl", "random_forest.pkl"
])

scaler, scaler_loaded_name = try_load_any([
    "scaler.joblib", "scaler.pkl", "standard_scaler.joblib"
])

kmeans_model, kmeans_loaded_name = try_load_any([
    "kmeans_model.joblib", "kmeans_model.pkl"
])

xgb_bank_model, xgb_loaded_name = try_load_any([
    "xgboost_bank_model.joblib", "xgboost_bank_model.pkl", "bank_xgboost.joblib"
])

bank_label_encoder = try_load("bank_label_encoder.joblib")  # optional encoder for bank labels

print(f"models loaded -> rf: {rf_loaded_name}, scaler: {scaler_loaded_name}, kmeans: {kmeans_loaded_name}, xgb: {xgb_loaded_name}")

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
# - encode_input: create a row aligned with training features
# - safe_transform: uses scaler if available
# ===========================================================
def encode_input(data):
    """
    Accepts dict-like data (from JSON or form)
    Returns numpy array shaped (1, n_features) aligned to NUM_COLS + CAT_COLS
    """
    row = []
    # numeric features
    for col in NUM_COLS:
        val = data.get(col)
        # Accept alternative keys (frontend might send different casing)
        if val is None:
            # try alternative keys
            val = data.get(col.lower()) or data.get(col.upper()) or data.get(col.replace(' ', ''))
        try:
            row.append(float(val) if val not in [None, ""] else np.nan)
        except Exception:
            row.append(np.nan)
    # categorical features -> encode using label_encoders built from loan_data.csv
    for col in CAT_COLS:
        v = data.get(col)
        if v is None:
            v = data.get(col.lower()) or data.get(col.upper()) or ""
        v = str(v)
        le = label_encoders.get(col)
        if le is not None:
            try:
                enc = int(le.transform([v])[0])
            except Exception:
                # if unseen category, try to pick an index of an existing class (0)
                enc = 0
        else:
            enc = 0
        row.append(enc)
    return np.array([row])

def scale_row(arr):
    """Scale numeric part (all features) using loaded scaler if present; otherwise returns arr."""
    if scaler is not None:
        try:
            return scaler.transform(arr)
        except Exception:
            # scaler may expect a precise number of features; fail gracefully
            return arr
    return arr

# ===========================================================
# ✅ Prediction Endpoint (Loan Default) - uses Random Forest
# ===========================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    # Ensure required model pieces exist
    if rf_model is None or scaler is None:
        return jsonify({"error": "Random Forest model or scaler not found. Please run training first."}), 500

    payload = request.get_json() or request.form.to_dict()
    X_raw = encode_input(payload)
    X_scaled = scale_row(X_raw)

    # Random forest predict_proba -> index 1 is probability of default=1
    try:
        proba_arr = rf_model.predict_proba(X_scaled)[0]
        # if binary, proba_arr length should be 2
        prob = float(proba_arr[1]) if len(proba_arr) > 1 else float(proba_arr[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # replicate your loan repayment calcs (keep logic unchanged)
    try:
        loan_amount = float(payload.get("LoanAmount", 0))
    except Exception:
        loan_amount = 0.0
    try:
        income = float(payload.get("Income", 1))
    except Exception:
        income = 1.0

    monthly_capacity = max(1, income * 0.20) / 12
    est_months = int(max(6, min(loan_amount / monthly_capacity, 360))) if monthly_capacity > 0 else 360

    monthly_rate = float(payload.get("InterestRate", 0)) / 100 / 12
    if monthly_rate > 0:
        denom = 1 - (1 + monthly_rate) ** (-est_months)
        monthly_payment = loan_amount * (monthly_rate / denom) if denom != 0 else loan_amount / est_months
    else:
        monthly_payment = loan_amount / est_months if est_months != 0 else loan_amount

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

    # Optionally predict cluster for this input if kmeans is present and expects same sized feature vector
    cluster_id = None
    if kmeans_model is not None:
        try:
            # If kmeans was trained on scaled numeric subset, we try to reuse X_scaled or transform to appropriate dims.
            # Many kmeans pipelines trained on PCA or scaled numeric features; we'll attempt to use X_scaled directly.
            cluster_id = int(kmeans_model.predict(X_scaled)[0])
        except Exception:
            # last-resort: try raw features
            try:
                cluster_id = int(kmeans_model.predict(X_raw)[0])
            except Exception:
                cluster_id = None

    response = {
        "probability": round(prob, 4),
        "estimated_months": est_months,
        "monthly_payment": round(monthly_payment, 2),
        "schedule": schedule,
        "suggestion": suggestion
    }
    if cluster_id is not None:
        response["cluster"] = cluster_id

    return jsonify(response)

# ===========================================================
# ✅ Visuals Endpoint
# ===========================================================
@app.route("/api/visuals", methods=["GET"])
def api_visuals():
    df_local = df.copy()
    # ensure 'Default' column exists
    if 'Default' not in df_local.columns:
        df_local['Default'] = 0
    default_by_edu = df_local.groupby("Education")["Default"].mean().reset_index().to_dict(orient="records")
    loan_vals = df_local["LoanAmount"].dropna().astype(float).tolist() if "LoanAmount" in df_local.columns else []
    corr = {}
    try:
        corr = df_local.select_dtypes(include=[np.number]).corr()["Default"].sort_values(ascending=False).head(10).to_dict()
    except Exception:
        corr = {}
    return jsonify({
        "default_by_education": default_by_edu,
        "loan_amount_values": loan_vals,
        "correlations": corr
    })

# ===========================================================
# ✅ Load Additional Bank Data + Models (for bank recommendation)
# ===========================================================
# bank_df may be used in several endpoints
bank_df = None
if os.path.exists(BANK_DATA_PATH):
    try:
        bank_df = pd.read_csv(BANK_DATA_PATH)
    except Exception:
        bank_df = None

# ===========================================================
# ✅ Bank Suggestion Endpoint (XGBoost)
# ===========================================================
@app.route("/api/bank_suggestion", methods=["POST"])
def api_bank_suggestion():
    if xgb_bank_model is None:
        return jsonify({"error": "Bank model not found"}), 500

    data = request.get_json() or {}
    try:
        credit = float(data.get("CreditScore", 0))
        income = float(data.get("Income", 0))
        age = float(data.get("Age", 0))
        loan_amt = float(data.get("LoanAmount", 0))
        education = data.get("Education", "")
        emp_type = data.get("EmploymentType", "")

        # Build input array - the exact ordering must match model training.
        # We try a reasonable, forgiving ordering; if it fails you'll see a helpful error.
        # Common training order: [CreditScore, LoanAmount, Income, Age, ... , edu_enc, emp_enc]
        # If bank_label_encoder exists, use it to decode predicted label.
        input_features = [credit, loan_amt, income, age]

        # Attempt encode education/employment using bank_df categories if present
        edu_code = 0
        emp_code = 0
        if bank_df is not None:
            # guess column names used in bank_df
            if "Education_Required" in bank_df.columns:
                cats = list(bank_df["Education_Required"].astype('category').cat.categories)
                if education in cats:
                    edu_code = cats.index(education)
            if "Employment_Type" in bank_df.columns:
                cats = list(bank_df["Employment_Type"].astype('category').cat.categories)
                if emp_type in cats:
                    emp_code = cats.index(emp_type)

        input_features.extend([edu_code, emp_code])
        input_arr = np.array([input_features])

        predicted_label = int(xgb_bank_model.predict(input_arr)[0])

        # decode best bank name
        best_bank = None
        if bank_label_encoder is not None:
            try:
                best_bank = bank_label_encoder.inverse_transform([predicted_label])[0]
            except Exception:
                best_bank = str(predicted_label)
        else:
            # fallback: try to find matching row in bank_df if exists
            if bank_df is not None and "Bank_Label" in bank_df.columns:
                # find a bank with matching label
                match_rows = bank_df[bank_df["Bank_Label"] == predicted_label]
                if not match_rows.empty and "Bank_Name" in match_rows.columns:
                    best_bank = match_rows.iloc[0]["Bank_Name"]
            if best_bank is None:
                best_bank = str(predicted_label)

        # search bank_df for details
        match = None
        if bank_df is not None and "Bank_Name" in bank_df.columns:
            candidates = bank_df[bank_df["Bank_Name"] == best_bank]
            if not candidates.empty:
                match = candidates.iloc[0].to_dict()

        # Return guessed bank info if available
        return jsonify({
            "best_bank": best_bank,
            "interest_rate": match.get("Interest_Rate") if match else None,
            "processing_fee": match.get("Processing_Fee") if match else None,
            "loan_type": match.get("Loan_Type") if match else None
        })

    except Exception as e:
        return jsonify({"error": f"Bank suggestion error: {str(e)}"}), 400

# ===========================================================
# ✅ Simple bank recommendation fallback (filtering using bank_data.csv)
# ===========================================================
@app.route('/api/recommend_bank', methods=['POST'])
def recommend_bank():
    if bank_df is None:
        return jsonify({"message": "Bank dataset not available."}), 404

    data = request.get_json() or {}
    try:
        credit_score = float(data.get('CreditScore', 0))
        income = float(data.get('Income', 0))
        age = float(data.get('Age', 0))
        education = data.get('Education', None)
        employment = data.get('EmploymentType', None)

        # Ensure bank_df columns exist before filtering
        dfb = bank_df.copy()
        cond = pd.Series(True, index=dfb.index)
        if 'Min_Credit_Score' in dfb.columns:
            cond &= (dfb['Min_Credit_Score'] <= credit_score)
        if 'Min_Income' in dfb.columns:
            cond &= (dfb['Min_Income'] <= income)
        if 'Min_Age' in dfb.columns and 'Max_Age' in dfb.columns:
            cond &= (dfb['Min_Age'] <= age) & (dfb['Max_Age'] >= age)
        if education and 'Education_Required' in dfb.columns:
            cond &= (dfb['Education_Required'] == education)
        if employment and 'Employment_Type' in dfb.columns:
            cond &= (dfb['Employment_Type'] == employment)

        recommended = dfb[cond]

        if recommended.empty:
            return jsonify({"message": "No matching banks found."}), 404

        return jsonify(recommended.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

from sklearn.preprocessing import StandardScaler

@app.route("/api/clusters", methods=["GET"])
def api_clusters():
    # Ensure model exists
    if kmeans_model is None:
        return jsonify({"error": "KMeans model not found. Please train and save 'kmeans_model.joblib' in models/"}), 500

    # Try to load bank data, otherwise use loan_data (df)
    df_cluster = None
    if os.path.exists(BANK_DATA_PATH):
        try:
            df_cluster = pd.read_csv(BANK_DATA_PATH)
        except Exception:
            df_cluster = None
    if df_cluster is None:
        df_cluster = df.copy()

    # Preferred numeric features used for clustering (the 9 features)
    preferred_features = [
        ('Age', ['Age', 'age']),
        ('Income', ['Income', 'income', 'AnnualIncome']),
        ('LoanAmount', ['LoanAmount', 'Loan_Amount', 'Loan Amount', 'loan_amount']),
        ('CreditScore', ['CreditScore', 'Credit_Score', 'Credit Score']),
        ('MonthsEmployed', ['MonthsEmployed', 'Months_Employed', 'Months Employed']),
        ('NumCreditLines', ['NumCreditLines', 'Num_Credit_Lines', 'Num Credit Lines']),
        ('InterestRate', ['InterestRate', 'Interest_Rate', 'Interest Rate']),
        ('LoanTerm', ['LoanTerm', 'Tenure', 'TenureMonths', 'Loan_Term']),
        ('DTIRatio', ['DTIRatio', 'DTI', 'DTI_Ratio', 'DTI Ratio'])
    ]

    # Build list of actual available column names in this df_cluster
    use_features = []
    feature_name_map = {}  # maps canonical -> actual column name found
    for canon, candidates in preferred_features:
        found = None
        for c in candidates:
            if c in df_cluster.columns:
                found = c
                break
        if found:
            use_features.append(found)
            feature_name_map[canon] = found

    if len(use_features) == 0:
        return jsonify({"error": "No clustering features found in dataset. Expected columns like Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio."}), 500

    # Prepare the subset used for prediction (drop rows with NA in those features)
    df_sub = df_cluster[use_features].dropna().copy()
    if df_sub.shape[0] == 0:
        return jsonify({"error": "No rows available after dropping NA for clustering features."}), 500

    X = df_sub.values  # numpy array

    # Attempt prediction in several increasingly forgiving ways:
    clusters = None
    last_err = None

    # 1) Try direct predict (if kmeans was saved on same raw scale)
    try:
        clusters = kmeans_model.predict(X)
    except Exception as e:
        last_err = e
        # 2) Try using global scaler if it matches feature count
        try:
            if scaler is not None:
                # scaler may expect different number of features; only use if shapes match
                if hasattr(scaler, "mean_") and scaler.mean_.shape[0] == X.shape[1]:
                    X_scaled = scaler.transform(X)
                    clusters = kmeans_model.predict(X_scaled)
                else:
                    # scaler exists but shape mismatch -> fall through to fit-local-scaler
                    raise ValueError("Global scaler shape mismatch")
        except Exception as e2:
            last_err = e2
            # 3) Fit a local StandardScaler on these features (most robust fallback)
            try:
                temp_scaler = StandardScaler()
                X_temp_scaled = temp_scaler.fit_transform(X)
                clusters = kmeans_model.predict(X_temp_scaled)
            except Exception as e3:
                last_err = e3
                clusters = None

    if clusters is None:
        # print to server logs for debugging
        print("❌ Clustering prediction failed:", str(last_err))
        return jsonify({"error": f"Clustering prediction failed: {str(last_err)}"}), 500

    # Attach the cluster IDs back to the main dataframe using df_sub's index
    df_cluster.loc[df_sub.index, 'Cluster'] = clusters.astype(int)

    # Build a concise cluster summary (count, mean income if available, mean default if available)
    summary_records = []
    if 'Cluster' in df_cluster.columns:
        grouped = df_cluster.groupby('Cluster')
        for cluster_id, group in grouped:
            rec = {"Cluster": int(cluster_id), "Count": int(group.shape[0])}
            if 'Income' in group.columns:
                # ensure numeric conversion
                try:
                    rec['Income_Mean'] = float(group['Income'].dropna().mean())
                except Exception:
                    rec['Income_Mean'] = None
            # try default-like columns
            default_col = None
            for cand in ['Default_Risk', 'Default', 'DefaultRisk']:
                if cand in group.columns:
                    default_col = cand
                    break
            if default_col:
                try:
                    rec['Default_Mean'] = float(group[default_col].dropna().mean())
                except Exception:
                    rec['Default_Mean'] = None
            summary_records.append(rec)

    # Convert to JSON-safe records (replace NaN with empty string)
    clusters_records = df_cluster[use_features + ['Cluster']].fillna("").to_dict(orient="records")

    return jsonify({
        "clusters": clusters_records,
        "summary": summary_records
    })

# ===========================================================
# ✅ Repayment Timeline & Strategy Endpoint (kept intact)
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
# ✅ Chatbot Endpoint (Gemini) - kept intact
# ===========================================================
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json() or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"reply": "Please type your question."})

    if not GEMINI_API_KEY:
        return jsonify({"reply": "Gemini API key missing. Add it to your .env file."})

    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        payload = {"contents": [{"role": "user", "parts": [{"text": msg}]}]}

        response = requests.post(url, headers=headers, params=params, json=payload, timeout=15)
        if response.status_code != 200:
            return jsonify({"reply": f"Gemini API error: {response.status_code}"}), 500

        resp_json = response.json()
        reply_text = None
        if "candidates" in resp_json:
            cand = resp_json["candidates"][0]
            parts = cand.get("content", {}).get("parts", [])
            if parts and "text" in parts[0]:
                reply_text = parts[0]["text"]

        if not reply_text:
            reply_text = resp_json.get("output", "Sorry, I couldn’t get a proper reply.")
        return jsonify({"reply": reply_text})

    except Exception as e:
        return jsonify({"reply": f"Gemini API error: {str(e)}"})

# ===========================================================
# ✅ Run Server
# ===========================================================
if __name__ == "__main__":
    app.run(debug=True)
