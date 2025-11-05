# ============================================
# train_models.py (Fully Fixed Version)
# ============================================

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ============================================
# STEP 1: Load dataset
# ============================================
DATA_PATH = os.path.join("data", "loan_data.csv")
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"❌ Error: '{DATA_PATH}' not found.")

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully!")

# ============================================
# STEP 2: Features & target
# ============================================
feature_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]
target_col = 'Default'

categorical_cols = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

# ============================================
# STEP 3: Encode categorical features
# ============================================
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================================
# STEP 4: Train-test split
# ============================================
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# STEP 5: Scaling + PCA (for visualization)
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ============================================
# STEP 6: Random Forest Classifier
# ============================================
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
print(f"🌲 Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# ============================================
# STEP 7: KNN Classifier
# ============================================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
print(f"🤖 KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

# ============================================
# STEP 8: KMeans Clustering (6 clusters)
# ============================================
cluster_features = [
    'Income', 'CreditScore', 'Age', 'Education',
    'MaritalStatus', 'EmploymentType'
]

X_cluster = df[cluster_features].copy()  # already numeric
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

kmeans_model = KMeans(n_clusters=6, random_state=42, n_init='auto')
kmeans_model.fit(X_cluster_scaled)

df['Cluster'] = kmeans_model.labels_

# Assign risk label for each cluster based on average default probability
cluster_risks = {}
for c in range(6):
    avg_default = df[df['Cluster'] == c][target_col].mean()
    if avg_default >= 0.7:
        cluster_risks[c] = "High Risk"
    elif avg_default >= 0.4:
        cluster_risks[c] = "Medium Risk"
    else:
        cluster_risks[c] = "Low Risk"

df['Cluster_Risk'] = df['Cluster'].map(cluster_risks)

print("🔹 KMeans clustering done successfully!")
print("Cluster Risk Assignment:", cluster_risks)

# ============================================
# STEP 9: Save models & encoders
# ============================================
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest_model.joblib")
joblib.dump(knn_model, "models/knn_model.joblib")
joblib.dump(kmeans_model, "models/kmeans_model.joblib")
joblib.dump(pca, "models/pca.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(encoders, "models/encoders.joblib")

print("\n✅ All models and encoders saved in 'models/' folder!")

# ============================================
# STEP 10: Summary
# ============================================
print("\n=== Classification Report (Random Forest) ===")
print(classification_report(y_test, y_pred_rf))
print("\n=== Classification Report (KNN) ===")
print(classification_report(y_test, y_pred_knn))
print("\n✅ Training pipeline completed successfully!")
