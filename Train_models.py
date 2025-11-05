# ============================================
# train_models.py (Fully Fixed & Synced Version)
# ============================================

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from scipy.stats import randint

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

# ============================================
# STEP 1: Load dataset
# ============================================
DATA_PATH = os.path.join("data", "loan_data.csv")

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"❌ Error: '{DATA_PATH}' not found. Please make sure the file exists.")

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully!")

# ============================================
# STEP 2: Feature selection
# ============================================
feature_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]
target_col = 'Default'

# ============================================
# STEP 3: Encode categorical features
# ============================================
categorical_cols = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================================
# STEP 4: Split data
# ============================================
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# STEP 5: Scaling + PCA
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ============================================
# STEP 6: Fast Random Forest
# ============================================
best_rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train_scaled, y_train)

y_pred_rf = best_rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"🌲 Random Forest Accuracy: {acc_rf:.4f}")

# ============================================
# STEP 7: KNN Classifier
# ============================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"🤖 KNN Accuracy: {acc_knn:.4f}")

# ============================================
# STEP 8: Fast KMeans Clustering (Customer Segmentation)
# ============================================

# Choose meaningful numeric segmentation features
cluster_features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'MonthsEmployed', 'NumCreditLines', 'InterestRate',
    'LoanTerm', 'DTIRatio'
]

# Scale only these features
X_cluster = df[cluster_features]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Use fewer clusters and faster convergence
kmeans = KMeans(
    n_clusters=4,        # 4 customer groups instead of 5
    random_state=42,
    max_iter=200,
    n_init='auto'        # much faster initialization
)
kmeans.fit(X_cluster_scaled)

print("🔹 KMeans clustering done successfully!")
print(f"Cluster Centers Shape: {kmeans.cluster_centers_.shape}")


# ============================================
# STEP 9: Save models
# ============================================
os.makedirs("models", exist_ok=True)

joblib.dump(best_rf, "models/random_forest_model.joblib")
joblib.dump(knn, "models/knn_model.joblib")
joblib.dump(kmeans, "models/kmeans_model.joblib")
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
