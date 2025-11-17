import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer








print("="*80)
print("🎯 RISK-BASED CUSTOMER CLUSTERING SYSTEM")
print("="*80)




# ==========================================================
# 1. Load data
# ==========================================================
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "loan_data.csv")
MODELS_DIR = os.path.join(BASE, "models")




print("\n📊 Loading data...")
df = pd.read_csv(DATA_PATH, engine='python', encoding='utf-8-sig', on_bad_lines='skip')




# ==========================================================
# 2. Feature selection
# ==========================================================
numeric_features = ['Income', 'CreditScore', 'Age', 'LoanAmount']
categorical_features = ['Education', 'MaritalStatus', 'EmploymentType']
risk_calc_features = ['Income', 'CreditScore', 'LoanAmount']  # For custom RiskScore




required_cols = numeric_features + categorical_features + ['Default']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"⚠️ Missing required columns: {missing}")




df_cluster = df[required_cols].copy()




for col in numeric_features:
    df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')




df_cluster['Default'] = pd.to_numeric(df_cluster['Default'], errors='coerce')




for col in categorical_features:
    df_cluster[col] = df_cluster[col].astype(str).fillna('Unknown')




df_cluster = df_cluster.dropna()
print(f"✅ Loaded {len(df_cluster)} valid records")




# ==========================================================
# 3. Scale numeric data for RiskScore
# ==========================================================
risk_scaler = MinMaxScaler()
risk_scaled = risk_scaler.fit_transform(df_cluster[risk_calc_features])
risk_df = pd.DataFrame(risk_scaled, columns=[f'{c}_norm' for c in risk_calc_features], index=df_cluster.index)




risk_scaler_params = {
    'min': risk_scaler.data_min_.tolist(),
    'max': risk_scaler.data_max_.tolist(),
    'features': risk_calc_features
}




# ==========================================================
# 4. Compute improved RiskScore (loan interacts with credit)
# ==========================================================
print("\n🎯 Creating improved risk-weighted features...")




credit_norm = risk_df['CreditScore_norm']
income_norm = risk_df['Income_norm']
loan_norm = risk_df['LoanAmount_norm']




# Interaction: high loan + low credit => much higher risk
loan_effect = loan_norm * (1 + (1 - credit_norm))




df_cluster['RiskScore'] = (
    (1 - credit_norm) * 4 +      # Poor credit = higher risk
    (1 - income_norm) * 2 +      # Low income = medium impact
    loan_effect * 5 +            # Loan impact depends on credit
    df_cluster['Default'] * 6    # Default history = strongest impact
) / 17.0   # Normalize total weight




# ==========================================================
# 5. Preprocessing for KMeans
# ==========================================================
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_features),  # keep MinMax for fair range-based distance
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
])




X = preprocessor.fit_transform(df_cluster[numeric_features + categorical_features])
feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))




# Combine features + weighted RiskScore
X_combined = np.hstack([X, df_cluster[['RiskScore']].values * 1.5])
final_feature_names = feature_names + ['RiskScore']




# ==========================================================
# 6. Train KMeans with 6 logical clusters
# ==========================================================
print("\n🎯 Training KMeans with 6 clusters...")
kmeans = KMeans(n_clusters=6, random_state=42, n_init=25, max_iter=600)
df_cluster['Cluster'] = kmeans.fit_predict(X_combined)




# ==========================================================
# 7. Cluster interpretation
# ==========================================================
print("\n📈 Analyzing cluster patterns...")
cluster_stats = df_cluster.groupby('Cluster').agg({
    'Default': 'mean',
    'RiskScore': 'mean',
    'CreditScore': 'mean',
    'Income': 'mean',
    'LoanAmount': 'mean'
}).round(3)




cluster_stats['CombinedRisk'] = (0.7 * cluster_stats['Default'] + 0.3 * cluster_stats['RiskScore'])
cluster_stats = cluster_stats.sort_values('CombinedRisk', ascending=False)




# Logical color-coded risk levels
risk_labels = ['Very High Risk', 'High Risk', 'Medium-High Risk', 'Medium Risk', 'Low-Medium Risk', 'Low Risk']
risk_colors = ['#8B0000', '#DC3545', '#FD7E14', '#FFC107', '#17A2B8', '#28A745']




cluster_risk = {}
cluster_colors = {}




for i, (cluster_id, _) in enumerate(cluster_stats.iterrows()):
    cluster_risk[cluster_id] = risk_labels[i]
    cluster_colors[cluster_id] = risk_colors[i]




print("\n📊 CLUSTER OVERVIEW:")
print("="*80)
for cluster_id, stats in cluster_stats.iterrows():
    risk = cluster_risk[cluster_id]
    print(f"\n🔹 Cluster {cluster_id}: {risk}")
    print(f"   Default Rate: {stats['Default']:.2%}")
    print(f"   RiskScore Avg: {stats['RiskScore']:.3f}")
    print(f"   Avg CreditScore: {stats['CreditScore']:.1f}")
    print(f"   Avg Income: ₹{stats['Income']:,.0f}")
    print(f"   Avg Loan: ₹{stats['LoanAmount']:,.0f}")




# ==========================================================
# 8. Save models and metadata
# ==========================================================
os.makedirs(MODELS_DIR, exist_ok=True)




print("\n💾 Saving models and metadata...")
joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_clustering.joblib"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "cluster_preprocessor.joblib"))
joblib.dump(cluster_risk, os.path.join(MODELS_DIR, "cluster_risk_labels.joblib"))
joblib.dump(cluster_colors, os.path.join(MODELS_DIR, "cluster_colors.joblib"))
joblib.dump(risk_scaler_params, os.path.join(MODELS_DIR, "risk_scaler_params.joblib"))
joblib.dump(final_feature_names, os.path.join(MODELS_DIR, "cluster_feature_names.joblib"))




cluster_viz_data = df_cluster[numeric_features + ['Default', 'Cluster', 'RiskScore']].copy()
cluster_viz_data['Risk'] = cluster_viz_data['Cluster'].map(cluster_risk)
cluster_viz_data['Color'] = cluster_viz_data['Cluster'].map(cluster_colors)
cluster_viz_data.to_csv(os.path.join(BASE, "data", "cluster_data.csv"), index=False)




print("\n✅ MODELS SAVED SUCCESSFULLY!")
print("="*80)
print("✨ SUCCESS! Loan ↑ with low credit now correctly raises risk and cluster logic is stable.")
print("="*80)
