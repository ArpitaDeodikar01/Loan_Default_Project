import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("="*80)
print("🎯 RISK-BASED CUSTOMER CLUSTERING SYSTEM")
print("="*80)

# Load data
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "loan_data.csv")
MODELS_DIR = os.path.join(BASE, "models")

print("\n📊 Loading data...")
df = pd.read_csv(DATA_PATH, engine='python', encoding='utf-8', on_bad_lines='skip')

# Feature selection - using the features you want for risk-based clustering
numeric_features = ['Income', 'CreditScore', 'Age', 'LoanAmount']
categorical_features = ['Education', 'MaritalStatus', 'EmploymentType']

# Ensure we have required columns
required_cols = numeric_features + categorical_features + ['Default']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"⚠️ Missing columns: {missing}")
    print(f"Available columns: {list(df.columns)}")
    raise ValueError(f"Missing required columns: {missing}")

# Clean data
df_cluster = df[required_cols].copy()
for col in numeric_features:
    df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
df_cluster['Default'] = pd.to_numeric(df_cluster['Default'], errors='coerce')

# Convert categorical to string
for col in categorical_features:
    df_cluster[col] = df_cluster[col].astype(str)

df_cluster = df_cluster.dropna()

print(f"✅ Loaded {len(df_cluster)} valid records")
print(f"📋 Numeric Features: {numeric_features}")
print(f"📋 Categorical Features: {categorical_features}")

# Create preprocessing pipeline with weighted features for better risk separation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# Fit and transform data
print("\n🔧 Preprocessing features...")
X = preprocessor.fit_transform(df_cluster[numeric_features + categorical_features])

# Create a risk score for better clustering
# Higher weight on CreditScore and Default history for risk-based clustering
print("\n🎯 Creating risk-weighted features...")
risk_features = df_cluster[numeric_features].copy()

# Normalize and create risk score components
risk_features['CreditScore_norm'] = (risk_features['CreditScore'] - risk_features['CreditScore'].min()) / (risk_features['CreditScore'].max() - risk_features['CreditScore'].min())
risk_features['Income_norm'] = (risk_features['Income'] - risk_features['Income'].min()) / (risk_features['Income'].max() - risk_features['Income'].min())
risk_features['LoanAmount_norm'] = (risk_features['LoanAmount'] - risk_features['LoanAmount'].min()) / (risk_features['LoanAmount'].max() - risk_features['LoanAmount'].min())

# Calculate risk score (lower credit score + higher loan amount + lower income = higher risk)
df_cluster['RiskScore'] = (
    (1 - risk_features['CreditScore_norm']) * 3.0 +  # Credit score impact (inverted)
    risk_features['LoanAmount_norm'] * 2.0 +          # Loan amount impact
    (1 - risk_features['Income_norm']) * 2.0 +        # Income impact (inverted)
    df_cluster['Default'] * 5.0                        # Default history (strongest weight)
) / 12.0  # Normalize to 0-1 range

print(f"✅ Total features after one-hot encoding: {X.shape[1]}")

# Train KMeans with 6 clusters for risk segmentation
print("\n🎯 Training KMeans with 6 risk clusters...")
kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=20,  # Increased for better convergence
    max_iter=500,  # Increased iterations
    algorithm='lloyd'  # More stable algorithm
)
kmeans.fit(X)

# Assign clusters to dataframe
df_cluster['Cluster'] = kmeans.labels_

# Calculate comprehensive risk metrics for each cluster
print("\n📈 Calculating cluster risk labels based on multiple factors...")
cluster_stats = df_cluster.groupby('Cluster').agg({
    'Default': 'mean',
    'RiskScore': 'mean',
    'CreditScore': 'mean',
    'Income': 'mean',
    'LoanAmount': 'mean'
}).round(4)

# Sort clusters by risk score (combination of default rate and risk score)
cluster_stats['CombinedRisk'] = (cluster_stats['Default'] * 0.6 + cluster_stats['RiskScore'] * 0.4)
cluster_stats = cluster_stats.sort_values('CombinedRisk', ascending=False)

# Assign risk labels and colors based on ranking
cluster_risk = {}
cluster_colors = {}

risk_levels = ['High Risk', 'High Risk', 'Medium-High Risk', 'Medium Risk', 'Low-Medium Risk', 'Low Risk']
risk_color_map = {
    'High Risk': '#dc3545',        # Red
    'Medium-High Risk': '#fd7e14',  # Orange
    'Medium Risk': '#ffc107',       # Yellow
    'Low-Medium Risk': '#20c997',   # Teal
    'Low Risk': '#28a745'           # Green
}

for idx, (cluster_id, row) in enumerate(cluster_stats.iterrows()):
    risk_label = risk_levels[idx]
    cluster_risk[cluster_id] = risk_label
    cluster_colors[cluster_id] = risk_color_map[risk_label]

# Calculate cluster statistics
print("\n📊 CLUSTER STATISTICS (Ranked by Risk):")
print("="*80)
for idx, (cluster_id, stats) in enumerate(cluster_stats.iterrows()):
    cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
    risk = cluster_risk[cluster_id]
    color = cluster_colors[cluster_id]
    default_rate = stats['Default']
    risk_score = stats['RiskScore']
    size = len(cluster_data)
    
    print(f"\n🎯 Cluster {cluster_id} - {risk}")
    print(f"   Color: {color}")
    print(f"   Size: {size} customers ({size/len(df_cluster)*100:.1f}%)")
    print(f"   Default Rate: {default_rate:.2%}")
    print(f"   Risk Score: {risk_score:.3f}")
    
    for feat in numeric_features:
        avg_val = cluster_data[feat].mean()
        if feat in ['Income', 'LoanAmount']:
            print(f"   Avg {feat}: ₹{avg_val:,.0f}")
        else:
            print(f"   Avg {feat}: {avg_val:,.1f}")

# Save models
os.makedirs(MODELS_DIR, exist_ok=True)

# IMPORTANT: Use different names to avoid conflict with train_models.py
print("\n💾 Saving risk clustering models...")
joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_clustering.joblib"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "cluster_preprocessor.joblib"))
joblib.dump(cluster_risk, os.path.join(MODELS_DIR, "cluster_risk_labels.joblib"))
joblib.dump(cluster_colors, os.path.join(MODELS_DIR, "cluster_colors.joblib"))

# Save cluster data for visualization
cluster_viz_data = df_cluster[numeric_features + ['Default', 'Cluster', 'RiskScore']].copy()
cluster_viz_data['Risk'] = cluster_viz_data['Cluster'].map(cluster_risk)
cluster_viz_data['Color'] = cluster_viz_data['Cluster'].map(cluster_colors)
cluster_viz_data.to_csv(os.path.join(BASE, "data", "cluster_data.csv"), index=False)

print("\n✅ MODELS SAVED SUCCESSFULLY!")
print("   📁 models/kmeans_clustering.joblib (Risk-based clustering)")
print("   📁 models/cluster_preprocessor.joblib")
print("   📁 models/cluster_risk_labels.joblib")
print("   📁 models/cluster_colors.joblib")
print("   📁 data/cluster_data.csv")
print(f"\n🎉 Total features in clustering model: {X.shape[1]}")
print("\n" + "="*80)
print("✨ SUCCESS! Now restart your Flask app and check Customer Clusters tab!")
print("="*80)

# Show risk distribution
print("\n📊 RISK DISTRIBUTION:")
risk_counts = df_cluster['Cluster'].map(cluster_risk).value_counts()
for risk_level in ['High Risk', 'Medium-High Risk', 'Medium Risk', 'Low-Medium Risk', 'Low Risk']:
    if risk_level in risk_counts:
        count = risk_counts[risk_level]
        pct = count/len(df_cluster)*100
        color = risk_color_map[risk_level]
        print(f"   {risk_level}: {count} customers ({pct:.1f}%)")