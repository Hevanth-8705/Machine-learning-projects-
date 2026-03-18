"""
Phase 4: Automated Preprocessing Pipeline (Final Fixed Version)
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

print("=" * 60)
print("PHASE 4: Automated Preprocessing Pipeline")
print("=" * 60)

# ── 1. Load Data ─────────────────────────────────────────
raw_df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"\n✓ Raw data loaded: {raw_df.shape}")

# ── 2. Basic cleanup ─────────────────────────────────────
raw_df = raw_df.drop(columns=['customerID'])
raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
raw_df['Churn'] = raw_df['Churn'].map({'Yes': 1, 'No': 0})

X = raw_df.drop(columns=['Churn'])
y = raw_df['Churn']

# ── 3. Column groups ─────────────────────────────────────
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                   'PhoneService', 'PaperlessBilling']

binary_service_feats = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies']

categorical_features = ['InternetService', 'Contract', 'PaymentMethod']

all_binary = binary_features + binary_service_feats

# ── 4. Transformers ──────────────────────────────────────

# Numeric pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Binary pipeline (FIXED → using OneHotEncoder instead of custom encoder)
binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
])

# Categorical pipeline
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine everything
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('bin', binary_transformer, all_binary),
    ('cat', categorical_transformer, categorical_features)
])

# ── 5. Full Pipeline ─────────────────────────────────────
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# ── 6. Train/Test Split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── 7. Train Pipeline ────────────────────────────────────
print("\n── Training pipeline ──")
start = time.time()

pipeline.fit(X_train, y_train)

print(f"✓ Training completed in {(time.time()-start):.2f}s")

# ── 8. Evaluate ──────────────────────────────────────────
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n── Pipeline Performance ──")
print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\n", classification_report(y_test, y_pred))

# ── 9. Predict for Power BI ─────────────────────────────
print("\n── Generating predictions for Power BI ──")

pipeline.fit(X, y)

raw_export = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
raw_export['TotalCharges'] = pd.to_numeric(raw_export['TotalCharges'], errors='coerce')
raw_export['Churn_actual'] = raw_export['Churn'].map({'Yes': 1, 'No': 0})

X_all = raw_export.drop(columns=['customerID', 'Churn'])

proba = pipeline.predict_proba(X_all)[:, 1]
pred = pipeline.predict(X_all)

export_df = pd.DataFrame({
    'CustomerID': raw_export['customerID'],
    'Churn_Actual': raw_export['Churn_actual'],
    'Churn_Predicted': pred,
    'Churn_Probability': proba.round(4),
    'Risk_Tier': pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                        labels=['Low', 'Medium', 'High']),
    'tenure': raw_export['tenure'],
    'MonthlyCharges': raw_export['MonthlyCharges'],
    'Contract': raw_export['Contract'],
    'InternetService': raw_export['InternetService'],
    'PaymentMethod': raw_export['PaymentMethod'],
})

export_df.to_csv('predictions_powerbi.csv', index=False)

print(f"✓ Exported → predictions_powerbi.csv ({len(export_df):,} rows)")

# ── 10. Save Pipeline ────────────────────────────────────
joblib.dump(pipeline, 'full_pipeline.pkl')
print("✓ Pipeline saved → full_pipeline.pkl")

print("\nPhase 4 complete.\n")