"""
Phase 2: Feature Engineering & Selection
- Encode categoricals (Label + OneHot)
- Scale numerics
- Feature selection via RFE + Feature Importance (30% reduction)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import train_test_split

PALETTE = ['#378ADD', '#1D9E75', '#E24B4A', '#EF9F27', '#7F77DD']

print("=" * 60)
print("PHASE 2: Feature Engineering & Selection")
print("=" * 60)

# ── 1. Load Cleaned Data ──────────────────────────────────────────────────────
df = pd.read_csv('churn_cleaned.csv')
print(f"\n✓ Loaded cleaned data: {df.shape}")

# ── 2. Binary columns → 0/1 ──────────────────────────────────────────────────
binary_yes_no = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
binary_services = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in binary_yes_no:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    print(f"  Binary encoded: {col}")

for col in binary_services:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
    print(f"  Binary encoded: {col}")

# Gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
print(f"  Binary encoded: gender")

# ── 3. One-Hot Encode multi-class categoricals ────────────────────────────────
multi_cat = ['InternetService', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cat, drop_first=True)
print(f"\n✓ One-hot encoded: {multi_cat}")
print(f"  Dataset shape after encoding: {df.shape}")

# ── 4. Feature / Target split ─────────────────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']
feature_names = X.columns.tolist()
print(f"\n✓ Total features before selection: {len(feature_names)}")

# ── 5. Train / Test split (before scaling — prevents data leakage) ────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} (80/20 stratified)")

# ── 6. Scale Numeric Features ─────────────────────────────────────────────────
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])   # use train stats only
print(f"✓ StandardScaler applied to: {numeric_cols}")

# ── 7. Feature Importance via Random Forest ───────────────────────────────────
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

importances = pd.Series(rf_selector.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

print(f"\n── Feature Importances (Random Forest) ──")
for feat, imp in importances.items():
    bar = '█' * int(imp * 300)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ── 8. SelectFromModel — threshold = mean importance ─────────────────────────
selector = SelectFromModel(rf_selector, threshold='mean', prefit=True)
X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)

selected_mask     = selector.get_support()
selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
removed_features  = [f for f, s in zip(feature_names, selected_mask) if not s]

n_orig     = len(feature_names)
n_selected = len(selected_features)
reduction  = (1 - n_selected / n_orig) * 100

print(f"\n✓ Feature selection complete!")
print(f"  Original features : {n_orig}")
print(f"  Selected features : {n_selected}")
print(f"  Removed features  : {len(removed_features)}")
print(f"  Dimensionality reduction: {reduction:.1f}%")
print(f"\n  Selected: {selected_features}")
print(f"  Removed : {removed_features}")

# ── 9. Visualise Feature Importance ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Feature Engineering & Selection', fontsize=14, fontweight='bold')

# All features
ax = axes[0]
colors = [PALETTE[1] if s else '#D3D1C7' for s in selected_mask[importances.index.map(lambda x: feature_names.index(x))]]
# Rebuild aligned arrays
imp_vals   = importances.values
imp_labels = importances.index.tolist()
feat_colors = [PALETTE[1] if f in selected_features else '#B4B2A9' for f in imp_labels]

bars = ax.barh(imp_labels[::-1], imp_vals[::-1], color=feat_colors[::-1], edgecolor='white')
ax.set_title(f'All Features — Importance Score\n(green = selected, {n_selected}/{n_orig})')
ax.set_xlabel('Feature Importance')
ax.axvline(importances.mean(), color=PALETTE[2], linestyle='--', linewidth=1.2, label=f'Mean threshold ({importances.mean():.3f})')
ax.legend(fontsize=9)

# Selected only
ax2 = axes[1]
sel_imp = importances[selected_features].sort_values()
ax2.barh(sel_imp.index, sel_imp.values, color=PALETTE[0], edgecolor='white')
ax2.set_title(f'Selected {n_selected} Features\n({reduction:.0f}% dimensionality reduction)')
ax2.set_xlabel('Feature Importance')
for i, (feat, val) in enumerate(sel_imp.items()):
    ax2.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('feature_selection.png', dpi=150, bbox_inches='tight')
print("\n✓ Feature selection plot saved → feature_selection.png")

# ── 10. Save artefacts ────────────────────────────────────────────────────────
import joblib
joblib.dump(scaler,           'scaler.pkl')
joblib.dump(selector,         'feature_selector.pkl')
joblib.dump(selected_features,'selected_features.pkl')

# Save processed splits with selected features
X_train_df = pd.DataFrame(X_train_sel, columns=selected_features)
X_test_df  = pd.DataFrame(X_test_sel,  columns=selected_features)
X_train_df['Churn'] = y_train.values
X_test_df['Churn']  = y_test.values

X_train_df.to_csv('train_processed.csv', index=False)
X_test_df.to_csv('test_processed.csv',  index=False)

print("✓ Saved: scaler.pkl | feature_selector.pkl | selected_features.pkl")
print("✓ Saved: train_processed.csv | test_processed.csv")
print(f"\nX_train shape: {X_train_sel.shape}")
print(f"X_test  shape: {X_test_sel.shape}")
print("\nPhase 2 complete.\n")