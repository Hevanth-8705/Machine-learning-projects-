"""
Phase 3: Model Training, Cross-Validation & Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer  # ✅ NEW
import joblib

PALETTE = ['#378ADD', '#1D9E75', '#E24B4A', '#EF9F27', '#7F77DD']

print("=" * 60)
print("PHASE 3: Model Training & Comparison")
print("=" * 60)

# ── 1. Load Processed Data ─────────────────────────────────────────
train_df = pd.read_csv('train_processed.csv')
test_df  = pd.read_csv('test_processed.csv')
selected_features = joblib.load('selected_features.pkl')

X_train = train_df[selected_features]
y_train = train_df['Churn']
X_test  = test_df[selected_features]
y_test  = test_df['Churn']

print(f"\n✓ Train: {X_train.shape} | Test: {X_test.shape}")
print(f"✓ Features: {selected_features}")

# ── 2. HANDLE MISSING VALUES (FIX) ────────────────────────────────
imputer = SimpleImputer(strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

# Debug check
print("\nNaN check after imputation:")
print("Train NaNs:", np.isnan(X_train).sum())
print("Test NaNs :", np.isnan(X_test).sum())

# ── 3. Define Models ─────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True, random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=0.5, class_weight='balanced', max_iter=1000, random_state=42
    )
}

# ── 4. Cross-Validation ──────────────────────────────────────────
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = {}
print("\n── 10-Fold Cross-Validation ──")

for name, model in models.items():
    print(f"\n  Training {name}...")
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    cv_results[name] = {
        'accuracy':  scores['test_accuracy'].mean(),
        'precision': scores['test_precision'].mean(),
        'recall':    scores['test_recall'].mean(),
        'f1':        scores['test_f1'].mean(),
        'auc_roc':   scores['test_roc_auc'].mean(),
        'cv_accuracy_std': scores['test_accuracy'].std(),
        'cv_folds': scores['test_accuracy'],
    }

    r = cv_results[name]
    print(f"    Accuracy : {r['accuracy']:.4f} ± {r['cv_accuracy_std']:.4f}")
    print(f"    Precision: {r['precision']:.4f}")
    print(f"    Recall   : {r['recall']:.4f}")
    print(f"    F1 Score : {r['f1']:.4f}")
    print(f"    AUC-ROC  : {r['auc_roc']:.4f}")

# ── 5. Final Training ────────────────────────────────────────────
print("\n── Training final models on full train set ──")

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"  ✓ {name} trained")

# ── 6. Test Evaluation ───────────────────────────────────────────
print("\n── Test Set Evaluation ──")

test_results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_results[name] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc_roc':   roc_auc_score(y_test, y_prob),
        'cm':        confusion_matrix(y_test, y_pred),
        'y_prob': y_prob
    }

    r = test_results[name]
    print(f"\n{name}:")
    print(f"    Accuracy : {r['accuracy']:.4f}")
    print(f"    Precision: {r['precision']:.4f}")
    print(f"    Recall   : {r['recall']:.4f}")
    print(f"    F1 Score : {r['f1']:.4f}")
    print(f"    AUC-ROC  : {r['auc_roc']:.4f}")

# Best model
best_model_name = max(test_results, key=lambda k: test_results[k]['f1'])
print(f"\n★ Best model: {best_model_name}")

# ── 7. Save Model ────────────────────────────────────────────────
joblib.dump(trained_models[best_model_name], 'best_model.pkl')
print("✓ Best model saved → best_model.pkl")

print("\nPhase 3 complete.\n")