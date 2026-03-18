"""
Phase 1: Data Ingestion, Cleaning & Exploratory Data Analysis
Dataset: Telco Customer Churn
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F8F8',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})

PALETTE = ['#378ADD', '#1D9E75', '#E24B4A', '#EF9F27', '#7F77DD']

print("=" * 60)
print("PHASE 1: Data Ingestion & EDA")
print("=" * 60)

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"\n✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\n── Data Types & Missing Values ──")
print(df.dtypes)
print(df.isnull().sum()[df.isnull().sum() > 0])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.drop(columns=['customerID'], inplace=True)

df['TotalCharges'].fillna(0, inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Plot
plt.figure(figsize=(10,6))
df['Churn'].value_counts().plot(kind='bar')
plt.title("Churn Distribution")
plt.savefig("eda_report.png")

df.to_csv('churn_cleaned.csv', index=False)

print("\n✓ Phase 1 completed successfully")