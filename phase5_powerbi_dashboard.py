"""
Phase 5: Power BI Dashboard Prep & Final Summary Visualisation
- Generates all KPI data for the Power BI dashboard
- Creates a matplotlib "mock dashboard" as portfolio screenshot
- Prints Power BI connection instructions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

PALETTE = ['#378ADD', '#1D9E75', '#E24B4A', '#EF9F27', '#7F77DD']

print("=" * 60)
print("PHASE 5: Power BI Dashboard & Final Summary")
print("=" * 60)

# ── 1. Load predictions export ────────────────────────────────────────────────
df = pd.read_csv('predictions_powerbi.csv')
print(f"\n✓ Loaded predictions: {df.shape}")

# ── 2. KPI Calculations ───────────────────────────────────────────────────────
total_customers   = len(df)
actual_churners   = df['Churn_Actual'].sum()
predicted_churners = df['Churn_Predicted'].sum()
churn_rate_actual = df['Churn_Actual'].mean() * 100
accuracy          = (df['Churn_Actual'] == df['Churn_Predicted']).mean() * 100

risk_dist = df['Risk_Tier'].value_counts()

print(f"\n── KPIs ──")
print(f"  Total customers     : {total_customers:,}")
print(f"  Actual churners     : {actual_churners:,} ({churn_rate_actual:.1f}%)")
print(f"  Predicted churners  : {predicted_churners:,}")
print(f"  Model accuracy      : {accuracy:.1f}%")
print(f"\n  Risk tier distribution:\n{risk_dist}")

# ── 3. Segment analysis ───────────────────────────────────────────────────────
contract_summary = df.groupby('Contract').agg(
    Customers=('CustomerID', 'count'),
    Actual_Churn_Rate=('Churn_Actual', lambda x: f"{x.mean()*100:.1f}%"),
    Predicted_Churn_Rate=('Churn_Predicted', lambda x: f"{x.mean()*100:.1f}%"),
    Avg_Monthly_Charges=('MonthlyCharges', lambda x: f"${x.mean():.0f}"),
).reset_index()
print(f"\n── By Contract Type ──\n{contract_summary.to_string(index=False)}")

internet_summary = df.groupby('InternetService').agg(
    Customers=('CustomerID', 'count'),
    Churn_Rate=('Churn_Actual', lambda x: f"{x.mean()*100:.1f}%"),
    Avg_Churn_Prob=('Churn_Probability', lambda x: f"{x.mean():.3f}")
).reset_index()
print(f"\n── By Internet Service ──\n{internet_summary.to_string(index=False)}")

# ── 4. Matplotlib Mock Power BI Dashboard ────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor='#F0F2F5')
fig.suptitle('Predictive Analytics Dashboard — Customer Churn',
             fontsize=18, fontweight='bold', y=0.98, color='#1A1A2E')

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.04, right=0.97, top=0.92, bottom=0.05)

def add_kpi_card(ax, title, value, subtitle, color):
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#E0E0E0')
        spine.set_linewidth(0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.72, value, transform=ax.transAxes, fontsize=26,
            fontweight='bold', ha='center', va='center', color=color)
    ax.text(0.5, 0.35, title, transform=ax.transAxes, fontsize=11,
            ha='center', va='center', color='#555555')
    ax.text(0.5, 0.12, subtitle, transform=ax.transAxes, fontsize=9,
            ha='center', va='center', color='#888888')

# KPI row
ax_k1 = fig.add_subplot(gs[0, 0])
ax_k2 = fig.add_subplot(gs[0, 1])
ax_k3 = fig.add_subplot(gs[0, 2])
ax_k4 = fig.add_subplot(gs[0, 3])

add_kpi_card(ax_k1, 'Total Customers', f'{total_customers:,}', 'Active accounts', '#378ADD')
add_kpi_card(ax_k2, 'Actual Churn Rate', f'{churn_rate_actual:.1f}%', f'{actual_churners:,} customers lost', '#E24B4A')
add_kpi_card(ax_k3, 'Model Accuracy', f'{accuracy:.1f}%', 'Random Forest (CV)', '#1D9E75')
add_kpi_card(ax_k4, 'High Risk Customers',
             f'{risk_dist.get("High", 0):,}',
             'Churn probability > 60%', '#EF9F27')

# Churn probability distribution
ax1 = fig.add_subplot(gs[1, :2])
ax1.set_facecolor('white')
bins = np.linspace(0, 1, 25)
no_churn_probs = df[df['Churn_Actual'] == 0]['Churn_Probability']
churn_probs    = df[df['Churn_Actual'] == 1]['Churn_Probability']
ax1.hist(no_churn_probs, bins=bins, alpha=0.65, color=PALETTE[1], label='No Churn')
ax1.hist(churn_probs,    bins=bins, alpha=0.65, color=PALETTE[2], label='Churn')
ax1.axvline(0.5, color='gray', linestyle='--', linewidth=1.2, label='Decision boundary (0.5)')
ax1.set_title('Predicted Churn Probability Distribution', fontweight='bold')
ax1.set_xlabel('Churn Probability'); ax1.set_ylabel('Customer Count')
ax1.legend(fontsize=9)
for s in ['top', 'right']: ax1.spines[s].set_visible(False)

# Churn rate by contract
ax2 = fig.add_subplot(gs[1, 2])
ax2.set_facecolor('white')
contract_churn = df.groupby('Contract')['Churn_Actual'].mean() * 100
contract_pred  = df.groupby('Contract')['Churn_Predicted'].mean() * 100
x_pos = np.arange(len(contract_churn))
ax2.bar(x_pos - 0.2, contract_churn.values, 0.38, label='Actual', color=PALETTE[2], alpha=0.85)
ax2.bar(x_pos + 0.2, contract_pred.values,  0.38, label='Predicted', color=PALETTE[0], alpha=0.85)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([c.replace('-', '\n') for c in contract_churn.index], fontsize=8)
ax2.set_title('Churn Rate by Contract', fontweight='bold')
ax2.set_ylabel('Churn Rate (%)')
ax2.legend(fontsize=8)
for s in ['top', 'right']: ax2.spines[s].set_visible(False)

# Risk tier donut
ax3 = fig.add_subplot(gs[1, 3])
ax3.set_facecolor('white')
tier_order = ['Low', 'Medium', 'High']
tier_vals  = [risk_dist.get(t, 0) for t in tier_order]
tier_colors = [PALETTE[1], PALETTE[3], PALETTE[2]]
wedges, texts, autotexts = ax3.pie(
    tier_vals, labels=tier_order, colors=tier_colors, autopct='%1.0f%%',
    startangle=90, wedgeprops={'width': 0.55}, pctdistance=0.75
)
for at in autotexts: at.set_fontsize(9)
ax3.set_title('Customers by Risk Tier', fontweight='bold')

# Monthly charges vs tenure scatter
ax4 = fig.add_subplot(gs[2, :2])
ax4.set_facecolor('white')
sample = df.sample(min(1500, len(df)), random_state=42)
colors_scatter = [PALETTE[2] if c == 1 else PALETTE[1] for c in sample['Churn_Actual']]
ax4.scatter(sample['tenure'], sample['MonthlyCharges'], c=colors_scatter,
            alpha=0.35, s=18, edgecolors='none')
ax4.set_title('Tenure vs Monthly Charges (coloured by Churn)', fontweight='bold')
ax4.set_xlabel('Tenure (months)'); ax4.set_ylabel('Monthly Charges ($)')
legend_patches = [mpatches.Patch(color=PALETTE[1], label='No Churn'),
                  mpatches.Patch(color=PALETTE[2], label='Churn')]
ax4.legend(handles=legend_patches, fontsize=9)
for s in ['top', 'right']: ax4.spines[s].set_visible(False)

# Avg churn probability by payment method
ax5 = fig.add_subplot(gs[2, 2:])
ax5.set_facecolor('white')
pay_risk = df.groupby('PaymentMethod')['Churn_Probability'].mean().sort_values()
short = [p.replace('(automatic)', '(auto)') for p in pay_risk.index]
bars = ax5.barh(short, pay_risk.values * 100, color=PALETTE[0], alpha=0.85, edgecolor='white')
for bar, val in zip(bars, pay_risk.values * 100):
    ax5.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
ax5.set_title('Avg Churn Probability by Payment Method', fontweight='bold')
ax5.set_xlabel('Avg Churn Probability (%)')
for s in ['top', 'right']: ax5.spines[s].set_visible(False)

plt.savefig('powerbi_dashboard_preview.png', dpi=150, bbox_inches='tight',
            facecolor='#F0F2F5')
print("\n✓ Dashboard preview saved → powerbi_dashboard_preview.png")

# ── 5. Power BI Instructions ─────────────────────────────────────────────────
print("""
── Power BI Connection Instructions ──

1. Open Power BI Desktop
2. Home → Get Data → Text/CSV
3. Select: predictions_powerbi.csv
4. Transform Data (Power Query):
   - Verify Risk_Tier is text type
   - Verify Churn_Probability is decimal
5. Build visuals:
   ┌─────────────────────────────────────────────────────┐
   │  RECOMMENDED VISUALS                                │
   │                                                     │
   │  • Card: Total Customers, Churn Rate, Accuracy      │
   │  • Donut: Risk_Tier distribution                    │
   │  • Clustered bar: Churn rate by Contract            │
   │  • Scatter: tenure vs MonthlyCharges (by Churn)     │
   │  • Histogram: Churn_Probability distribution        │
   │  • Table: High risk customers (Risk_Tier = High)    │
   │  • Slicer: Contract, InternetService, Risk_Tier     │
   └─────────────────────────────────────────────────────┘
6. Add DAX measures:
   Churn Rate = DIVIDE(COUNTROWS(FILTER(predictions_powerbi, [Churn_Actual]=1)), COUNTROWS(predictions_powerbi))
   Model Accuracy = DIVIDE(COUNTROWS(FILTER(predictions_powerbi, [Churn_Actual]=[Churn_Predicted])), COUNTROWS(predictions_powerbi))
""")

print("Phase 5 complete.")
print("\n" + "=" * 60)
print("ALL PHASES COMPLETE — Project ready for portfolio!")
print("=" * 60)
print("""
Files generated:
  phase1_eda.py                  → EDA notebook
  phase2_feature_engineering.py → Feature selection
  phase3_model_training.py       → Model comparison
  phase4_pipeline.py             → Automated pipeline
  phase5_powerbi_dashboard.py    → Dashboard prep

Saved artefacts:
  churn_cleaned.csv              → Cleaned dataset
  train_processed.csv            → Processed train split
  test_processed.csv             → Processed test split
  predictions_powerbi.csv        → Power BI data source
  model_results_summary.csv      → Model metrics table
  scaler.pkl                     → StandardScaler
  feature_selector.pkl           → Feature selector
  best_model.pkl                 → Best trained model
  full_pipeline.pkl              → Production pipeline

Plots:
  eda_report.png                 → EDA visuals
  feature_selection.png          → Feature importance
  model_comparison.png           → Model metrics & ROC
  cv_folds.png                   → Cross-validation
  powerbi_dashboard_preview.png  → Dashboard screenshot
""")