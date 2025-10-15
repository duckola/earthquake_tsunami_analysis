# ============================================
#  PHILIPPINES TYPHOON DATA REGRESSION ANALYSIS
#  SAVE ALL RESULTS IN ONE EXCEL WORKBOOK
# ============================================

# 🔹 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# ============================================
# 2. LOAD THE DATA
# ============================================

# Replace with your file path if needed
file_path = "philippines_typhoon_monthly_2014_2024.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ============================================
# 3. SELECT VARIABLES
# ============================================

# Dependent variable (Y)
dep_var = "Number_of_Typhoons"

# Independent variables (X)
iv1 = "Vertical_Wind_Shear"
iv2 = "Nino3.4_SST_anomaly"
iv3 = "SeaLevelPressure"

ivs = [iv1, iv2, iv3]

# Clean dataset: numeric conversion and drop missing values
data = df[[dep_var] + ivs].copy()
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

print(f"\n Cleaned data rows: {data.shape[0]}")
print(data.describe())

# ============================================
# 4. SCATTER PLOTS
# ============================================

os.makedirs("plots", exist_ok=True)

for col in ivs:
    plt.figure(figsize=(6, 4))
    plt.scatter(data[col], data[dep_var], color='royalblue', alpha=0.7)
    plt.title(f"{dep_var} vs {col}")
    plt.xlabel(col)
    plt.ylabel(dep_var)
    # Add a linear trendline
    m, b = np.polyfit(data[col], data[dep_var], 1)
    xs = np.linspace(data[col].min(), data[col].max(), 100)
    plt.plot(xs, m * xs + b, color='red')
    plt.tight_layout()
    plt.savefig(f"plots/{dep_var}_vs_{col}.png")
    plt.close()

print(" Scatterplots saved in the 'plots' folder")

# ============================================
# 5. MULTIPLE REGRESSION MODELS
# ============================================

def run_regression(df, y_col, x_cols):
    """Helper to fit an OLS regression model"""
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    return sm.OLS(y, X).fit()

# Build four models
model_3IV = run_regression(data, dep_var, ivs)
model_12 = run_regression(data, dep_var, [iv1, iv2])
model_13 = run_regression(data, dep_var, [iv1, iv3])
model_23 = run_regression(data, dep_var, [iv2, iv3])

# Compare Adjusted R²
model_results = pd.DataFrame({
    "Model": [
        f"3 IVs ({iv1}, {iv2}, {iv3})",
        f"2 IVs ({iv1}, {iv2})",
        f"2 IVs ({iv1}, {iv3})",
        f"2 IVs ({iv2}, {iv3})"
    ],
    "R_squared": [
        model_3IV.rsquared,
        model_12.rsquared,
        model_13.rsquared,
        model_23.rsquared
    ],
    "Adj_R_squared": [
        model_3IV.rsquared_adj,
        model_12.rsquared_adj,
        model_13.rsquared_adj,
        model_23.rsquared_adj
    ]
})

print("\nModel Comparison (Adjusted R²):")
print(model_results)

# ============================================
# 6. SELECT THE BEST MODEL
# ============================================

best_idx = model_results["Adj_R_squared"].idxmax()
best_model_name = model_results.loc[best_idx, "Model"]
best_adjR2 = model_results.loc[best_idx, "Adj_R_squared"]

best_model = [model_3IV, model_12, model_13, model_23][best_idx]

print("\n Best model:", best_model_name)
print("Adjusted R² =", round(best_adjR2, 4))

# ============================================
# 7. COEFFICIENT HYPOTHESIS TESTING
# ============================================

coef_table = pd.DataFrame({
    "Term": best_model.params.index,
    "Coefficient": best_model.params.values,
    "StdErr": best_model.bse.values,
    "t": best_model.tvalues.values,
    "p-value": best_model.pvalues.values,
    "CI_low": best_model.conf_int()[0].values,
    "CI_high": best_model.conf_int()[1].values
})

print("\nCoefficient Table:")
print(coef_table)

# ============================================
# 8. FORECASTING
# ============================================

means = data[ivs].mean().to_dict()
new_input = pd.DataFrame([means])
X_new = sm.add_constant(new_input)

forecast = best_model.get_prediction(X_new).summary_frame(alpha=0.05)

print("\nForecast Result:")
print(forecast)

# ============================================
# 9. SAVE EVERYTHING TO ONE EXCEL FILE
# ============================================

output_path = "typhoon_analysis_results.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)
    data.describe().to_excel(writer, sheet_name="Summary Statistics")
    model_results.to_excel(writer, sheet_name="Model Comparison", index=False)
    coef_table.to_excel(writer, sheet_name="Best Model Coefficients", index=False)
    forecast.to_excel(writer, sheet_name="Forecast Result")

print(f"\n All outputs saved in: {output_path}")

# ============================================
# 10. OPTIONAL: Display Model Summary
# ============================================
print("\n Full regression summary (best model):")
print(best_model.summary())
