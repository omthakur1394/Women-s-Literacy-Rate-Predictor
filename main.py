import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Load and prepare data ---
# Use the correct file name
df = pd.read_csv("datafile.csv")

# --- Clean column names (remove leading/trailing spaces) ---
df.columns = df.columns.str.strip()

# --- Target ---
target = "Women (age 15-49) who are literate4 (%)"

# --- Features (removed the column that was not in the CSV) ---
features = [
    "Female population age 6 years and above who ever attended school (%)",
    "Women (age 15-49)  with 10 or more years of schooling (%)",
    "Women (age 15-49)  who have ever used the internet (%)",
    "Population living in households with electricity (%)",
    "Households using clean fuel for cooking3 (%)",
    "Population living in households that use an improved sanitation facility2 (%)"
]

# --- Data Cleaning ---
# Convert all relevant columns to a numeric format
# This will replace non-numeric values like '(19.5)' with numbers
# and turn any errors into NaN (Not a Number)
all_cols = features + [target]
for col in all_cols:
    # Ensure column exists before trying to clean it
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[()]', '', regex=True), errors='coerce')

# --- Drop rows with any null values in the selected columns ---
df = df.dropna(subset=all_cols)

# --- Prepare X and y ---
X = df[features]
y = df[target]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --- Model (Random Forest) ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Prediction ---
y_pred = model.predict(X_test)

# --- Evaluation ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")