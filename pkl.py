import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- Load and prepare data ---
df = pd.read_csv("datafile.csv")
df.columns = df.columns.str.strip()

# --- Define target and features ---
target = "Women (age 15-49) who are literate4 (%)"
features = [
    "Female population age 6 years and above who ever attended school (%)",
    "Women (age 15-49)  with 10 or more years of schooling (%)",
    "Women (age 15-49)  who have ever used the internet (%)",
    "Population living in households with electricity (%)",
    "Households using clean fuel for cooking3 (%)",
    "Population living in households that use an improved sanitation facility2 (%)"
]
all_cols = features + [target]

# --- Clean and convert data ---
for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[()]', '', regex=True), errors='coerce')

# --- Finalize the cleaned dataframe ---
df_cleaned = df.dropna(subset=all_cols)

# --- Prepare X and y for final model training ---
X = df_cleaned[features]
y = df_cleaned[target]

# --- Train the final model ---
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

# --- Save everything needed for the app in one file ---
# We save the model, the cleaned data for exploration, and the feature list.
app_data = {
    'model': final_model,
    'data': df_cleaned,
    'features': features
}

with open('app_data.pkl', 'wb') as file:
    pickle.dump(app_data, file)

print("Professional app data saved to app_data.pkl")