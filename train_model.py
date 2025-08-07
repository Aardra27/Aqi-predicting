import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from google.colab import files
import io

# Upload merged CSV with AQI
print("📂 Upload CSV file")
uploaded = files.upload()
csv_file = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[csv_file]))
print(f"✅ Uploaded: {csv_file}")

# Drop non-numeric or unwanted columns
df = df.select_dtypes(include=np.number).dropna()

# Split features and target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature columns
joblib.dump(model, "best_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("✅ Model training complete and files saved.")
