import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("combined_air_quality_data.csv")
df = df.dropna()

# Select features and target
X = df.drop(columns=['AQI', 'AQI_Bucket'], errors='ignore')
y = df['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature columns
joblib.dump(model, "best_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
