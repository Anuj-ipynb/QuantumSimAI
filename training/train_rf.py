"""
Train a Random Forest baseline for quantum noise prediction.
Version-safe and reproducible.
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/raw/noise_dataset.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "rf_surrogate.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Run generate_data.py first."
    )

df = pd.read_csv(DATA_PATH)

X = df[["n_qubits", "depth", "p1", "p2"]]
y = df["distribution_fidelity"]

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation (version-safe)
# -----------------------------
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, MODEL_PATH)

print("Random Forest training complete")
print(f" RMSE: {rmse:.6f}")
print(f" Model saved to: {MODEL_PATH.resolve()}")
