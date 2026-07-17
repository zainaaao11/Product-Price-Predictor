import joblib
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Load the saved data ────
X_train = sp.load_npz("X_train_final.npz")
X_test  = sp.load_npz("X_test_final.npz")
y_train = joblib.load("y_train.pkl")
y_test  = joblib.load("y_test.pkl")

print(f"Data loaded!")
print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

# ─── Now you can train ────
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("started training!")

model.fit(X_train, y_train)

print("Model trained!")

# ─── Make Predictions ────
y_pred_log = model.predict(X_test)

# ─── Convert back from log_price to real MAD price ───────
y_pred_price = np.expm1(y_pred_log)
y_test_price = np.expm1(y_test)

# ─── Calculate Metrics ────────
rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
mae  = mean_absolute_error(y_test_price, y_pred_price)
r2   = r2_score(y_test_price, y_pred_price)

print(f"=== MODEL EVALUATION ===")
print(f"RMSE: {rmse:.2f} MAD")
print(f"MAE:  {mae:.2f} MAD")
print(f"R²:   {r2:.4f}")

# ─── Compare a few predictions vs real prices ─────────────
print(f"\n=== SAMPLE PREDICTIONS ===")
for i in range(20):
    print(f"Real: {y_test_price.iloc[i]:.2f} MAD  |  Predicted: {y_pred_price[i]:.2f} MAD  |  Error: {abs(y_test_price.iloc[i] - y_pred_price[i]):.2f} MAD")

