import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Load Data ────
X_train = np.load("X_train_final.npy")
X_test  = np.load("X_test_final.npy")
y_train = joblib.load("y_train.pkl")
y_test  = joblib.load("y_test.pkl")

print(f"Data loaded!")
print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

# ─── Train XGBoost ─────
print(f"\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    tree_method="hist",   # faster training on CPU
    verbosity=1           # shows training progress
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],   # monitors test performance during training
    verbose=50                      # prints progress every 50 trees
)

print(f"XGBoost trained!")

# ─── Evaluate ────
y_pred_log   = model.predict(X_test)
y_pred_price = np.expm1(y_pred_log)
y_test_price = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
mae  = mean_absolute_error(y_test_price, y_pred_price)
r2   = r2_score(y_test_price, y_pred_price)

print(f"\n=== XGBOOST EVALUATION ===")
print(f"RMSE: {rmse:.2f} MAD")
print(f"MAE:  {mae:.2f} MAD")
print(f"R²:   {r2:.4f}")

# ─── Save Model ───
joblib.dump(model, "xgboost_model.pkl")
print(f"\nModel saved!")

# ─── Compare a few predictions vs real prices ────
print(f"\n=== SAMPLE PREDICTIONS ===")
for i in range(20):
    print(f"Real: {y_test_price.iloc[i]:.2f} MAD  |  Predicted: {y_pred_price[i]:.2f} MAD  |  Error: {abs(y_test_price.iloc[i] - y_pred_price[i]):.2f} MAD")

