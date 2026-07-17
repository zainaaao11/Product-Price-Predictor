from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import numpy as np
import joblib

X_train = np.load("X_train_final.npy")
X_test  = np.load("X_test_final.npy")
y_train = joblib.load("y_train.pkl")
y_test  = joblib.load("y_test.pkl")

param_dist = {
    "n_estimators":     [200, 300, 500],
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [4, 6, 8],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42, tree_method="hist"),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring="neg_mean_absolute_error", # the negative sign because sickit learn always assumes that a maximum value is better.However,a smaller MAE is better so taking the neg will turn the smallest MAE into the highest
    random_state=42,
    verbose=2
)

search.fit(X_train, y_train)

print(f"\nBest parameters found:")
print(search.best_params_)
print(f"\nBest cross-validation MAE: {-search.best_score_:.2f}")