import xgboost as xgb
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append('D:/Innowise/DS project')
import config

X_train = pd.read_csv(config.X_train_path)
y_train = pd.read_csv(config.y_train_path)
X_val = pd.read_csv(config.X_val_path)
y_val = pd.read_csv(config.y_val_path)
best_grid_model_save_path= config.best_grid_model_save_path

# Define a grid of parameters for selection
xgb_pars = {
    'eta': [ 0.1, 0.3],
    'max_depth': [3, 6, 9],
    'subsample': [0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'objective': ['reg:squarederror']
}

# Convert the data to xgb.DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Use GridSearchCV to select the best hyperparameters
xgb_model = xgb.XGBRegressor()
grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_pars, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid.best_params_
print("Best Parameters:", best_params)

# Save model
best_model = grid.best_estimator_
joblib.dump(best_model, best_grid_model_save_path)
