import xgboost as xgb
import pandas as pd
import joblib
import sys
sys.path.append('D:/Innowise/DS project')
import config

X_train = pd.read_csv(config.X_train_path)
y_train = pd.read_csv(config.y_train_path)
X_val = pd.read_csv(config.X_val_path)
y_val = pd.read_csv(config.y_val_path)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'reg:squarederror',
    'subsample': 0.9,
    'eta': 0.3,
    'seed': 0
}
# Train the model using xgb.train()
xgb_model = xgb.train(params, dtrain, num_boost_round=100,
                evals=[(dtrain, 'train'),
                (dval, 'eval')],
                early_stopping_rounds=1, verbose_eval=20)

# Save model
joblib.dump(xgb_model, config.xgb_model_save_path)
