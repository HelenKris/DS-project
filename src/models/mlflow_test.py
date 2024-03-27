import config
import mlflow

# import os
# from mlflow.models.signature import infer_signature
import xgboost as xgb
import pandas as pd

mlflow.set_experiment("xgboost")
mlflow.xgboost.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
    X_train = pd.read_csv(config.X_train_path)
    y_train = pd.read_csv(config.y_train_path)
    X_val = pd.read_csv(config.X_val_path)
    y_val = pd.read_csv(config.y_val_path)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.3,
        "subsample": 0.9,
    }

    evals = [(dtrain, "train"), (dval, "eval")]

    bst = xgb.train(
        params, dtrain, evals=evals, num_boost_round=10, early_stopping_rounds=1
    )

    mlflow.xgboost.log_model(bst, "xgboost_model")
