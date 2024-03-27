import pandas as pd
import xgboost as xgb
import joblib as jb

# import config

# X_test_path = config.X_test_path
# xgb_model_save_path = config.xgb_model_save_path
# predict_outputs_path = config.predict_outputs_path


def evaluate(X_test_path: str, xgb_model_save_path: str, predict_outputs_path: str):
    X_test = pd.read_csv(X_test_path)
    model = jb.load(xgb_model_save_path)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    output_df = X_test[["shop_id", "item_id"]].copy()
    output_df["item_cnt_month"] = y_pred
    output_df.to_csv(predict_outputs_path, index=False)


# if __name__ == "__main__":
#     evaluate()
