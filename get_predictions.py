import config

from src.models.predict_model import predict_model

# src\models\predict_model
X_test_path = config.X_test_path
xgb_model_save_path = config.xgb_model_save_path
predict_outputs_path = config.predict_outputs_path

if __name__ == "__main__":
    predict_model(
        X_test_path=config.X_test_path,
        xgb_model_save_path=config.xgb_model_save_path,
        predict_outputs_path=config.predict_outputs_path,
    )
