import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import config

app = FastAPI()
predict_outputs_path = config.predict_outputs_path


class InputData(BaseModel):
    shop_id: int
    item_id: int


@app.post("/predict")
async def predict(data: InputData):
    shop_id = data.shop_id
    item_id = data.item_id
    try:
        predictions_df = pd.read_csv(predict_outputs_path)
        prediction_row = predictions_df[
            (predictions_df["shop_id"] == shop_id)
            & (predictions_df["item_id"] == item_id)
        ]
        if prediction_row.empty:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found for the provided shop_id and item_id.",
            )
        item_cnt_month = prediction_row["item_cnt_month"].iloc[0]
        return {"Your prediction is": item_cnt_month}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Predictions file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
