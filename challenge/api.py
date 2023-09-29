import fastapi
from pydantic import BaseModel
from typing import List
import pandas as pd

from challenge import model

class Flight(BaseModel):
    OPERA: str 
    TIPOVUELO: str
    MES: int

class PredictionRequest(BaseModel):
    flights: List[Flight]

class PredictionResponse(BaseModel):
    predict: List[int]

def create_app(model_filepath: str) -> fastapi.FastAPI:

    # Load serialized model and create FastAPI app
    delay_model = model.DelayModel.load(model_filepath)
    app = fastapi.FastAPI()
        
    # Define API endpoints
    @app.get("/health", status_code=200)
    async def get_health() -> dict:
        return {
            "status": "OK"
        }

    @app.post("/predict", status_code=200) 
    async def post_predict(prediction_request: PredictionRequest) -> PredictionResponse:

        # Get request body
        flight_operators = [flight.OPERA for flight in prediction_request.flights]
        flight_month = [flight.MES for flight in prediction_request.flights]
        flight_type = [flight.TIPOVUELO for flight in prediction_request.flights]
        features = pd.DataFrame({
            "OPERA": flight_operators,
            "TIPOVUELO": flight_type,
            "MES": flight_month
        })

        # Preprocess data, raise exception if invalid
        try:
            features = delay_model.preprocess(features)
        except ValueError as e:
            print(str(e))
            raise fastapi.HTTPException(status_code=400, detail=str(e))
        
        predictions = delay_model.predict(features)
        return PredictionResponse(predict=predictions)  
    
    return app


# Read model filepath from environment variable
import os

# get filepath or use defaulr
model_filepath = os.getenv("MODEL_FILEPATH", "./tmp/delay_model.pkl")
app = create_app(model_filepath)