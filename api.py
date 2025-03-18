from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.sklearn
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictPipeline

# Initialize FastAPI app
app = FastAPI()

# Define expected feature columns
FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Define request schema
class PredictionRequest(BaseModel):
    features: list

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert input to DataFrame with correct column names
        input_data = pd.DataFrame([request.features], columns=FEATURE_COLUMNS)
        
        # Load Prediction Pipeline
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_data)

        # Log inference in MLflow
        with mlflow.start_run():
            mlflow.log_param("input_features", str(request.features))  # Convert list to string
            mlflow.log_metric("prediction", float(prediction[0]))  # Ensure it's a float

        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn filename:app --reload
