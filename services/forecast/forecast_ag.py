from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pandas as pd
from autogluon.tabular import TabularPredictor
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "ag_forecast_model"))

# Global variable to hold the model in memory
predictor = None

def get_model_path():
    """Find valid model directory across possible paths."""
    candidate_paths = [
        MODEL_DIR,
        os.path.join(BASE_DIR, "ag_forecast_model"),
        os.path.join(os.getcwd(), "ag_forecast_model"),
        "/app/ag_forecast_model",
    ]
    for p in candidate_paths:
        if os.path.exists(p) and (os.path.exists(os.path.join(p, "learner.pkl")) or os.path.exists(os.path.join(p, "predictor.pkl"))):
            return p
    return None

def load_model():
    """Load the model into memory."""
    global predictor
    model_path = get_model_path()
    if model_path:
        print(f"Loading AutoGluon model from {model_path} into memory...")
        predictor = TabularPredictor.load(model_path)
        print("Model loaded successfully.")
        return True
    else:
        print(f"Warning: Model directory not found at {MODEL_DIR}. Please run training or mount the model directory.")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="AutoGluon Forecast API Service", lifespan=lifespan)

# Note: We do not need the 'ID' or target column here
class ForecastInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int

@app.post("/forecast")
def get_forecast(data: ForecastInput):
    global predictor
    
    if predictor is None:
        # Try lazy loading if model was trained or mounted after startup
        if not load_model() or predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Forecasting service: model not found in ag_forecast_model. Please train the model or mount the model volume."
            )
    
    try:
        # Convert input JSON to DataFrame
        input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        df_input = pd.DataFrame([input_data])
        
        # Predict class (0 or 1)
        prediction = predictor.predict(df_input)
        
        # Predict probability
        # AutoGluon returns a DataFrame with probabilities for class 0 and class 1
        probability = predictor.predict_proba(df_input)
        
        return {
            "status": "success",
            "prediction": int(prediction.iloc[0]),
            # Get the probability for class '1' (default/fraud)
            "probability_of_default": float(probability.iloc[0][1])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    load_model()
    
    # Run test predictions on sample records
    test_cases = [
        ForecastInput(
            LIMIT_BAL=20000,
            SEX=2,
            EDUCATION=2,
            MARRIAGE=1,
            AGE=24,
            PAY_0=2,
            PAY_2=2,
            PAY_3=-1,
            PAY_4=-1,
            PAY_5=-2,
            PAY_6=-2,
            BILL_AMT1=3913,
            BILL_AMT2=3102,
            BILL_AMT3=689,
            BILL_AMT4=0,
            BILL_AMT5=0,
            BILL_AMT6=0,
            PAY_AMT1=0,
            PAY_AMT2=689,
            PAY_AMT3=0,
            PAY_AMT4=0,
            PAY_AMT5=0,
            PAY_AMT6=0,
        ),
        ForecastInput(
            LIMIT_BAL=90000,
            SEX=2,
            EDUCATION=2,
            MARRIAGE=2,
            AGE=34,
            PAY_0=0,
            PAY_2=0,
            PAY_3=0,
            PAY_4=0,
            PAY_5=0,
            PAY_6=0,
            BILL_AMT1=29239,
            BILL_AMT2=14027,
            BILL_AMT3=13559,
            BILL_AMT4=14331,
            BILL_AMT5=14948,
            BILL_AMT6=15549,
            PAY_AMT1=1518,
            PAY_AMT2=1500,
            PAY_AMT3=1000,
            PAY_AMT4=1000,
            PAY_AMT5=1000,
            PAY_AMT6=5000,
        )
    ]
    
    print("\n--- Running Test Inferences ---")
    for i, test_input in enumerate(test_cases, start=1):
        forecast_result = get_forecast(test_input)
        print(f"Sample {i} Result: {forecast_result}")
