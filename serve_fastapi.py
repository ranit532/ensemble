import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from utils.io_utils import latest_joblib

# Define the input data model
class ChurnInput(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    gender: str
    subscription_plan: str

app = FastAPI(
    title="Ensemble Classifier API",
    description="API for predicting customer churn using an ensemble model.",
    version="1.0.0"
)

# Load the latest model at startup
model_path = latest_joblib(os.path.join(os.path.dirname(__file__), 'models'))
model = None
if model_path:
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok'}

@app.get('/model')
def model_info():
    """Returns the name of the currently loaded model."""
    return {'model': os.path.basename(model_path) if model_path else None}

@app.post('/predict')
def predict(data: ChurnInput):
    """
    Makes a churn prediction based on input features.
    
    - **feature_0 to feature_9**: Numerical features.
    - **gender**: 'Male' or 'Female'.
    - **subscription_plan**: 'Basic', 'Premium', or 'Standard'.
    """
    if not model:
        return {"error": "Model not loaded. Please train a model first."}

    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
        confidence = probability[0][prediction[0]]

        return {
            "prediction": churn_status,
            "confidence": f"{confidence:.4f}",
            "model_used": os.path.basename(model_path)
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# To run this API, use the command:
# uvicorn serve_fastapi:app --reload
