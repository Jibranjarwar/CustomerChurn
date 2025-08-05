from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
import os
import sys
import logging
import yaml
from typing import List, Dict, Any, Union
import numpy as np
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

sys.path.append('modeling')
from modeling.features import get_features_and_target

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

MODEL_PATH = "models/model.pkl"
CONFIG_PATH = "modeling/config.yaml"

try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    RISK_THRESHOLDS = config.get("api", {}).get("risk_thresholds", {
        "high": 0.7,
        "medium": 0.4
    })
    logger.info(f"Loaded risk thresholds: {RISK_THRESHOLDS}")
except Exception as e:
    logger.warning(f"Using default risk thresholds: {e}")
    RISK_THRESHOLDS = {"high": 0.7, "medium": 0.4}

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded: {type(model).__name__}")
    if hasattr(model, 'n_features_in_'):
        logger.info(f"Model expects {model.n_features_in_} features")
except FileNotFoundError:
    logger.error(f"Model not found at {MODEL_PATH}")
    model = None
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

class CustomerData(BaseModel):
    """Input data schema for customer churn prediction"""
    monthly_spend: float
    spend_change: float
    monthly_spend_last_month: float
    logins_last_30_days: float
    logins_last_60_days: float
    logins_per_day: float
    login_trend: float
    support_tickets: float
    tickets_per_month: float
    tickets_per_tenure: float
    tenure_days: float
    tenure_bucket_less_than_3m: int
    tenure_bucket_3_to_6m: int
    tenure_bucket_6_to_12m: int
    tenure_bucket_1_to_2y: int
    tenure_bucket_2y_plus: int
    
    @validator('monthly_spend')
    def validate_spend(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Spend values cannot be negative')
        return v
    
    @validator('tenure_days')
    def validate_tenure(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Tenure days cannot be negative')
        return v
    
    @validator('support_tickets')
    def validate_tickets(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Support tickets cannot be negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "monthly_spend": 150.0,
                "spend_change": -10.0,
                "monthly_spend_last_month": 160.0,
                "logins_last_30_days": 25.0,
                "logins_last_60_days": 45.0,
                "logins_per_day": 0.83,
                "login_trend": -5.0,
                "support_tickets": 3.0,
                "tickets_per_month": 0.15,
                "tickets_per_tenure": 0.08,
                "tenure_days": 180.0,
                "tenure_bucket_less_than_3m": 0,
                "tenure_bucket_3_to_6m": 1,
                "tenure_bucket_6_to_12m": 0,
                "tenure_bucket_1_to_2y": 0,
                "tenure_bucket_2y_plus": 0
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for churn prediction"""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    confidence_score: float
    risk_level: str

@app.get("/")
async def root() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData, customer_id: str = "unknown") -> PredictionResponse:
    """Predict customer churn probability"""
    start_time = time.time()
    request_id = f"{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Prediction request: {request_id}")
    
    if model is None:
        logger.error(f"Model not loaded for request {request_id}")
        raise HTTPException(status_code=500, detail="Model not loaded. Train the model first.")
    
    try:
        input_dict = customer_data.dict()
        
        column_mapping = {
            'tenure_bucket_less_than_3m': 'tenure_bucket_<3m',
            'tenure_bucket_3_to_6m': 'tenure_bucket_3-6m',
            'tenure_bucket_6_to_12m': 'tenure_bucket_6-12m',
            'tenure_bucket_1_to_2y': 'tenure_bucket_1-2y',
            'tenure_bucket_2y_plus': 'tenure_bucket_2y+'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in input_dict:
                input_dict[new_name] = input_dict.pop(old_name)
        
        df = pd.DataFrame([input_dict])
        X, _ = get_features_and_target(df)
        
        churn_probability = model.predict_proba(X)[0, 1]
        churn_prediction = model.predict(X)[0]
        
        confidence_score = max(churn_probability, 1 - churn_probability)
        
        if churn_probability >= RISK_THRESHOLDS["high"]:
            risk_level = "High"
        elif churn_probability >= RISK_THRESHOLDS["medium"]:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction completed: {customer_id}, prob={churn_probability:.3f}, risk={risk_level}, time={processing_time:.3f}s")
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(churn_probability, 4),
            churn_prediction=bool(churn_prediction),
            confidence_score=round(confidence_score, 4),
            risk_level=risk_level
        )
        
    except ValueError as e:
        logger.error(f"Invalid input for {customer_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except KeyError as e:
        logger.error(f"Missing field for {customer_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction failed for {customer_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(customers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Predict churn for multiple customers"""
    start_time = time.time()
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Batch prediction: {batch_id}, customers={len(customers)}")
    
    if model is None:
        logger.error(f"Model not loaded for batch {batch_id}")
        raise HTTPException(status_code=500, detail="Model not loaded. Train the model first.")
    
    if not customers:
        logger.warning(f"Empty customer list for batch {batch_id}")
        raise HTTPException(status_code=400, detail="Empty customer list provided")
    
    try:
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        for i, customer_data in enumerate(customers):
            customer_id = customer_data.get('customer_id', f"customer_{i}")
            
            if 'customer_id' in customer_data:
                del customer_data['customer_id']
            
            try:
                customer_input = CustomerData(**customer_data)
                prediction = await predict_churn(customer_input, customer_id)
                results.append(prediction.dict())
                successful_predictions += 1
            except Exception as e:
                failed_predictions += 1
                logger.error(f"Failed to process {customer_id} in batch {batch_id}: {str(e)}")
                results.append({
                    "customer_id": customer_id,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch completed: {batch_id}, success={successful_predictions}, failed={failed_predictions}, time={processing_time:.3f}s")
        
        return {
            "predictions": results,
            "total_customers": len(results),
            "successful_predictions": successful_predictions
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model"""
    logger.info("Model info requested")
    
    if model is None:
        logger.warning("Model info requested but no model loaded")
        raise HTTPException(status_code=404, detail="No model loaded")
    
    model_info = {
        "model_type": type(model).__name__,
        "feature_importances": model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
        "model_path": MODEL_PATH,
        "risk_thresholds": RISK_THRESHOLDS
    }
    
    logger.info(f"Model info: {model_info['model_type']}, features={model_info['n_features']}")
    
    return model_info

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Customer Churn Prediction API server")
    uvicorn.run(app, host="0.0.0.0", port=8000) 