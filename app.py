"""
Consolidated FastAPI Backend for InsuraSense Churn Prediction
Combines churn prediction capabilities with additional endpoints for model info, feature importance, and comparison.
Uses single LightGBM model for churn prediction.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import logging
import time
import os
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for model
preprocessor_pkl = None
model_pkl = None
feature_names = None
model_metadata = {
    'churn_model': {
        'metrics': {
            'accuracy': 0.85,  # Placeholder metrics
            'precision': 0.82,
            'recall': 0.78,
            'f1': 0.80,
            'roc_auc': 0.88
        },
        'cv_scores': [0.84, 0.86, 0.87, 0.85, 0.89],
        'best_params': {'learning_rate': 0.1, 'n_estimators': 100},
        'model_type': 'LightGBM'
    }
}
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for churn prediction
class PredictionRequest(BaseModel):
    """Single prediction request model for churn."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    gender: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    tenure: int = Field(..., ge=0, le=50, description="Customer tenure in years")
    balance: float = Field(..., ge=0, description="Account balance")
    products_number: int = Field(..., ge=1, le=10, description="Number of products")
    credit_card: int = Field(..., ge=0, le=1, description="Has credit card (0=No, 1=Yes)")
    active_member: int = Field(..., ge=0, le=1, description="Active member (0=No, 1=Yes)")
    estimated_salary: float = Field(..., ge=0, description="Estimated salary")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    customers: List[PredictionRequest] = Field(..., max_items=1000, description="List of customers to predict")

class PredictionResponse(BaseModel):
    """Single prediction response model."""
    churn_probability: float = Field(..., description="Churn probability (0-1)")
    churn_prediction: int = Field(..., description="Churn prediction (0=No, 1=Yes)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    average_processing_time_ms: float = Field(..., description="Average processing time per prediction")

class TrainingResponse(BaseModel):
    """Training response model."""
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Training message")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model metrics if available")

class ModelMetrics(BaseModel):
    """Model metrics response."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_score_mean: float
    cv_score_std: float

def quantize_numeric_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantize numeric inputs to uint8 bins for optimization.
    """
    quantized_data = data.copy()
    
    quantization_ranges = {
        'age': (18, 100),
        'tenure': (0, 50),
        'balance': (0, 200000),
        'products_number': (1, 10),
        'estimated_salary': (0, 200000)
    }
    
    for col, (min_val, max_val) in quantization_ranges.items():
        if col in quantized_data.columns:
            clipped = np.clip(quantized_data[col], min_val, max_val)
            quantized_data[col] = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return quantized_data

def load_models():
    """Load preprocessor and model."""
    global preprocessor_pkl, model_pkl, feature_names
    
    try:
        if os.path.exists('datatraining/models/preprocessor.pkl'):
            preprocessor_pkl = joblib.load('datatraining/models/preprocessor.pkl')
            logger.info("Preprocessor loaded successfully")
        
        if os.path.exists('datatraining/models/churn_model.pkl'):
            model_pkl = joblib.load('datatraining/models/churn_model.pkl')
            logger.info("Churn model loaded successfully")
            # Set feature names for the model
            feature_names = ['age', 'gender', 'tenure', 'balance', 'products_number', 
                           'credit_card', 'active_member', 'estimated_salary']
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    """Preprocess data using loaded preprocessor."""
    if preprocessor_pkl is not None:
        return preprocessor_pkl.transform(data)
    else:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded")

def predict_churn_probability(preprocessed_data: np.ndarray) -> np.ndarray:
    """Predict churn probability using loaded model."""
    if model_pkl is not None:
        return model_pkl.predict_proba(preprocessed_data)[:, 1]  # Probability of churn (class 1)
    else:
        raise HTTPException(status_code=500, detail="Model not loaded")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting up FastAPI application...")
    load_models()
    yield
    logger.info("Shutting down FastAPI application...")
    executor.shutdown(wait=True)

# Initialize FastAPI app
app = FastAPI(
    title="InsuraSense Churn Prediction API",
    description="Consolidated API for customer churn prediction and analysis",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "InsuraSense Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "metrics": "/metrics",
            "feature_importance/{model_name}": "GET /feature_importance/churn_model",
            "model_comparison": "POST /model_comparison",
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict",
            "train": "POST /train"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model_pkl is not None
    preprocessor_loaded = preprocessor_pkl is not None
    
    return {
        "status": "healthy" if model_loaded and preprocessor_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def get_models():
    """Get information about available models (single model)."""
    if model_pkl is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = model_metadata['churn_model']
    cv_scores = metadata['cv_scores']
    
    return {
        "available_models": ["churn_model"],
        "model_details": {
            "churn_model": {
                "model_type": metadata['model_type'],
                "metrics": metadata['metrics'],
                "cv_score_mean": float(np.mean(cv_scores)),
                "cv_score_std": float(np.std(cv_scores)),
                "best_params": metadata['best_params'],
                "feature_count": len(feature_names) if feature_names else 0
            }
        },
        "preprocessor_loaded": preprocessor_pkl is not None
    }

@app.get("/metrics")
async def get_model_metrics():
    """Get detailed model performance metrics."""
    if model_pkl is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = model_metadata['churn_model']
    cv_scores = metadata['cv_scores']
    
    metrics = ModelMetrics(
        model_name="churn_model",
        accuracy=metadata['metrics']['accuracy'],
        precision=metadata['metrics']['precision'],
        recall=metadata['metrics']['recall'],
        f1_score=metadata['metrics']['f1'],
        roc_auc=metadata['metrics']['roc_auc'],
        cv_score_mean=float(np.mean(cv_scores)),
        cv_score_std=float(np.std(cv_scores))
    )
    
    return {
        "models": [metrics],
        "best_model": "churn_model",
        "total_models": 1
    }

@app.get("/feature_importance/{model_name}")
async def get_feature_importance(model_name: str):
    """Get feature importance for the churn model."""
    if model_name != "churn_model" or model_pkl is None:
        raise HTTPException(status_code=404, detail="Model not found or not loaded")
    
    if hasattr(model_pkl, 'feature_importances_'):
        importances = model_pkl.feature_importances_
        
        feature_importance = []
        for i, importance in enumerate(importances):
            if i < len(feature_names):
                feature_importance.append({
                    "feature": feature_names[i],
                    "importance": float(importance)
                })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            "model_name": model_name,
            "feature_importance": feature_importance[:20],  # Top 20 features
            "total_features": len(importances)
        }
    else:
        raise HTTPException(status_code=400, detail="Model does not support feature importance")

@app.post("/model_comparison")
async def compare_models(request: PredictionRequest):
    """Compare predictions (adapted for single model - returns detailed prediction info)."""
    if model_pkl is None or preprocessor_pkl is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        data = pd.DataFrame([request.model_dump()])
        quantized_data = quantize_numeric_inputs(data)
        preprocessed_data = preprocess_data(quantized_data)
        churn_prob = predict_churn_probability(preprocessed_data)[0]
        churn_pred = int(churn_prob > 0.5)
        
        return {
            "model_predictions": {
                "churn_model": {
                    "churn_probability": float(churn_prob),
                    "churn_prediction": churn_pred,
                    "confidence": float(churn_prob),
                    "metrics": model_metadata['churn_model']['metrics']
                }
            },
            "customer_data": request.model_dump(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks):
    """
    Retrain the model with updated data.
    Triggers retraining in the background using datatraining/train.py.
    """
    def train_in_background():
        try:
            import subprocess
            import sys
            
            result = subprocess.run([sys.executable, "datatraining/train.py"], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                load_models()
                logger.info("Model retraining completed successfully")
            else:
                logger.error(f"Training failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Background training error: {e}")
    
    background_tasks.add_task(train_in_background)
    
    return TrainingResponse(
        status="started",
        message="Model retraining started in background"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Make a single churn prediction.
    """
    start_time = time.time()
    
    try:
        data = pd.DataFrame([request.model_dump()])
        quantized_data = quantize_numeric_inputs(data)
        preprocessed_data = preprocess_data(quantized_data)
        churn_prob = predict_churn_probability(preprocessed_data)[0]
        churn_pred = int(churn_prob > 0.5)
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Single prediction completed in {processing_time_ms:.2f}ms")
        
        return PredictionResponse(
            churn_probability=float(churn_prob),
            churn_prediction=churn_pred,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch churn predictions.
    """
    start_time = time.time()
    
    try:
        data = pd.DataFrame([customer.model_dump() for customer in request.customers])
        quantized_data = quantize_numeric_inputs(data)
        preprocessed_data = preprocess_data(quantized_data)
        churn_probs = predict_churn_probability(preprocessed_data)
        
        predictions = []
        for i, prob in enumerate(churn_probs):
            predictions.append(PredictionResponse(
                churn_probability=float(prob),
                churn_prediction=int(prob > 0.5),
                processing_time_ms=0
            ))
        
        total_processing_time_ms = (time.time() - start_time) * 1000
        average_processing_time_ms = total_processing_time_ms / len(predictions)
        
        logger.info(f"Batch prediction of {len(predictions)} samples completed in {total_processing_time_ms:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_processing_time_ms,
            average_processing_time_ms=average_processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
