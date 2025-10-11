"""
FastAPI Backend for Real-time Credit Risk Prediction
This provides a REST API for real-time predictions and model management.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import uvicorn
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="InsuraSense API",
    description="Real-time Credit Risk Prediction and Customer Churn Analysis API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}
feature_names = []
preprocessor = None
model_metadata = {}

# Pydantic models for API
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    marital_status: str = Field(..., description="Marital status")
    home_market_value: str = Field(..., description="Home market value category")
    annual_income: Optional[float] = Field(None, ge=0, description="Annual income")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "marital_status": "MARRIED",
                "home_market_value": "HIGH",
                "annual_income": 75000.0
            }
        }

class PredictionRequest(BaseModel):
    customers: List[CustomerData]
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "age": 35,
                        "marital_status": "MARRIED",
                        "home_market_value": "HIGH",
                        "annual_income": 75000.0
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    timestamp: str

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_score_mean: float
    cv_score_std: float

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]
    model_name: Optional[str] = Field(None, description="Specific model to use")

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_used: str
    total_customers: int
    processing_time: float
    timestamp: str

# Load models on startup
async def load_models():
    """Load pre-trained models and preprocessor"""
    global models, feature_names, preprocessor, model_metadata
    
    try:
        # Load preprocessor
        if os.path.exists('enhanced_preprocessor.pkl'):
            preprocessor = joblib.load('enhanced_preprocessor.pkl')
            logger.info("Preprocessor loaded successfully")
        else:
            logger.warning("Preprocessor file not found")
        
        # Load feature names
        if os.path.exists('enhanced_feature_names.pkl'):
            feature_names = joblib.load('enhanced_feature_names.pkl')
            logger.info(f"Feature names loaded: {len(feature_names)} features")
        else:
            logger.warning("Feature names file not found")
        
        # Load all models
        if os.path.exists('enhanced_all_models.pkl'):
            models = joblib.load('enhanced_all_models.pkl')
            logger.info(f"Models loaded: {list(models.keys())}")
            
            # Extract model metadata
            for name, model_data in models.items():
                model_metadata[name] = {
                    'metrics': model_data.get('metrics', {}),
                    'cv_scores': model_data.get('cv_scores', []),
                    'best_params': model_data.get('best_params', {}),
                    'model_type': type(model_data['model']).__name__
                }
        else:
            logger.warning("Models file not found")
        
        logger.info("Model loading completed")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting InsuraSense API...")
    await load_models()
    logger.info("API startup completed")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "InsuraSense API - Credit Risk Prediction",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if models else "not_loaded"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "model_status": model_status,
        "preprocessor_loaded": preprocessor is not None
    }

@app.get("/models")
async def get_models():
    """Get information about available models"""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    model_info = {}
    for name, metadata in model_metadata.items():
        model_info[name] = {
            "model_type": metadata['model_type'],
            "metrics": metadata['metrics'],
            "cv_score_mean": float(np.mean(metadata['cv_scores'])) if metadata['cv_scores'] else 0,
            "cv_score_std": float(np.std(metadata['cv_scores'])) if metadata['cv_scores'] else 0,
            "best_params": metadata['best_params']
        }
    
    return {
        "available_models": list(models.keys()),
        "model_details": model_info,
        "feature_count": len(feature_names),
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(request: PredictionRequest):
    """Predict credit risk for a single customer or small batch"""
    if not models or not preprocessor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert request to DataFrame
        customer_data = []
        for customer in request.customers:
            customer_data.append({
                'AGE': customer.age,
                'MARITAL_STATUS': customer.marital_status,
                'HOME_MARKET_VALUE': customer.home_market_value,
                'ANNUAL_INCOME': customer.annual_income if customer.annual_income else np.nan
            })
        
        df = pd.DataFrame(customer_data)
        
        # Preprocess the data
        X_processed = preprocessor.transform(df)
        
        # Get best model
        best_model_name = max(models.keys(), key=lambda k: models[k]['metrics']['roc_auc'])
        best_model = models[best_model_name]['model']
        
        # Make predictions
        predictions_proba = best_model.predict_proba(X_processed)
        predictions_binary = best_model.predict(X_processed)
        
        # Format predictions
        predictions = []
        for i, customer in enumerate(request.customers):
            good_credit_prob = predictions_proba[i][1]
            bad_credit_prob = predictions_proba[i][0]
            
            predictions.append({
                "customer_id": i + 1,
                "customer_data": customer.dict(),
                "prediction": {
                    "good_credit_probability": float(good_credit_prob),
                    "bad_credit_probability": float(bad_credit_prob),
                    "predicted_class": int(predictions_binary[i]),
                    "confidence": float(max(good_credit_prob, bad_credit_prob)),
                    "risk_level": get_risk_level(good_credit_prob)
                }
            })
        
        return PredictionResponse(
            predictions=predictions,
            model_info={
                "model_used": best_model_name,
                "model_metrics": models[best_model_name]['metrics'],
                "feature_count": len(feature_names)
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(bg_tasks: BackgroundTasks, request: BatchPredictionRequest):
    """Predict credit risk for large batches of customers"""
    if not models or not preprocessor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert request to DataFrame
        customer_data = []
        for customer in request.customers:
            customer_data.append({
                'AGE': customer.age,
                'MARITAL_STATUS': customer.marital_status,
                'HOME_MARKET_VALUE': customer.home_market_value,
                'ANNUAL_INCOME': customer.annual_income if customer.annual_income else np.nan
            })
        
        df = pd.DataFrame(customer_data)
        
        # Preprocess the data
        X_processed = preprocessor.transform(df)
        
        # Select model
        if request.model_name and request.model_name in models:
            model_name = request.model_name
        else:
            model_name = max(models.keys(), key=lambda k: models[k]['metrics']['roc_auc'])
        
        model = models[model_name]['model']
        
        # Make predictions
        predictions_proba = model.predict_proba(X_processed)
        predictions_binary = model.predict(X_processed)
        
        # Format predictions
        predictions = []
        for i, customer in enumerate(request.customers):
            good_credit_prob = predictions_proba[i][1]
            bad_credit_prob = predictions_proba[i][0]
            
            predictions.append({
                "customer_id": i + 1,
                "good_credit_probability": float(good_credit_prob),
                "bad_credit_probability": float(bad_credit_prob),
                "predicted_class": int(predictions_binary[i]),
                "confidence": float(max(good_credit_prob, bad_credit_prob)),
                "risk_level": get_risk_level(good_credit_prob)
            })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log batch processing
        bg_tasks.add_task(log_batch_processing, len(request.customers), model_name, processing_time)
        
        return BatchPredictionResponse(
            predictions=predictions,
            model_used=model_name,
            total_customers=len(request.customers),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics")
async def get_model_metrics():
    """Get detailed model performance metrics"""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    metrics_list = []
    for name, model_data in models.items():
        metrics_list.append(ModelMetrics(
            model_name=name,
            accuracy=model_data['metrics']['accuracy'],
            precision=model_data['metrics']['precision'],
            recall=model_data['metrics']['recall'],
            f1_score=model_data['metrics']['f1'],
            roc_auc=model_data['metrics']['roc_auc'],
            cv_score_mean=float(np.mean(model_data['cv_scores'])),
            cv_score_std=float(np.std(model_data['cv_scores']))
        ))
    
    return {
        "models": metrics_list,
        "best_model": max(metrics_list, key=lambda x: x.roc_auc).model_name,
        "total_models": len(metrics_list)
    }

@app.get("/feature_importance/{model_name}")
async def get_feature_importance(model_name: str):
    """Get feature importance for a specific model"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[model_name]['model']
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create feature importance data
        feature_importance = []
        for i, importance in enumerate(importances):
            feature_importance.append({
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "importance": float(importance)
            })
        
        # Sort by importance
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
    """Compare predictions across all models"""
    if not models or not preprocessor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert request to DataFrame
        customer_data = []
        for customer in request.customers:
            customer_data.append({
                'AGE': customer.age,
                'MARITAL_STATUS': customer.marital_status,
                'HOME_MARKET_VALUE': customer.home_market_value,
                'ANNUAL_INCOME': customer.annual_income if customer.annual_income else np.nan
            })
        
        df = pd.DataFrame(customer_data)
        X_processed = preprocessor.transform(df)
        
        # Get predictions from all models
        model_predictions = {}
        for name, model_data in models.items():
            model = model_data['model']
            predictions_proba = model.predict_proba(X_processed)
            
            model_predictions[name] = {
                "predictions": [
                    {
                        "good_credit_probability": float(pred[1]),
                        "bad_credit_probability": float(pred[0]),
                        "confidence": float(max(pred))
                    }
                    for pred in predictions_proba
                ],
                "metrics": model_data['metrics']
            }
        
        return {
            "model_predictions": model_predictions,
            "customer_count": len(request.customers),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

# Utility functions
def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability >= 0.8:
        return "Very High Risk"
    elif probability >= 0.6:
        return "High Risk"
    elif probability >= 0.4:
        return "Medium Risk"
    elif probability >= 0.2:
        return "Low Risk"
    else:
        return "Very Low Risk"

async def log_batch_processing(customer_count: int, model_name: str, processing_time: float):
    """Background task to log batch processing statistics"""
    logger.info(f"Batch processed: {customer_count} customers using {model_name} in {processing_time:.2f}s")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "api_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
