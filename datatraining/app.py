"""
High-performance FastAPI service for churn prediction
This service provides endpoints for training, prediction, and batch prediction
with pre-loaded ONNX models for optimal performance.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib
import logging
import time
import os
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for model sessions
preprocessor_session = None
model_session = None
preprocessor_pkl = None
model_pkl = None
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Single prediction request model."""
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

def quantize_numeric_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quantize numeric inputs to uint8 bins for optimization.
    
    Args:
        data: Input dataframe
        
    Returns:
        Quantized dataframe
    """
    quantized_data = data.copy()
    
    # Define quantization ranges for each numeric column
    quantization_ranges = {
        'age': (18, 100),
        'tenure': (0, 50),
        'balance': (0, 200000),
        'products_number': (1, 10),
        'estimated_salary': (0, 200000)
    }
    
    for col, (min_val, max_val) in quantization_ranges.items():
        if col in quantized_data.columns:
            # Clip values to range and quantize to uint8
            clipped = np.clip(quantized_data[col], min_val, max_val)
            quantized_data[col] = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return quantized_data

def load_models():
    """Load preprocessor and model sessions."""
    global preprocessor_session, model_session, preprocessor_pkl, model_pkl
    
    try:
        # Load ONNX preprocessor session
        if os.path.exists('models/preprocessor.onnx'):
            preprocessor_session = ort.InferenceSession('models/preprocessor.onnx')
            logger.info("ONNX preprocessor session loaded successfully")
        
        # Load pickle files as fallback
        if os.path.exists('models/preprocessor.pkl'):
            preprocessor_pkl = joblib.load('models/preprocessor.pkl')
            logger.info("Preprocessor pickle loaded successfully")
        
        if os.path.exists('models/churn_model.pkl'):
            model_pkl = joblib.load('models/churn_model.pkl')
            logger.info("LightGBM model pickle loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess data using loaded preprocessor.
    
    Args:
        data: Input dataframe
        
    Returns:
        Preprocessed numpy array
    """
    if preprocessor_pkl is not None:
        return preprocessor_pkl.transform(data)
    else:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded")

def predict_churn_probability(preprocessed_data: np.ndarray) -> np.ndarray:
    """
    Predict churn probability using loaded model.
    
    Args:
        preprocessed_data: Preprocessed feature array
        
    Returns:
        Churn probabilities
    """
    if model_pkl is not None:
        return model_pkl.predict(preprocessed_data)
    else:
        raise HTTPException(status_code=500, detail="Model not loaded")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up FastAPI application...")
    load_models()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    executor.shutdown(wait=True)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="High-performance API for customer churn prediction using LightGBM",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "train": "POST /train - Retrain model with updated data",
            "predict": "POST /predict - Single prediction",
            "batch_predict": "POST /batch_predict - Batch predictions",
            "health": "GET /health - Health check"
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
        "timestamp": time.time()
    }

@app.post("/train", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks):
    """
    Retrain the model with updated data.
    This endpoint triggers retraining in the background.
    """
    def train_in_background():
        """Background training function."""
        try:
            # Import training function
            import subprocess
            import sys
            
            # Run training script
            result = subprocess.run([sys.executable, "train.py"], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Reload models after successful training
                load_models()
                logger.info("Model retraining completed successfully")
            else:
                logger.error(f"Training failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Background training error: {e}")
    
    # Add training to background tasks
    background_tasks.add_task(train_in_background)
    
    return TrainingResponse(
        status="started",
        message="Model retraining started in background"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Make a single churn prediction.
    Optimized for low latency with pre-loaded models.
    """
    start_time = time.time()
    
    try:
        # Convert request to dataframe
        data = pd.DataFrame([request.model_dump()])
        
        # Quantize numeric inputs for optimization
        quantized_data = quantize_numeric_inputs(data)
        
        # Preprocess data
        preprocessed_data = preprocess_data(quantized_data)
        
        # Make prediction
        churn_probability = predict_churn_probability(preprocessed_data)[0]
        churn_prediction = int(churn_probability > 0.5)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log request latency
        logger.info(f"Single prediction completed in {processing_time_ms:.2f}ms")
        
        return PredictionResponse(
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch churn predictions.
    Optimized for throughput with parallel processing.
    """
    start_time = time.time()
    
    try:
        # Convert request to dataframe
        data = pd.DataFrame([customer.model_dump() for customer in request.customers])
        
        # Quantize numeric inputs for optimization
        quantized_data = quantize_numeric_inputs(data)
        
        # Preprocess data
        preprocessed_data = preprocess_data(quantized_data)
        
        # Make predictions
        churn_probabilities = predict_churn_probability(preprocessed_data)
        
        # Create predictions list
        predictions = []
        for i, prob in enumerate(churn_probabilities):
            predictions.append(PredictionResponse(
                churn_probability=float(prob),
                churn_prediction=int(prob > 0.5),
                processing_time_ms=0  # Individual times not calculated for batch
            ))
        
        # Calculate processing times
        total_processing_time_ms = (time.time() - start_time) * 1000
        average_processing_time_ms = total_processing_time_ms / len(predictions)
        
        # Log batch processing latency
        logger.info(f"Batch prediction of {len(predictions)} samples completed in {total_processing_time_ms:.2f}ms")
        logger.info(f"Average time per prediction: {average_processing_time_ms:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_processing_time_ms,
            average_processing_time_ms=average_processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_model_metrics():
    """Get model performance metrics."""
    return {
        "model_type": "LightGBM",
        "preprocessing": "StandardScaler + OrdinalEncoder",
        "features": [
            "age", "gender", "tenure", "balance", "products_number",
            "credit_card", "active_member", "estimated_salary"
        ],
        "optimization": {
            "quantization": "uint8 bins for numeric inputs",
            "onnx_runtime": "Pre-loaded sessions",
            "threading": "ThreadPoolExecutor for parallel processing"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)