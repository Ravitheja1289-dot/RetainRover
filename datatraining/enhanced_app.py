"""
Enhanced FastAPI service for ultra-fast dataset processing
This service can process any CSV dataset and generate predictions within seconds.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import time
import os
import io
from fast_processor import fast_processor, process_uploaded_dataset, predict_on_new_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ultra-Fast Dataset Processor API",
    description="Process any CSV dataset and generate predictions within seconds",
    version="2.0.0"
)

# Request/Response Models
class ProcessDatasetResponse(BaseModel):
    """Response model for dataset processing."""
    status: str
    dataset_shape: Optional[tuple] = None
    target_column: Optional[str] = None
    feature_count: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    processing_time_seconds: Optional[float] = None
    model_saved: Optional[bool] = None
    error: Optional[str] = None

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    status: str
    predictions: Optional[List[Dict]] = None
    sample_count: Optional[int] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: float

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ultra-Fast Dataset Processor API",
        "version": "2.0.0",
        "description": "Process any CSV dataset and generate predictions within seconds",
        "endpoints": {
            "upload_dataset": "POST /upload-dataset - Upload and process CSV dataset",
            "predict": "POST /predict - Make predictions on new data",
            "health": "GET /health - Health check",
            "download_model": "GET /download-model - Download trained model",
            "model_info": "GET /model-info - Get model information"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = fast_processor.is_trained
    return HealthResponse(
        status="healthy" if model_loaded else "no_model_loaded",
        model_loaded=model_loaded,
        timestamp=time.time()
    )

@app.post("/upload-dataset", response_model=ProcessDatasetResponse)
async def upload_and_process_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a CSV dataset.
    This endpoint automatically detects data types, handles missing values,
    and trains a model within seconds.
    """
    logger.info(f"Processing uploaded file: {file.filename}")
    start_time = time.time()
    
    try:
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Process dataset
        result = process_uploaded_dataset(csv_content, target_column)
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Dataset processing completed in {processing_time:.2f} seconds")
        
        return ProcessDatasetResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing uploaded dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_new_data(file: UploadFile = File(...)):
    """
    Make predictions on new CSV data using the trained model.
    """
    if not fast_processor.is_trained:
        raise HTTPException(status_code=400, detail="No model trained. Please upload a dataset first.")
    
    logger.info(f"Making predictions on file: {file.filename}")
    start_time = time.time()
    
    try:
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Make predictions
        result = predict_on_new_data(csv_content)
        
        # Log prediction time
        prediction_time = time.time() - start_time
        logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-text-dataset")
async def process_text_dataset(
    csv_content: str,
    target_column: Optional[str] = None
):
    """
    Process dataset from text content (for direct CSV text input).
    """
    logger.info("Processing dataset from text content")
    start_time = time.time()
    
    try:
        result = process_uploaded_dataset(csv_content, target_column)
        
        processing_time = time.time() - start_time
        logger.info(f"Text dataset processing completed in {processing_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing text dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model."""
    if not fast_processor.is_trained:
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}
    
    return {
        "status": "model_loaded",
        "target_column": fast_processor.target_column,
        "feature_count": len(fast_processor.feature_names) if fast_processor.feature_names else 0,
        "feature_names": fast_processor.feature_names,
        "model_type": "LightGBM",
        "is_trained": fast_processor.is_trained
    }

@app.get("/download-model")
async def download_model():
    """Download the trained model files."""
    if not fast_processor.is_trained:
        raise HTTPException(status_code=404, detail="No trained model found")
    
    # Check if model files exist
    model_path = "models/fast_model.pkl"
    preprocessor_path = "models/fast_preprocessor.pkl"
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        # Create a zip file with both models
        import zipfile
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file, 'w') as zip_file:
                zip_file.write(model_path, 'fast_model.pkl')
                zip_file.write(preprocessor_path, 'fast_preprocessor.pkl')
            
            return FileResponse(
                tmp_file.name,
                media_type='application/zip',
                filename='trained_models.zip'
            )
    else:
        raise HTTPException(status_code=404, detail="Model files not found")

@app.get("/sample-data")
async def get_sample_data():
    """Get sample data for testing."""
    sample_csv = """customer_id,age,gender,tenure,balance,products_number,credit_card,active_member,estimated_salary,churn
1,45,1,39,83807.86,1,1,1,119346.88,0
2,39,0,1,93826.63,1,0,1,79084.10,0
3,42,1,1,149348.88,3,1,1,10062.80,1
4,43,0,40,125510.82,2,1,0,119346.88,0
5,42,1,2,79084.10,1,1,1,149348.88,1"""
    
    return {
        "sample_csv": sample_csv,
        "description": "Sample churn prediction dataset with 5 records",
        "usage": "Copy this CSV content and use it with the /process-text-dataset endpoint"
    }

@app.get("/performance-test")
async def performance_test():
    """Run a performance test with sample data."""
    logger.info("Running performance test...")
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create test dataset
    test_data = {
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.randint(0, 2, n_samples),
        'tenure': np.random.randint(0, 20, n_samples),
        'balance': np.random.lognormal(10, 1, n_samples),
        'products_number': np.random.randint(1, 5, n_samples),
        'credit_card': np.random.randint(0, 2, n_samples),
        'active_member': np.random.randint(0, 2, n_samples),
        'estimated_salary': np.random.lognormal(10.5, 0.8, n_samples),
        'churn': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(test_data)
    csv_content = df.to_csv(index=False)
    
    # Test processing speed
    start_time = time.time()
    result = process_uploaded_dataset(csv_content, 'churn')
    processing_time = time.time() - start_time
    
    return {
        "test_samples": n_samples,
        "processing_time_seconds": processing_time,
        "samples_per_second": n_samples / processing_time if processing_time > 0 else 0,
        "result": result
    }

if __name__ == "__main__":
    import uvicorn
    
    # Try to load existing models on startup
    if fast_processor.load_models():
        logger.info("Existing models loaded successfully")
    else:
        logger.info("No existing models found. Ready to process new datasets.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)