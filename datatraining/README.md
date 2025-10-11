# End-to-End Churn Model Training + High-Performance API

A comprehensive Python project for customer churn prediction with LightGBM and FastAPI, featuring ONNX optimization for high-performance inference.

## 🚀 Features

- **Data Preprocessing**: Automatic imputation, encoding, and scaling
- **LightGBM Training**: Optimized binary classification with early stopping
- **ONNX Export**: High-performance model serving with ONNX Runtime
- **FastAPI Service**: RESTful API with single and batch prediction endpoints
- **Docker Support**: Containerized deployment with health checks
- **Performance Optimization**: 
  - Pre-loaded ONNX sessions
  - Numeric input quantization to uint8
  - Parallel processing for batch predictions
  - Request latency logging

## 📁 Project Structure

```
├── data/
│   └── churn_data.csv          # Training dataset
├── models/
│   ├── churn_model.pkl         # Trained LightGBM model
│   ├── preprocessor.pkl        # Fitted preprocessing pipeline
│   └── preprocessor.onnx       # ONNX preprocessor
├── train.py                    # Training script
├── app.py                      # FastAPI application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── README.md                   # This file
```

## 🛠️ Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd churn-prediction-api
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t churn-prediction-api .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 churn-prediction-api
   ```

## 🎯 Usage

### 1. Training the Model

Train the LightGBM model with your data:

```bash
python train.py
```

This will:
- Load data from `data/churn_data.csv`
- Preprocess features (imputation, encoding, scaling)
- Train LightGBM classifier with 80/20 train/test split
- Export model and preprocessor to ONNX format
- Log accuracy and ROC-AUC metrics

### 2. Running the API

Start the FastAPI service:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 45,
       "gender": 1,
       "tenure": 39,
       "balance": 83807.86,
       "products_number": 1,
       "credit_card": 1,
       "active_member": 1,
       "estimated_salary": 119346.88
     }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customers": [
         {
           "age": 45,
           "gender": 1,
           "tenure": 39,
           "balance": 83807.86,
           "products_number": 1,
           "credit_card": 1,
           "active_member": 1,
           "estimated_salary": 119346.88
         },
         {
           "age": 39,
           "gender": 0,
           "tenure": 1,
           "balance": 93826.63,
           "products_number": 1,
           "credit_card": 0,
           "active_member": 1,
           "estimated_salary": 79084.10
         }
       ]
     }'
```

#### Retrain Model
```bash
curl -X POST "http://localhost:8000/train"
```

## 📊 Data Format

The training data should be in CSV format with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| customer_id | int | Unique customer identifier |
| age | int | Customer age (18-100) |
| gender | int | Gender (0=Female, 1=Male) |
| tenure | int | Customer tenure in years (0-50) |
| balance | float | Account balance |
| products_number | int | Number of products (1-10) |
| credit_card | int | Has credit card (0=No, 1=Yes) |
| active_member | int | Active member (0=No, 1=Yes) |
| estimated_salary | float | Estimated salary |
| churn | int | Target variable (0=No, 1=Yes) |

## ⚡ Performance Features

### ONNX Optimization
- Pre-loaded ONNX Runtime sessions for minimal latency
- Quantized numeric inputs to uint8 bins
- Optimized inference pipeline

### FastAPI Features
- Async/await support for high concurrency
- Background task processing for model retraining
- Comprehensive request/response validation with Pydantic
- Automatic API documentation at `/docs`

### Docker Optimization
- Multi-stage builds for smaller image size
- Health checks for container monitoring
- Optimized Python settings for performance

## 🔧 Configuration

### Environment Variables
- `PYTHONUNBUFFERED=1`: Enable real-time logging
- `PYTHONDONTWRITEBYTECODE=1`: Disable .pyc files
- `OMP_NUM_THREADS=1`: Optimize for single-threaded inference

### Model Parameters
LightGBM parameters are optimized for binary classification:
- `num_leaves`: 31
- `learning_rate`: 0.05
- `feature_fraction`: 0.9
- `bagging_fraction`: 0.8
- Early stopping with 10 rounds patience

## 📈 Monitoring

The API includes comprehensive logging:
- Request latency for each prediction
- Model loading status
- Training progress and metrics
- Error handling and debugging information

## 🧪 Testing

Test the API endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Get API documentation
open http://localhost:8000/docs

# Test single prediction
python -c "
import requests
import json

data = {
    'age': 45,
    'gender': 1,
    'tenure': 39,
    'balance': 83807.86,
    'products_number': 1,
    'credit_card': 1,
    'active_member': 1,
    'estimated_salary': 119346.88
}

response = requests.post('http://localhost:8000/predict', json=data)
print(json.dumps(response.json(), indent=2))
"
```

## 🚀 Deployment

### Production Deployment

For production deployment, consider:

1. **Load Balancing**: Use multiple FastAPI instances behind a load balancer
2. **Model Versioning**: Implement model version management
3. **Monitoring**: Add Prometheus metrics and Grafana dashboards
4. **Security**: Implement authentication and rate limiting
5. **Scaling**: Use Kubernetes for auto-scaling

### Docker Compose

Create a `docker-compose.yml` for easy deployment:

```yaml
version: '3.8'
services:
  churn-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.