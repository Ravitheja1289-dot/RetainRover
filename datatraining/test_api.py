"""
Test script for the Churn Prediction API
This script demonstrates how to interact with the API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_single_prediction():
    """Test single prediction endpoint."""
    print("\nTesting single prediction...")
    
    # Sample customer data
    customer_data = {
        "age": 45,
        "gender": 1,
        "tenure": 39,
        "balance": 83807.86,
        "products_number": 1,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 119346.88
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=customer_data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction successful")
            print(f"Churn Probability: {result['churn_probability']:.4f}")
            print(f"Churn Prediction: {result['churn_prediction']}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            print(f"Total Request Time: {(end_time - start_time) * 1000:.2f}ms")
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Single prediction error: {e}")

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\nTesting batch prediction...")
    
    # Sample batch data
    batch_data = {
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
            },
            {
                "age": 42,
                "gender": 1,
                "tenure": 1,
                "balance": 149348.88,
                "products_number": 3,
                "credit_card": 1,
                "active_member": 1,
                "estimated_salary": 10062.80
            }
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/batch_predict", json=batch_data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful")
            print(f"Number of predictions: {len(result['predictions'])}")
            print(f"Total processing time: {result['total_processing_time_ms']:.2f}ms")
            print(f"Average processing time: {result['average_processing_time_ms']:.2f}ms")
            print(f"Total request time: {(end_time - start_time) * 1000:.2f}ms")
            
            # Print individual predictions
            for i, pred in enumerate(result['predictions']):
                print(f"  Customer {i+1}: Prob={pred['churn_probability']:.4f}, "
                      f"Pred={pred['churn_prediction']}")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")

def test_model_metrics():
    """Test model metrics endpoint."""
    print("\nTesting model metrics...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            result = response.json()
            print("✅ Model metrics retrieved")
            print(f"Model type: {result['model_type']}")
            print(f"Features: {result['features']}")
            print(f"Optimizations: {result['optimization']}")
        else:
            print(f"❌ Model metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model metrics error: {e}")

def performance_test():
    """Run a simple performance test."""
    print("\nRunning performance test...")
    
    customer_data = {
        "age": 45,
        "gender": 1,
        "tenure": 39,
        "balance": 83807.86,
        "products_number": 1,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 119346.88
    }
    
    num_requests = 10
    total_time = 0
    successful_requests = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/predict", json=customer_data)
            end_time = time.time()
            
            if response.status_code == 200:
                total_time += (end_time - start_time) * 1000
                successful_requests += 1
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if successful_requests > 0:
        avg_time = total_time / successful_requests
        print(f"✅ Performance test completed")
        print(f"Successful requests: {successful_requests}/{num_requests}")
        print(f"Average response time: {avg_time:.2f}ms")
        print(f"Requests per second: {1000/avg_time:.2f}")
    else:
        print("❌ Performance test failed - no successful requests")

def main():
    """Run all tests."""
    print("🧪 Churn Prediction API Test Suite")
    print("=" * 50)
    
    # Wait a moment for the server to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    test_health_check()
    test_single_prediction()
    test_batch_prediction()
    test_model_metrics()
    performance_test()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()