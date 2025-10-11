"""
Test script for the ultra-fast dataset processor
This demonstrates processing any dataset within seconds.
"""

import requests
import pandas as pd
import numpy as np
import time
import io

# Test data generation
def generate_test_dataset(n_samples=20000, dataset_type="churn"):
    """Generate test datasets for different scenarios."""
    
    if dataset_type == "churn":
        # Churn prediction dataset
        np.random.seed(42)
        data = {
            'customer_id': range(1, n_samples + 1),
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
        
    elif dataset_type == "sales":
        # Sales prediction dataset
        np.random.seed(42)
        data = {
            'product_id': range(1, n_samples + 1),
            'price': np.random.uniform(10, 500, n_samples),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
            'advertising_budget': np.random.uniform(1000, 10000, n_samples),
            'competitor_price': np.random.uniform(8, 600, n_samples),
            'sales': np.random.uniform(0, 1000, n_samples)
        }
        
    elif dataset_type == "classification":
        # Multi-class classification dataset
        np.random.seed(42)
        data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'feature_5': np.random.normal(0, 1, n_samples),
            'class': np.random.choice(['A', 'B', 'C'], n_samples)
        }
    
    return pd.DataFrame(data)

def test_local_processing():
    """Test the fast processor locally."""
    print("🧪 Testing Local Fast Processor")
    print("=" * 50)
    
    from fast_processor import FastDatasetProcessor
    
    processor = FastDatasetProcessor()
    
    # Test different dataset types
    test_datasets = [
        ("Churn Dataset", "churn"),
        ("Sales Dataset", "sales"), 
        ("Classification Dataset", "classification")
    ]
    
    for name, dataset_type in test_datasets:
        print(f"\n📊 Testing {name}")
        
        # Generate test data
        df = generate_test_dataset(500, dataset_type)
        csv_content = df.to_csv(index=False)
        
        # Process dataset
        start_time = time.time()
        result = processor.process_dataset_from_csv(csv_content)
        processing_time = time.time() - start_time
        
        print(f"  ✅ Processed {df.shape[0]} samples in {processing_time:.2f} seconds")
        print(f"  📈 Accuracy: {result.get('metrics', {}).get('accuracy', 'N/A')}")
        print(f"  🎯 Target: {result.get('target_column', 'N/A')}")

def test_api_endpoints():
    """Test the API endpoints."""
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health Check: {response.status_code}")
    except:
        print("❌ API not running. Start with: python enhanced_app.py")
        return
    
    # Test sample data endpoint
    try:
        response = requests.get(f"{base_url}/sample-data")
        print(f"✅ Sample Data: {response.status_code}")
    except Exception as e:
        print(f"❌ Sample Data Error: {e}")
    
    # Test dataset processing
    try:
        df = generate_test_dataset(100, "churn")
        csv_content = df.to_csv(index=False)
        
        response = requests.post(
            f"{base_url}/process-text-dataset",
            data={"csv_content": csv_content}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Dataset Processing: {result['status']}")
            print(f"  📊 Processed {result.get('dataset_shape', [0, 0])[0]} samples")
            print(f"  ⏱️ Time: {result.get('processing_time_seconds', 0):.2f}s")
        else:
            print(f"❌ Dataset Processing Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Dataset Processing Error: {e}")

def benchmark_performance():
    """Benchmark processing performance."""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    from fast_processor import FastDatasetProcessor
    
    processor = FastDatasetProcessor()
    
    # Test different dataset sizes
    sizes = [100, 500, 1000, 2000, 5000]
    
    for size in sizes:
        print(f"\n📊 Testing with {size} samples:")
        
        df = generate_test_dataset(size, "churn")
        csv_content = df.to_csv(index=False)
        
        start_time = time.time()
        result = processor.process_dataset_from_csv(csv_content)
        processing_time = time.time() - start_time
        
        samples_per_second = size / processing_time if processing_time > 0 else 0
        
        print(f"  ⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"  🚀 Speed: {samples_per_second:.0f} samples/second")
        print(f"  📈 Accuracy: {result.get('metrics', {}).get('accuracy', 'N/A')}")

def main():
    """Run all tests."""
    print("🚀 Ultra-Fast Dataset Processor - Test Suite")
    print("=" * 60)
    
    # Test local processing
    test_local_processing()
    
    # Test API endpoints
    test_api_endpoints()
    
    # Benchmark performance
    benchmark_performance()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\n📋 Usage Instructions:")
    print("1. Local processing: from fast_processor import FastDatasetProcessor")
    print("2. API processing: POST to /process-text-dataset with CSV content")
    print("3. File upload: POST to /upload-dataset with CSV file")
    print("4. Predictions: POST to /predict with new CSV data")

if __name__ == "__main__":
    main()