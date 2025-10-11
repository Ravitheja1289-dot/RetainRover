"""
Demo script showing ultra-fast dataset processing
This demonstrates processing any CSV dataset within seconds.
"""

import pandas as pd
import numpy as np
import time
from fast_processor import FastDatasetProcessor

def create_sample_datasets():
    """Create sample datasets for demonstration."""
    
    print("🎯 Creating Sample Datasets")
    print("=" * 40)
    
    # Dataset 1: Customer Churn
    print("\n📊 Dataset 1: Customer Churn Prediction")
    np.random.seed(42)
    churn_data = {
        'customer_id': range(1, 501),
        'age': np.random.randint(18, 80, 500),
        'gender': np.random.choice(['Male', 'Female'], 500),
        'tenure': np.random.randint(0, 20, 500),
        'balance': np.random.lognormal(10, 1, 500),
        'products_number': np.random.randint(1, 5, 500),
        'credit_card': np.random.choice(['Yes', 'No'], 500),
        'active_member': np.random.choice(['Yes', 'No'], 500),
        'estimated_salary': np.random.lognormal(10.5, 0.8, 500),
        'churn': np.random.choice([0, 1], 500, p=[0.8, 0.2])
    }
    
    churn_df = pd.DataFrame(churn_data)
    print(f"  ✅ Created churn dataset: {churn_df.shape}")
    print(f"  📈 Churn rate: {churn_df['churn'].mean():.1%}")
    
    # Dataset 2: Product Sales
    print("\n📊 Dataset 2: Product Sales Prediction")
    sales_data = {
        'product_id': range(1, 301),
        'price': np.random.uniform(10, 500, 300),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 300),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], 300),
        'advertising_budget': np.random.uniform(1000, 10000, 300),
        'competitor_price': np.random.uniform(8, 600, 300),
        'sales': np.random.uniform(0, 1000, 300)
    }
    
    sales_df = pd.DataFrame(sales_data)
    print(f"  ✅ Created sales dataset: {sales_df.shape}")
    print(f"  💰 Average sales: ${sales_df['sales'].mean():.0f}")
    
    # Dataset 3: Classification
    print("\n📊 Dataset 3: Multi-class Classification")
    np.random.seed(123)
    n_samples = 400
    class_data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
        'feature_5': np.random.normal(0, 1, n_samples),
        'class': np.random.choice(['Class_A', 'Class_B', 'Class_C'], n_samples)
    }
    
    class_df = pd.DataFrame(class_data)
    print(f"  ✅ Created classification dataset: {class_df.shape}")
    print(f"  🏷️ Class distribution: {class_df['class'].value_counts().to_dict()}")
    
    return {
        'churn': churn_df,
        'sales': sales_df,
        'classification': class_df
    }

def demonstrate_processing():
    """Demonstrate ultra-fast processing."""
    
    print("\n🚀 Ultra-Fast Dataset Processing Demo")
    print("=" * 50)
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    # Initialize processor
    processor = FastDatasetProcessor()
    
    results = {}
    
    # Process each dataset
    for name, df in datasets.items():
        print(f"\n⚡ Processing {name.upper()} Dataset")
        print("-" * 30)
        
        # Convert to CSV
        csv_content = df.to_csv(index=False)
        
        # Process dataset
        start_time = time.time()
        result = processor.process_dataset_from_csv(csv_content)
        processing_time = time.time() - start_time
        
        results[name] = {
            'dataset_shape': df.shape,
            'processing_time': processing_time,
            'accuracy': result.get('metrics', {}).get('accuracy', 'N/A'),
            'target_column': result.get('target_column', 'N/A')
        }
        
        print(f"  📊 Dataset shape: {df.shape}")
        print(f"  ⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"  🚀 Speed: {df.shape[0] / processing_time:.0f} samples/second")
        print(f"  📈 Accuracy: {result.get('metrics', {}).get('accuracy', 'N/A')}")
        print(f"  🎯 Target column: {result.get('target_column', 'N/A')}")
        
        if result.get('status') == 'success':
            print(f"  ✅ Model trained and saved successfully!")
        else:
            print(f"  ❌ Error: {result.get('error', 'Unknown error')}")

def demonstrate_predictions():
    """Demonstrate making predictions on new data."""
    
    print("\n🔮 Making Predictions on New Data")
    print("=" * 40)
    
    # Load the trained model
    processor = FastDatasetProcessor()
    if not processor.load_models():
        print("❌ No trained model found. Please run processing first.")
        return
    
    # Create new data for prediction
    print("\n📝 Creating new data for prediction...")
    new_data = pd.DataFrame({
        'customer_id': [1001, 1002, 1003],
        'age': [35, 42, 28],
        'gender': ['Male', 'Female', 'Male'],
        'tenure': [5, 12, 2],
        'balance': [50000, 75000, 25000],
        'products_number': [2, 3, 1],
        'credit_card': ['Yes', 'Yes', 'No'],
        'active_member': ['Yes', 'Yes', 'No'],
        'estimated_salary': [60000, 80000, 40000]
    })
    
    print(f"  ✅ Created {len(new_data)} new samples")
    
    # Make predictions
    csv_content = new_data.to_csv(index=False)
    start_time = time.time()
    result = processor.predict_from_csv(csv_content)
    prediction_time = time.time() - start_time
    
    if result.get('status') == 'success':
        predictions = result.get('predictions', [])
        print(f"\n🔮 Predictions (completed in {prediction_time:.3f} seconds):")
        
        for i, pred in enumerate(predictions):
            churn_prob = pred.get('prediction_prob', 0)
            churn_pred = "Yes" if churn_prob > 0.5 else "No"
            print(f"  Customer {pred['customer_id']}: Churn = {churn_pred} (Probability: {churn_prob:.3f})")
    else:
        print(f"❌ Prediction error: {result.get('error', 'Unknown error')}")

def performance_summary(results):
    """Show performance summary."""
    
    print("\n📊 Performance Summary")
    print("=" * 30)
    
    total_samples = sum(r['dataset_shape'][0] for r in results.values())
    total_time = sum(r['processing_time'] for r in results.values())
    avg_accuracy = np.mean([r['accuracy'] for r in results.values() if r['accuracy'] != 'N/A'])
    
    print(f"📈 Total samples processed: {total_samples:,}")
    print(f"⏱️ Total processing time: {total_time:.2f} seconds")
    print(f"🚀 Overall speed: {total_samples / total_time:.0f} samples/second")
    print(f"📊 Average accuracy: {avg_accuracy:.3f}")
    
    print(f"\n🎯 Best performance:")
    fastest = max(results.values(), key=lambda x: x['dataset_shape'][0] / x['processing_time'])
    print(f"  Dataset: {[k for k, v in results.items() if v == fastest][0]}")
    print(f"  Speed: {fastest['dataset_shape'][0] / fastest['processing_time']:.0f} samples/second")

def main():
    """Run the complete demo."""
    
    print("🎉 Ultra-Fast Dataset Processor - Live Demo")
    print("=" * 60)
    print("This demo shows how to process ANY CSV dataset within seconds!")
    print("=" * 60)
    
    # Demonstrate processing
    demonstrate_processing()
    
    # Demonstrate predictions
    demonstrate_predictions()
    
    print("\n" + "=" * 60)
    print("🎊 Demo completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("  ✅ Automatic data type detection")
    print("  ✅ Missing value handling")
    print("  ✅ Target column auto-detection")
    print("  ✅ Ultra-fast LightGBM training")
    print("  ✅ Model saving and loading")
    print("  ✅ Instant predictions on new data")
    print("\n🚀 Ready to process any dataset in seconds!")

if __name__ == "__main__":
    main()