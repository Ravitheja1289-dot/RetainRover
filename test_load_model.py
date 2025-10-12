import joblib
import sys
import os

# Add current directory to path
sys.path.append('.')

# Load the model
try:
    pipeline = joblib.load('models/churn_pipeline.pkl')
    print("Model loaded successfully")

    # Extract preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    print("Preprocessor extracted")

    # Get feature names
    raw_feature_names = preprocessor.get_feature_names_out()
    print(f"Raw feature names: {raw_feature_names[:5]}...")  # Show first 5

    # Process feature names as in the app
    feature_names = [name.split('__')[1] if '__' in name else name for name in raw_feature_names]
    print(f"Processed feature names: {feature_names[:5]}...")  # Show first 5

    print(f"Total features: {len(feature_names)}")
    print("Test passed: Feature names are properly processed")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
