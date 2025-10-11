"""
Ultra-fast dataset processor for any CSV dataset
This module can automatically detect data types, handle missing values,
and generate predictions within seconds for any dataset.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os
import logging
from typing import Tuple, Dict, Any, List, Optional
import time
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastDatasetProcessor:
    """Ultra-fast dataset processor for any CSV data."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_column = None
        self.is_trained = False
        
    def auto_detect_target_column(self, df: pd.DataFrame) -> str:
        """Automatically detect the target column."""
        # Common target column names
        target_candidates = ['target', 'label', 'churn', 'y', 'outcome', 'result', 'class']
        
        for col in target_candidates:
            if col.lower() in [c.lower() for c in df.columns]:
                return col
        
        # If no common target found, use the last column
        return df.columns[-1]
    
    def auto_detect_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect feature types."""
        feature_types = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    feature_types[col] = 'numeric'
                except:
                    # Check unique values for categorical
                    unique_vals = df[col].nunique()
                    if unique_vals <= 20:  # Likely categorical
                        feature_types[col] = 'categorical'
                    else:  # Likely text, convert to numeric
                        feature_types[col] = 'numeric'
            else:
                feature_types[col] = 'numeric'
        
        return feature_types
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess any dataset automatically."""
        logger.info("Starting automatic data preprocessing...")
        start_time = time.time()
        
        # Auto-detect target column if not provided
        if target_col is None:
            target_col = self.auto_detect_target_column(df)
        
        self.target_column = target_col
        logger.info(f"Target column detected: {target_col}")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle target encoding if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            logger.info(f"Target encoded. Classes: {le.classes_}")
        
        # Auto-detect feature types
        feature_types = self.auto_detect_feature_types(X)
        logger.info(f"Feature types detected: {feature_types}")
        
        # Separate numeric and categorical features
        numeric_features = [col for col, ftype in feature_types.items() if ftype == 'numeric']
        categorical_features = [col for col, ftype in feature_types.items() if ftype == 'categorical']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Use OrdinalEncoder for speed
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Create feature names
        feature_names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if trans != 'passthrough':
                feature_names.extend([f"{name}_{col}" for col in cols])
            else:
                feature_names.extend(cols)
        
        self.feature_names = feature_names
        
        processing_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {processing_time:.2f} seconds")
        logger.info(f"Processed {len(X_processed)} samples with {len(feature_names)} features")
        
        return X_processed, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        """Train LightGBM model with optimized parameters."""
        logger.info("Training LightGBM model...")
        start_time = time.time()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # LightGBM parameters optimized for speed
        params = {
            'objective': 'binary' if len(np.unique(y)) == 2 else 'multiclass',
            'metric': 'binary_logloss' if len(np.unique(y)) == 2 else 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,  # Higher learning rate for speed
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'num_class': len(np.unique(y)) if len(np.unique(y)) > 2 else 1
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model with fewer rounds for speed
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=50,  # Reduced for speed
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Evaluate
        y_pred_proba = self.model.predict(X_test)
        if len(np.unique(y)) == 2:
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            metrics = {'accuracy': accuracy, 'roc_auc': roc_auc}
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {'accuracy': accuracy}
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def process_dataset_from_csv(self, csv_content: str, target_col: str = None) -> Dict[str, Any]:
        """Process dataset from CSV content and return results."""
        logger.info("Processing dataset from CSV content...")
        total_start_time = time.time()
        
        try:
            # Parse CSV
            df = pd.read_csv(StringIO(csv_content))
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Preprocess
            X, y = self.preprocess_data(df, target_col)
            
            # Train model
            metrics = self.train_model(X, y)
            
            # Save models
            self.save_models()
            
            total_time = time.time() - total_start_time
            
            return {
                'status': 'success',
                'dataset_shape': df.shape,
                'target_column': self.target_column,
                'feature_count': len(self.feature_names),
                'metrics': metrics,
                'processing_time_seconds': total_time,
                'model_saved': True
            }
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def process_dataset_from_file(self, file_path: str, target_col: str = None) -> Dict[str, Any]:
        """Process dataset from file path."""
        logger.info(f"Processing dataset from file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                csv_content = f.read()
            
            return self.process_dataset_from_csv(csv_content, target_col)
            
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def save_models(self):
        """Save trained models."""
        if self.is_trained:
            os.makedirs('models', exist_ok=True)
            
            # Save model
            joblib.dump(self.model, 'models/fast_model.pkl')
            
            # Save preprocessor
            joblib.dump(self.preprocessor, 'models/fast_preprocessor.pkl')
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'target_column': self.target_column,
                'is_trained': self.is_trained
            }
            joblib.dump(metadata, 'models/metadata.pkl')
            
            logger.info("Models saved successfully")
    
    def load_models(self):
        """Load trained models."""
        try:
            self.model = joblib.load('models/fast_model.pkl')
            self.preprocessor = joblib.load('models/fast_preprocessor.pkl')
            metadata = joblib.load('models/metadata.pkl')
            
            self.feature_names = metadata['feature_names']
            self.target_column = metadata['target_column']
            self.is_trained = metadata['is_trained']
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_from_csv(self, csv_content: str) -> Dict[str, Any]:
        """Make predictions on new CSV data."""
        if not self.is_trained:
            return {'status': 'error', 'error': 'Model not trained'}
        
        try:
            # Parse CSV
            df = pd.read_csv(StringIO(csv_content))
            
            # Preprocess (without target)
            X_processed = self.preprocessor.transform(df)
            
            # Make predictions
            predictions = self.predict(X_processed)
            
            # Add predictions to dataframe
            df['prediction'] = predictions
            df['prediction_prob'] = predictions  # For binary classification
            
            return {
                'status': 'success',
                'predictions': df.to_dict('records'),
                'sample_count': len(df)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# Global processor instance
fast_processor = FastDatasetProcessor()

def process_uploaded_dataset(csv_content: str, target_col: str = None) -> Dict[str, Any]:
    """Main function to process uploaded dataset."""
    return fast_processor.process_dataset_from_csv(csv_content, target_col)

def predict_on_new_data(csv_content: str) -> Dict[str, Any]:
    """Make predictions on new data."""
    return fast_processor.predict_from_csv(csv_content)

if __name__ == "__main__":
    # Test with sample data
    sample_data = """customer_id,age,gender,tenure,balance,products_number,credit_card,active_member,estimated_salary,churn
1,45,1,39,83807.86,1,1,1,119346.88,0
2,39,0,1,93826.63,1,0,1,79084.10,0
3,42,1,1,149348.88,3,1,1,10062.80,1"""
    
    result = process_uploaded_dataset(sample_data)
    print("Processing result:", result)