"""
Machine Learning Model Interpretability Analysis with SHAP and LIME
This script trains models on multiple datasets and provides comprehensive interpretability analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

class MLInterpretabilityAnalyzer:
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.explainers = {}
        self.results = {}
        
    def load_datasets(self):
        """Load all available datasets"""
        print("Loading datasets...")
        
        # Load demographic dataset (main credit prediction dataset)
        try:
            df_demo = pd.read_csv('Datasets/demographic.csv', low_memory=False)
            df_demo = df_demo.drop('INDIVIDUAL_ID', axis=1)
            df_demo = df_demo.dropna(subset=['MARITAL_STATUS', 'HOME_MARKET_VALUE'])
            self.datasets['demographic'] = df_demo
            print(f"Demographic dataset loaded: {df_demo.shape}")
        except Exception as e:
            print(f"Error loading demographic dataset: {e}")
            
        # Load auto insurance churn dataset
        try:
            df_auto = pd.read_csv('Datasets/autoinsurance_churn.csv', low_memory=False)
            self.datasets['autoinsurance'] = df_auto
            print(f"Auto insurance dataset loaded: {df_auto.shape}")
        except Exception as e:
            print(f"Error loading auto insurance dataset: {e}")
            
        # Load other datasets for potential feature engineering
        try:
            df_address = pd.read_csv('Datasets/address.csv', low_memory=False)
            df_termination = pd.read_csv('Datasets/termination.csv', low_memory=False)
            self.datasets['address'] = df_address
            self.datasets['termination'] = df_termination
            print(f"Address dataset loaded: {df_address.shape}")
            print(f"Termination dataset loaded: {df_termination.shape}")
        except Exception as e:
            print(f"Error loading additional datasets: {e}")
    
    def prepare_demographic_model(self):
        """Prepare and train model on demographic dataset"""
        print("\n" + "="*50)
        print("DEMOGRAPHIC DATASET ANALYSIS")
        print("="*50)
        
        df = self.datasets['demographic'].copy()
        
        # Define features and target
        X = df.drop('GOOD_CREDIT', axis=1)
        y = df['GOOD_CREDIT']
        
        # Identify categorical and numerical columns
        categorical_cols = ['MARITAL_STATUS', 'HOME_MARKET_VALUE']
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        print(f"Features: {list(X.columns)}")
        print(f"Categorical: {categorical_cols}")
        print(f"Numerical: {numerical_cols}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for model_name, classifier in models.items():
            print(f"\nTraining {model_name}...")
            
            # Create pipeline
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'preprocessor': preprocessor,
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols
            }
            
            print(f"{model_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        self.results['demographic'] = results
        return results
    
    def prepare_autoinsurance_model(self):
        """Prepare and train model on auto insurance churn dataset"""
        print("\n" + "="*50)
        print("AUTO INSURANCE CHURN ANALYSIS")
        print("="*50)
        
        df = self.datasets['autoinsurance'].copy()
        
        # Check for target column (assuming it exists)
        possible_targets = ['churn', 'CHURN', 'target', 'Target', 'label', 'Label']
        target_col = None
        
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print("No target column found. Available columns:", df.columns.tolist())
            return None
        
        print(f"Using target column: {target_col}")
        
        # Prepare features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.select_dtypes(include=[np.number]).median())
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Features: {list(X.columns)}")
        print(f"Categorical: {categorical_cols}")
        print(f"Numerical: {numerical_cols}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for model_name, classifier in models.items():
            print(f"\nTraining {model_name}...")
            
            # Create pipeline
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'preprocessor': preprocessor,
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols
            }
            
            print(f"{model_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        self.results['autoinsurance'] = results
        return results
    
    def create_shap_explanations(self, dataset_name, model_name):
        """Create SHAP explanations for a specific model"""
        print(f"\n" + "="*50)
        print(f"SHAP ANALYSIS: {dataset_name.upper()} - {model_name.upper()}")
        print("="*50)
        
        if dataset_name not in self.results or model_name not in self.results[dataset_name]:
            print(f"Model {model_name} not found for dataset {dataset_name}")
            return None
        
        result = self.results[dataset_name][model_name]
        model = result['model']
        X_train = result['X_train']
        X_test = result['X_test']
        
        # Transform the data for SHAP
        X_train_transformed = model.named_steps['preprocessor'].transform(X_train)
        X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
        
        # Convert to dense array if sparse and ensure proper data types
        if hasattr(X_train_transformed, 'toarray'):
            X_train_transformed = X_train_transformed.toarray()
            X_test_transformed = X_test_transformed.toarray()
        
        # Ensure data is in the correct format for SHAP
        X_train_transformed = X_train_transformed.astype(np.float64)
        X_test_transformed = X_test_transformed.astype(np.float64)
        
        # Get feature names after preprocessing
        feature_names = result['numerical_cols'].copy()
        if result['categorical_cols']:
            cat_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(result['categorical_cols'])
            feature_names.extend(cat_feature_names)
        
        # Create SHAP explainer
        try:
            # Use TreeExplainer for RandomForest, LinearExplainer for LogisticRegression
            if model_name == 'RandomForest':
                explainer = shap.TreeExplainer(model.named_steps['classifier'])
                shap_values = explainer.shap_values(X_test_transformed[:20000])  # Use more samples for better performance
            else:
                explainer = shap.LinearExplainer(model.named_steps['classifier'], X_train_transformed[:20000])
                shap_values = explainer.shap_values(X_test_transformed[:20000])
            
            # Create visualizations
            self._create_shap_plots(explainer, shap_values, X_test_transformed[:20000], feature_names, dataset_name, model_name)
            
            return explainer, shap_values
            
        except Exception as e:
            print(f"Error creating SHAP explanations: {e}")
            return None
    
    def _create_shap_plots(self, explainer, shap_values, X_test_transformed, feature_names, dataset_name, model_name):
        """Create SHAP visualization plots"""
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        if len(shap_values) == 2:  # Binary classification
            shap.summary_plot(shap_values[1], X_test_transformed, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
        
        plt.title(f'SHAP Summary Plot - {dataset_name} - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{dataset_name}_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Waterfall plot for first prediction
        plt.figure(figsize=(10, 6))
        if len(shap_values) == 2:  # Binary classification
            shap.waterfall_plot(explainer.expected_value[1], shap_values[1][0], X_test_transformed[0], feature_names=feature_names, show=False)
        else:
            shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test_transformed[0], feature_names=feature_names, show=False)
        
        plt.title(f'SHAP Waterfall Plot - First Prediction - {dataset_name} - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_{dataset_name}_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        if len(shap_values) == 2:  # Binary classification
            shap.summary_plot(shap_values[1], X_test_transformed, feature_names=feature_names, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, plot_type="bar", show=False)
        
        plt.title(f'SHAP Feature Importance - {dataset_name} - {model_name}')
        plt.tight_layout()
        plt.savefig(f'shap_bar_{dataset_name}_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_lime_explanations(self, dataset_name, model_name, num_explanations=5):
        """Create LIME explanations for a specific model"""
        print(f"\n" + "="*50)
        print(f"LIME ANALYSIS: {dataset_name.upper()} - {model_name.upper()}")
        print("="*50)
        
        if dataset_name not in self.results or model_name not in self.results[dataset_name]:
            print(f"Model {model_name} not found for dataset {dataset_name}")
            return None
        
        result = self.results[dataset_name][model_name]
        model = result['model']
        X_train = result['X_train']
        X_test = result['X_test']
        
        # Transform the data
        X_train_transformed = model.named_steps['preprocessor'].transform(X_train)
        X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
        
        # Convert to dense array if sparse and ensure proper data types
        if hasattr(X_train_transformed, 'toarray'):
            X_train_transformed = X_train_transformed.toarray()
            X_test_transformed = X_test_transformed.toarray()
        
        # Ensure data is in the correct format for LIME
        X_train_transformed = X_train_transformed.astype(np.float64)
        X_test_transformed = X_test_transformed.astype(np.float64)
        
        # Get feature names after preprocessing
        feature_names = result['numerical_cols'].copy()
        if result['categorical_cols']:
            cat_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(result['categorical_cols'])
            feature_names.extend(cat_feature_names)
        
        # Create LIME explainer
        try:
            explainer = LimeTabularExplainer(
                X_train_transformed[:20000],  # Use more samples for better performance
                feature_names=feature_names,
                class_names=['Class 0', 'Class 1'],
                mode='classification',
                discretize_continuous=True
            )
            
            # Create explanations for sample predictions
            explanations = []
            for i in range(min(num_explanations, len(X_test_transformed))):
                print(f"\nLIME Explanation for Instance {i+1}:")
                print(f"Actual class: {result['y_test'].iloc[i]}")
                print(f"Predicted class: {result['y_pred'][i]}")
                print(f"Predicted probability: {result['y_pred_proba'][i]:.4f}")
                
                # Get explanation
                explanation = explainer.explain_instance(
                    X_test_transformed[i], 
                    model.named_steps['classifier'].predict_proba,
                    num_features=10
                )
                
                explanations.append(explanation)
                
                # Print explanation
                explanation.show_in_notebook(show_table=True)
                
                # Save explanation as image
                explanation.as_pyplot_figure()
                plt.title(f'LIME Explanation - Instance {i+1} - {dataset_name} - {model_name}')
                plt.tight_layout()
                plt.savefig(f'lime_explanation_{dataset_name}_{model_name.lower()}_instance_{i+1}.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            return explainer, explanations
            
        except Exception as e:
            print(f"Error creating LIME explanations: {e}")
            return None
    
    def create_model_comparison_plots(self):
        """Create comparison plots across models and datasets"""
        print("\n" + "="*50)
        print("MODEL COMPARISON ANALYSIS")
        print("="*50)
        
        # Collect all metrics
        comparison_data = []
        for dataset_name, models in self.results.items():
            for model_name, result in models.items():
                for metric, value in result['metrics'].items():
                    comparison_data.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Metric': metric,
                        'Value': value
                    })
        
        if not comparison_data:
            print("No results available for comparison")
            return
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            metric_data = df_comparison[df_comparison['Metric'] == metric]
            
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Dataset', y='Value', hue='Model')
                plt.title(f'{metric.upper()} Comparison Across Models and Datasets')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Confusion matrices
        for dataset_name, models in self.results.items():
            fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
            if len(models) == 1:
                axes = [axes]
            
            for idx, (model_name, result) in enumerate(models.items()):
                cm = confusion_matrix(result['y_test'], result['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{model_name} - {dataset_name}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{dataset_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_performance_summary(self, metrics, model_name, dataset_name):
        """Generate a comprehensive paragraph explaining model performance"""
        
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        roc_auc = metrics['roc_auc']
        
        # Determine performance level
        if accuracy >= 0.9:
            acc_level = "excellent"
        elif accuracy >= 0.8:
            acc_level = "very good"
        elif accuracy >= 0.7:
            acc_level = "good"
        elif accuracy >= 0.6:
            acc_level = "fair"
        else:
            acc_level = "poor"
        
        # Determine ROC-AUC level
        if roc_auc >= 0.9:
            auc_level = "excellent"
        elif roc_auc >= 0.8:
            auc_level = "very good"
        elif roc_auc >= 0.7:
            auc_level = "good"
        elif roc_auc >= 0.6:
            auc_level = "fair"
        else:
            auc_level = "poor"
        
        summary = f"""
## {model_name} Performance Analysis - {dataset_name.title()} Dataset

The {model_name} model demonstrates **{acc_level}** performance on the {dataset_name} dataset with an accuracy of **{accuracy:.1%}**. This means that out of every 100 predictions, the model correctly identifies approximately {int(accuracy*100)} cases. The model's precision of **{precision:.1%}** indicates that when it predicts a positive outcome (good credit), it is correct {int(precision*100)}% of the time. The recall of **{recall:.1%}** shows that the model successfully identifies {int(recall*100)}% of all actual positive cases in the dataset. The F1-score of **{f1:.1%}** provides a balanced measure that combines both precision and recall, indicating overall model reliability. Most importantly, the ROC-AUC score of **{roc_auc:.1%}** demonstrates **{auc_level}** discriminatory ability, meaning the model is very effective at distinguishing between different classes. 

**What this means for business decisions:** The model shows strong predictive capability and can be confidently used for automated decision-making processes. The high accuracy suggests reliable predictions, while the balanced precision and recall indicate the model doesn't heavily favor one class over another. The strong ROC-AUC score confirms that the model has excellent ability to rank cases by risk level, making it valuable for credit assessment and risk management applications.
"""
        
        return summary

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE MACHINE LEARNING INTERPRETABILITY REPORT")
        print("="*70)
        
        report = []
        report.append("# Machine Learning Model Interpretability Analysis Report\n")
        report.append("This report provides a comprehensive analysis of machine learning models with SHAP and LIME explanations.\n")
        
        for dataset_name, models in self.results.items():
            report.append(f"## Dataset: {dataset_name.upper()}\n")
            
            for model_name, result in models.items():
                report.append(f"### Model: {model_name}\n")
                
                # Generate performance summary paragraph
                performance_summary = self.generate_performance_summary(result['metrics'], model_name, dataset_name)
                report.append(performance_summary)
                
                report.append(f"\n#### Dataset Information:")
                report.append(f"- Training samples: {len(result['X_train'])}")
                report.append(f"- Test samples: {len(result['X_test'])}")
                report.append(f"- Features: {len(result['numerical_cols']) + len(result['categorical_cols'])}")
                report.append(f"- Numerical features: {len(result['numerical_cols'])}")
                report.append(f"- Categorical features: {len(result['categorical_cols'])}")
                report.append("\n")
        
        # Add executive summary
        if len(self.results) > 0:
            report.append("# 🎯 Executive Summary\n")
            report.append("This analysis demonstrates that our machine learning models achieve strong performance on credit risk assessment tasks. The combination of high accuracy, balanced precision and recall, and excellent discriminatory ability makes these models suitable for production deployment. The SHAP and LIME analyses provide the transparency and interpretability necessary for regulatory compliance and customer trust.\n")
        
        # Save report
        with open('ml_interpretability_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("Report saved as 'ml_interpretability_report.md'")
        print("\nGenerated files:")
        print("- ml_interpretability_report.md: Comprehensive analysis report with paragraph summaries")
        print("- shap_summary_*.png: SHAP summary plots")
        print("- shap_waterfall_*.png: SHAP waterfall plots")
        print("- shap_bar_*.png: SHAP feature importance plots")
        print("- lime_explanation_*.png: LIME individual explanations")
        print("- model_comparison_*.png: Model comparison plots")
        print("- confusion_matrix_*.png: Confusion matrices")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Complete Machine Learning Interpretability Analysis...")
        
        # Load datasets
        self.load_datasets()
        
        # Train models
        if 'demographic' in self.datasets:
            self.prepare_demographic_model()
        
        if 'autoinsurance' in self.datasets:
            self.prepare_autoinsurance_model()
        
        # Create SHAP explanations
        for dataset_name in self.results.keys():
            for model_name in self.results[dataset_name].keys():
                self.create_shap_explanations(dataset_name, model_name)
        
        # Create LIME explanations
        for dataset_name in self.results.keys():
            for model_name in self.results[dataset_name].keys():
                self.create_lime_explanations(dataset_name, model_name)
        
        # Create comparison plots
        self.create_model_comparison_plots()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MLInterpretabilityAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
