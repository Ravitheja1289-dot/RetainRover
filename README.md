# Machine Learning Model Interpretability Analysis with SHAP and LIME

This project provides comprehensive model interpretability analysis using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) for credit prediction and auto insurance churn datasets.

## 🎯 Project Overview

This project demonstrates how to:
- Train machine learning models on large datasets
- Implement SHAP for global and local model interpretability
- Use LIME for individual prediction explanations
- Compare model performance and interpretability
- Generate comprehensive analysis reports

## 📊 Datasets

The project includes several large datasets:

1. **Demographic Dataset** (`Datasets/demographic.csv`)
   - Credit prediction task
   - Features: Income, Marital Status, Home Market Value, etc.
   - Target: GOOD_CREDIT (binary classification)

2. **Auto Insurance Churn Dataset** (`Datasets/autoinsurance_churn.csv`)
   - Customer churn prediction
   - Various customer and policy features

3. **Additional Datasets**
   - Address data (`Datasets/address.csv`)
   - Customer data (`Datasets/customer.csv`)
   - Termination data (`Datasets/termination.csv`)

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Analysis

#### Option 1: Run the Complete Analysis Script
```bash
python ml_interpretability_analysis.py
```

#### Option 2: Use the Jupyter Notebook
```bash
jupyter notebook ml_interpretability_notebook.ipynb
```

## 📁 Project Structure

```
Megathon-25/
├── Datasets/                          # Large datasets (200MB+ each)
│   ├── demographic.csv                # Main credit prediction dataset
│   ├── autoinsurance_churn.csv        # Auto insurance churn data
│   ├── address.csv                    # Address/location data
│   ├── customer.csv                   # Customer information
│   └── termination.csv                # Account termination data
├── ml_interpretability_analysis.py    # Complete analysis script
├── ml_interpretability_notebook.ipynb # Interactive Jupyter notebook
├── accuracy.ipynb                     # Original analysis notebook
├── credit_model.py                    # Original model script
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🔍 Analysis Components

### 1. Model Training
- **Random Forest Classifier**: Non-linear ensemble method
- **Logistic Regression**: Linear interpretable model
- Proper preprocessing with categorical encoding
- Balanced class weights for imbalanced datasets

### 2. SHAP Analysis
- **Global Explanations**: Feature importance across all predictions
- **Local Explanations**: Individual prediction breakdowns
- **Summary Plots**: Feature impact visualization
- **Waterfall Plots**: Step-by-step prediction process
- **Bar Plots**: Feature importance rankings

### 3. LIME Analysis
- **Local Interpretable Explanations**: Individual instance explanations
- **Feature Importance**: Per-prediction feature contributions
- **Visual Explanations**: Easy-to-understand decision breakdowns
- **Multiple Instance Analysis**: Various prediction scenarios

### 4. Model Comparison
- **Performance Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Confusion Matrices**: Classification performance visualization
- **Feature Importance Comparison**: SHAP vs traditional methods
- **Interpretability Analysis**: Model explainability comparison

### 5. Non-Technical Summaries
- **Paragraph Explanations**: Business-friendly performance summaries
- **Executive Reports**: High-level insights for stakeholders
- **Plain Language Analysis**: Easy-to-understand explanations
- **Business Impact Assessment**: ROI and strategic recommendations

## 📈 Generated Outputs

The analysis generates several types of outputs:

### Visualizations
- `shap_summary_*.png`: SHAP summary plots
- `shap_waterfall_*.png`: SHAP waterfall plots
- `shap_bar_*.png`: SHAP feature importance plots
- `lime_explanation_*.png`: LIME individual explanations
- `model_comparison_*.png`: Model performance comparisons
- `confusion_matrix_*.png`: Confusion matrices

### Reports
- `ml_interpretability_report.md`: Comprehensive analysis report with paragraph summaries
- **Non-technical summaries**: Business-friendly explanations of all metrics
- **Executive summaries**: High-level insights for stakeholders
- **Performance paragraphs**: Plain language analysis of model results
- Feature importance rankings with business context
- Interpretability insights with practical applications

## 🛠️ Key Features

### Comprehensive Analysis
- Multiple datasets support
- Various model types (Random Forest, Logistic Regression)
- Both global and local interpretability
- Performance comparison across models

### Production Ready
- Scalable preprocessing pipelines
- Error handling and validation
- Configurable analysis parameters
- Automated report generation

### Interpretability Focus
- SHAP for consistent explanations
- LIME for local understanding
- Multiple visualization types
- Business-friendly insights

### Non-Technical Communication
- **Paragraph summaries**: Convert technical metrics into business language
- **Executive reports**: High-level insights for decision makers
- **Plain language analysis**: Easy-to-understand explanations
- **Business impact assessment**: ROI and strategic recommendations

## 📊 Model Performance

The models are evaluated using multiple metrics, with each metric explained in business-friendly language:

- **Accuracy**: Overall correct predictions (e.g., "82.3% accuracy means the model correctly identifies 82 out of every 100 cases")
- **Precision**: True positives / (True positives + False positives) (e.g., "78.9% precision means when the model predicts good credit, it's correct 78% of the time")
- **Recall**: True positives / (True positives + False negatives) (e.g., "85.7% recall means the model successfully identifies 85% of all actual good credit cases")
- **F1-Score**: Harmonic mean of precision and recall (balanced performance measure)
- **ROC-AUC**: Area under the ROC curve (discriminatory ability - how well the model distinguishes between classes)

**Non-Technical Summary**: All metrics are automatically converted into paragraph explanations that describe what the numbers mean in practical business terms, making the analysis accessible to non-technical stakeholders.

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **shap**: SHAP explanations
- **lime**: LIME explanations
- **matplotlib/seaborn**: Visualizations
- **jupyter**: Interactive notebooks

### Performance Considerations
- Large datasets (200MB+) are handled efficiently
- Sampling for SHAP/LIME explanations (1000 instances)
- Memory-optimized preprocessing
- Parallel processing where possible

## 📝 Usage Examples

### Basic Usage
```python
from ml_interpretability_analysis import MLInterpretabilityAnalyzer

# Initialize analyzer
analyzer = MLInterpretabilityAnalyzer()

# Run complete analysis
analyzer.run_complete_analysis()
```

### Custom Analysis
```python
# Load specific datasets
analyzer.load_datasets()

# Train specific models
analyzer.prepare_demographic_model()

# Create SHAP explanations
analyzer.create_shap_explanations('demographic', 'RandomForest')

# Generate LIME explanations
analyzer.create_lime_explanations('demographic', 'RandomForest')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SHAP library for model interpretability
- LIME library for local explanations
- Scikit-learn for machine learning algorithms
- The open-source community for various tools and libraries

## 📞 Support

For questions or issues:
1. Check the existing issues in the repository
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Note**: This project handles large datasets (200MB+) and may require significant computational resources for full analysis. Consider using sampling or cloud computing resources for production use.
