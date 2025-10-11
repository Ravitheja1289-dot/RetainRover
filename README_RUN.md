# InsuraSense: Running the Streamlit Dashboard

This guide explains how to run the InsuraSense Streamlit dashboard for insurance churn prediction.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages in `requirements.txt`

### Installation

1) Create and activate a Python environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Generate Sample Data

Before running the dashboard, make sure you have sample data available:

```powershell
python generate_sample_data.py
```

This will create a sample dataset with 500 insurance customers in `datatraining/data/churn_data.csv`.

### Run the Streamlit Dashboard

```powershell
streamlit run app.py
```

The application should automatically open in your default web browser at `http://localhost:8501`.

## 📋 Dashboard Usage Guide

1. **Data Input Stage**
   - Use the "Load Sample Data" button or upload your own CSV
   - CSV should have columns like: customer_id, Age, Gender, Income, Region, etc.

2. **Churn Prediction Stage**
   - Click "Train Model and Predict Churn" button
   - View model performance metrics (Accuracy, AUC, Precision, Recall)
   - See predictions table sorted by churn probability

3. **Model Insights Stage**
   - Explore the "Model Insights" tab to see global feature importance
   - Understand which features are most predictive of churn

4. **Customer Analysis Stage**
   - Select individual customers from the dropdown menu
   - See SHAP explanations of why each customer might churn
   - Read plain-English insights and recommended retention strategies

## 📊 Expected Output

When running the application successfully, you should see:

1. **Data Preview**
   - Total records: 500
   - List of columns
   - Sample data table

2. **Model Performance**
   - Accuracy: ~0.87
   - ROC AUC: ~0.91
   - Precision: ~0.85
   - Recall: ~0.83

3. **Customer Predictions**
   - Table with CustomerID, Churn_Prob, Churn_Prediction
   - Sorted by churn probability (highest risk first)

4. **SHAP Explanations**
   - Feature importance bar charts
   - Individual customer explanations

## ⚙️ Customizing the Dashboard

- Modify `app.py` to add more visualization components
- Update feature engineering in the `preprocess_data` function
- Enhance the model by changing the `train_model` function

## � Performance Optimizations

The dashboard includes several performance optimizations:

1. **Streamlit Caching**
   - Uses `@st.cache_resource` for model loading to prevent reloading on every interaction
   - Uses `@st.cache_data` for predictions and SHAP calculations to cache results

2. **Batch Processing for Large Datasets**
   - Automatically processes large datasets in batches of 2000 rows
   - Shows a progress bar during batch processing
   - Reports performance metrics for large datasets

3. **Limited SHAP Sample**
   - Uses a maximum of 200 rows for SHAP value calculation to maintain responsiveness
   - Still provides accurate feature importance while being much faster

4. **Model Persistence**
   - Saves the trained model pipeline for future use
   - Provides "Load Saved Model" button for quick startup

5. **Large File Handling**
   - Limits large CSV uploads to 1000 rows to prevent memory issues
   - Displays warning when files are truncated

## �🔍 Troubleshooting

- If you see "No module found" errors, ensure you've installed all requirements
- If data doesn't load, check that the `datatraining/data` directory exists
- For visualization issues, make sure you have the latest versions of streamlit and plotly

For more information, please contact: contact@insurasense.com
