import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Datasets/demographic.csv', low_memory=False)

# Drop ID column
df = df.drop('INDIVIDUAL_ID', axis=1)
print(f"Dataset shape after dropping ID: {df.shape}")

# Handle missing values: Drop rows with missing categorical features
# (MARITAL_STATUS and HOME_MARKET_VALUE have missings)
initial_rows = len(df)
df = df.dropna(subset=['MARITAL_STATUS', 'HOME_MARKET_VALUE'])
print(f"Rows after dropping missing categoricals: {len(df)} (dropped {initial_rows - len(df)} rows)")

# Define features and target
X = df.drop('GOOD_CREDIT', axis=1)
y = df['GOOD_CREDIT']

# Identify categorical and numerical columns
categorical_cols = ['MARITAL_STATUS', 'HOME_MARKET_VALUE']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),  # Impute numerical if any missings (though none expected)
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Create pipeline with preprocessor and classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Feature importances (after preprocessing)
feature_names = (numerical_cols + 
                 list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)))
importances = model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importances:")
print(feature_importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()

# Optional: Save predictions
predictions_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred, 'proba': y_pred_proba})
predictions_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")
