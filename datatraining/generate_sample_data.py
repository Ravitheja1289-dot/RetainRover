"""
Generate realistic sample churn data for testing
This script creates a more comprehensive dataset with realistic patterns.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def generate_churn_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic churn data with patterns.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with churn data
    """
    np.random.seed(random_state)
    
    # Generate customer IDs
    customer_ids = range(1, n_samples + 1)
    
    # Generate age (18-80, with some skew towards middle age)
    ages = np.random.normal(45, 15, n_samples)
    ages = np.clip(ages, 18, 80).astype(int)
    
    # Generate gender (0=Female, 1=Male)
    genders = np.random.choice([0, 1], n_samples, p=[0.48, 0.52])
    
    # Generate tenure (0-20 years, correlated with age)
    base_tenure = np.random.exponential(5, n_samples)
    tenure = np.clip(base_tenure + (ages - 30) * 0.1, 0, 20).astype(int)
    
    # Generate balance (with some correlation to tenure and age)
    base_balance = np.random.lognormal(10, 1, n_samples)
    balance = base_balance * (1 + tenure * 0.1) * (1 + (ages - 30) * 0.01)
    
    # Generate number of products (1-4, correlated with tenure)
    base_products = np.random.poisson(2, n_samples)
    products_number = np.clip(base_products + (tenure > 5).astype(int), 1, 4)
    
    # Generate credit card (correlated with balance and tenure)
    credit_card_prob = 0.3 + 0.4 * (balance > np.percentile(balance, 50)) + 0.2 * (tenure > 3)
    credit_card = np.random.binomial(1, np.clip(credit_card_prob, 0, 1), n_samples)
    
    # Generate active member (correlated with products and tenure)
    active_prob = 0.4 + 0.3 * (products_number > 2) + 0.2 * (tenure > 2)
    active_member = np.random.binomial(1, np.clip(active_prob, 0, 1), n_samples)
    
    # Generate estimated salary (correlated with age and balance)
    base_salary = np.random.lognormal(10.5, 0.8, n_samples)
    estimated_salary = base_salary * (1 + (ages - 30) * 0.02) * (1 + (balance > np.percentile(balance, 50)) * 0.3)
    
    # Generate churn (with realistic patterns)
    # Higher churn for: low balance, few products, inactive members, short tenure
    churn_prob = (
        0.15 +  # Base churn rate (increased for more realistic data)
        -0.4 * (balance > np.percentile(balance, 60)) +  # Lower churn for high balance
        -0.25 * (products_number > 2) +  # Lower churn for multiple products
        -0.4 * active_member +  # Lower churn for active members
        -0.3 * (tenure > 2) +  # Lower churn for longer tenure
        -0.15 * credit_card +  # Lower churn for credit card holders
        0.2 * (ages > 55) +  # Higher churn for older customers
        0.1 * (balance < np.percentile(balance, 20)) +  # Higher churn for low balance
        np.random.normal(0, 0.15, n_samples)  # Random noise
    )
    
    churn = np.random.binomial(1, np.clip(churn_prob, 0, 1), n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'tenure': tenure,
        'balance': balance.round(2),
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary.round(2),
        'churn': churn
    })
    
    return data

def add_missing_values(data: pd.DataFrame, missing_rate: float = 0.05) -> pd.DataFrame:
    """
    Add realistic missing values to the dataset.
    
    Args:
        data: Input dataframe
        missing_rate: Proportion of values to make missing
        
    Returns:
        DataFrame with missing values
    """
    data_with_missing = data.copy()
    
    # Add missing values to specific columns
    np.random.seed(42)
    
    # Balance might be missing for new customers
    new_customer_mask = data['tenure'] < 1
    balance_missing = np.random.binomial(1, missing_rate * 3, len(data))
    data_with_missing.loc[new_customer_mask & balance_missing, 'balance'] = np.nan
    
    # Estimated salary might be missing
    salary_missing = np.random.binomial(1, missing_rate, len(data))
    data_with_missing.loc[salary_missing, 'estimated_salary'] = np.nan
    
    # Products number might be missing for very new customers
    very_new_mask = data['tenure'] == 0
    products_missing = np.random.binomial(1, missing_rate * 2, len(data))
    data_with_missing.loc[very_new_mask & products_missing, 'products_number'] = np.nan
    
    return data_with_missing

def main():
    """Generate and save sample churn data."""
    print("Generating sample churn data...")
    
    # Generate base data
    data = generate_churn_data(n_samples=1000)
    
    # Add missing values for realism
    data_with_missing = add_missing_values(data, missing_rate=0.05)
    
    # Save to CSV
    data_with_missing.to_csv('data/churn_data.csv', index=False)
    
    print(f"✅ Generated {len(data_with_missing)} samples")
    print(f"📊 Churn rate: {data_with_missing['churn'].mean():.2%}")
    print(f"📈 Missing values:")
    print(data_with_missing.isnull().sum())
    print(f"💾 Saved to data/churn_data.csv")

if __name__ == "__main__":
    main()