"""
Generate a sample insurance customer churn dataset for demonstration purposes.
"""

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os

def generate_sample_data(n_samples=20000, output_file='datatraining/data/churn_data.csv'):
    """
    Generate a sample insurance customer dataset with demographics and policy information.
    
    Parameters:
        n_samples (int): Number of samples to generate
        output_file (str): Path to save the CSV file
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate customer IDs
    customer_ids = [1000 + i for i in range(1, n_samples+1)]
    
    # Generate demographic features
    ages = np.random.randint(18, 85, n_samples)
    genders = np.random.choice(['M', 'F'], n_samples)
    
    # Generate income data with reasonable distribution
    incomes = np.exp(np.random.normal(11, 0.5, n_samples)).astype(int)  # Log-normal distribution
    incomes = np.clip(incomes, 20000, 250000)  # Clip to reasonable range
    
    # Generate regions
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    
    # Generate policy features
    tenures = np.random.randint(0, 15, n_samples)  # Years with the company
    credit_scores = np.random.randint(350, 850, n_samples)
    
    # Policy types with weighted distribution
    policy_types = np.random.choice(
        ['Comprehensive', 'Third Party', 'Third Party Fire & Theft', 'Liability Only'],
        n_samples,
        p=[0.5, 0.3, 0.15, 0.05]  # Probability weights
    )
    
    # Premium amounts correlated with policy type
    premiums = []
    for policy in policy_types:
        if policy == 'Comprehensive':
            premiums.append(np.random.randint(1000, 2000))
        elif policy == 'Third Party':
            premiums.append(np.random.randint(600, 1200))
        elif policy == 'Third Party Fire & Theft':
            premiums.append(np.random.randint(800, 1400))
        else:  # Liability Only
            premiums.append(np.random.randint(500, 900))
    
    # Claim history
    claim_history = np.random.poisson(1, n_samples)  # Poisson distribution for claim counts
    claim_history = np.clip(claim_history, 0, 5)  # Cap at 5 claims
    
    # Generate churn based on factors that make business sense
    # Higher probability of churn for:
    # - Lower tenure
    # - Higher premiums
    # - Lower credit scores
    # - More claims
    
    # Create base probabilities
    churn_prob = np.zeros(n_samples)
    
    # Tenure effect (newer customers more likely to churn)
    churn_prob += (5 - np.clip(tenures, 0, 5)) * 0.05  # +0 to +0.25
    
    # Premium effect (higher premiums increase churn)
    normalized_premiums = StandardScaler().fit_transform(np.array(premiums).reshape(-1, 1)).flatten()
    churn_prob += normalized_premiums * 0.1  # -0.2 to +0.2
    
    # Credit score effect (lower scores increase churn)
    normalized_credit = StandardScaler().fit_transform(np.array(credit_scores).reshape(-1, 1)).flatten()
    churn_prob -= normalized_credit * 0.1  # -0.2 to +0.2
    
    # Claims effect (more claims increase churn)
    churn_prob += claim_history * 0.03  # +0 to +0.15
    
    # Add some noise
    churn_prob += np.random.normal(0, 0.05, n_samples)
    
    # Scale to 0-1 range
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())
    
    # Convert to binary
    churn = (churn_prob > 0.25).astype(int)  # Adjust threshold to get desired churn rate
    
    # Create dataframe
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'Age': ages,
        'Gender': genders,
        'Income': incomes,
        'Region': regions,
        'Tenure': tenures,
        'CreditScore': credit_scores,
        'PolicyType': policy_types,
        'Premium': premiums,
        'ClaimHistory': claim_history,
        'churn': churn
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Generated {n_samples} samples and saved to {output_file}")
    print(f"Churn rate: {df['churn'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    generate_sample_data(n_samples=20000)