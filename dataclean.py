import pandas as pd

df = pd.read_csv("D:\Anjan-Data\Megathon-25\Datasets\autoinsurance_churn.csv")

# Strip whitespace
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Optimize dtypes
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = df[col].astype('int32')
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype('category')

# Remove duplicates and low-variance columns
df = df.drop_duplicates()
df = df.loc[:, df.nunique() > 1]

# Save compressed
df.to_csv("optimized.csv.gz", index=False, compression='gzip')
