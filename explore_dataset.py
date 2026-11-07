
import pandas as pd

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Display first rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Number of diseases
print("\nNUMBER OF DISEASES:")
print(df["Disease"].nunique())

# Diseases distribution
print("\nDISEASE DISTRIBUTION:")
print(df["Disease"].value_counts())
