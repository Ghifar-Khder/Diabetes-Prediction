import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Create the directory if it doesn't exist
os.makedirs('data/data-split', exist_ok=True)

# Read the CSV file
df = pd.read_csv('data/diabetes.csv')

# Create a copy of the dataframe to mark zeros as missing in specified columns
columns_to_check = df.columns.difference(['Pregnancies', 'Outcome'])
df_clean = df.copy()
df_clean[columns_to_check] = df_clean[columns_to_check].replace(0, np.nan)

# Delete any case that has NaN value for Age, Glucose, BloodPressure, or BMI
print(f"Original dataset shape: {df_clean.shape}")
df_clean = df_clean.dropna(subset=['Age', 'Glucose', 'BloodPressure', 'BMI'])
print(f"Dataset shape after removing NaN in Age, Glucose, BloodPressure or BMI: {df_clean.shape}")

# Drop the DiabetesPedigreeFunction column
df_clean = df_clean.drop('DiabetesPedigreeFunction', axis=1)
print(f"Dataset shape after dropping DiabetesPedigreeFunction: {df_clean.shape}")

# Split directly into 80% train and 20% test
train_df, test_df = train_test_split(
    df_clean, test_size=0.2, random_state=42, stratify=df_clean['Outcome']
)

# Print the shapes of the resulting datasets
print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Verify the proportions
total_samples = len(df_clean)
print(f"\nProportions:")
print(f"Train: {len(train_df)/total_samples:.2%} ({len(train_df)} samples)")
print(f"Test: {len(test_df)/total_samples:.2%} ({len(test_df)} samples)")

# Check the distribution of the target variable in each split
print(f"\nOutcome distribution in Train set:")
print(train_df['Outcome'].value_counts())
print(f"\nOutcome distribution in Test set:")
print(test_df['Outcome'].value_counts())

# Save the splits to CSV files
train_df.to_csv('data/data-split/train_set.csv', index=False)
test_df.to_csv('data/data-split/test_set.csv', index=False)

print("\nDatasets saved to:")
print("- data/data-split/train_set.csv")
print("- data/data-split/test_set.csv")