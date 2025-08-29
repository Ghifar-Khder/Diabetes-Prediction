import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results/data-analysis directory if it doesn't exist
os.makedirs('results/data-analysis', exist_ok=True)

# Read the CSV file
df = pd.read_csv('data/data-split/train_set.csv') 

# Display basic dataset information
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Identify columns where 0 represents missing values (all except Pregnancies and Outcome)
columns_to_check = df.columns.difference(['Pregnancies', 'Outcome'])

# Create a copy of the data to mark zeros as missing
df_clean = df.copy()
df_clean[columns_to_check] = df_clean[columns_to_check].replace(0, np.nan)

# Count missing values (now that zeros have been converted to NaN)
missing_counts = df_clean.isnull().sum()
missing_percentage = (missing_counts / len(df_clean)) * 100

print("\nMissing values per column:")
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_percentage.round(2)  # Round to 2 decimal places
})
print(missing_df)

# Plot missing values with combined count and percentage labels
plt.figure(figsize=(14, 7))

# Bar plot for missing counts with fixed y-axis range
bars = plt.bar(missing_counts.index, missing_counts.values, alpha=0.7, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Missing Count', color='skyblue')
plt.xticks(rotation=45, ha='right')

# Set y-axis range to 0-500
plt.ylim(0, 500)

# Add combined labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = missing_percentage.values[i]
    # Position labels at a fixed height above the bars
    label_height = height + 20  # Fixed offset of 20 units
    plt.text(bar.get_x() + bar.get_width()/2., label_height,
             f'{int(height)} ({percentage:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

plt.title('Missing Values Analysis')
plt.tight_layout()
plt.savefig('results/data-analysis/missing_values_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Check if last column is binary outcome
last_column = df.columns[-1]
print(f"\nOutcome value counts ({last_column}):")
print(df[last_column].value_counts())

# Calculate correlation matrix using cleaned data (with zeros as NaN)

# and delete missing values
corr_matrix = df_clean.corr(numeric_only=True)

# Plot correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues',center=0.5,
            square=True, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('results/data-analysis/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Show correlation with outcome column
print(f"\nCorrelations with {last_column}:")
print(corr_matrix[last_column].sort_values(ascending=False))

# Show how many observations are used for each correlation
print("\nNumber of complete observations for each correlation pair:")
# Create a matrix showing count of complete pairs
complete_pairs = pd.DataFrame(index=df_clean.columns, columns=df_clean.columns)
for col1 in df_clean.columns:
    for col2 in df_clean.columns:
        complete_pairs.loc[col1, col2] = df_clean[[col1, col2]].dropna().shape[0]

print(complete_pairs)