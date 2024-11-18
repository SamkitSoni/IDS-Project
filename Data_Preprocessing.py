import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Step 1: Load datasets
# Update these paths if needed
red_wine_path = "C:/Users/HP/OneDrive/Desktop/IDS/winequality-red.csv"
white_wine_path = "C:/Users/HP/OneDrive/Desktop/IDS/winequality-White.csv"

# Read red and white wine datasets
red_wine = pd.read_csv(red_wine_path, delimiter=';')
white_wine = pd.read_csv(white_wine_path, delimiter=';')

# Add a column to identify wine type
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

# Step 2: Check for missing values
missing_values = wine_data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Step 3: Detect outliers using z-scores
# Identify numeric columns (excluding 'quality' and 'wine_type')
numeric_cols = wine_data.columns[:-2]

# Compute z-scores for numeric columns
z_scores = wine_data[numeric_cols].apply(zscore)

# Count the number of outliers in each column
outliers = (np.abs(z_scores) > 3).sum()
print("Outliers per feature:\n", outliers)

# Optional: Remove outliers (uncomment below if you want to remove outliers)
# wine_data = wine_data[(np.abs(z_scores) <= 3).all(axis=1)]

# Step 4: Normalize numeric data
scaler = MinMaxScaler()
wine_data[numeric_cols] = scaler.fit_transform(wine_data[numeric_cols])

# Step 5: Save preprocessed data
output_path = "C:/Users/HP/OneDrive/Desktop/IDS/output_file.csv"
wine_data.to_csv(output_path, index=False)
print(f"Preprocessed data saved to: {output_path}")
