import matplotlib
matplotlib.use('Agg')  # Set the Agg backend for non-interactive plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('winequality-red.csv', sep=';')

# Step 4: Preliminary Analysis

# a. Descriptive Statistics
print("Descriptive Statistics:")
print(data.describe())

# b. Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # Save the plot instead of showing it

# c. Visualizations

# Box plot for Alcohol Content vs Quality
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='alcohol', data=data)
plt.title('Alcohol Content vs Quality')
plt.savefig('alcohol_vs_quality.png')  # Save the plot instead of showing it

# Pair plot to visualize relationships between features
pairplot = sns.pairplot(data, hue='quality')
pairplot.fig.savefig('pairplot.png')  # Save the pairplot using the `fig.savefig()` method

# d. Inferences
# Calculate correlations with quality
correlations = data.corr()['quality'].sort_values(ascending=False)
print("Correlations with Quality:")
print(correlations)

# Additional analysis can be performed based on the visualizations and correlations.
