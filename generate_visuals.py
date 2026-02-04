import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create assets directory if it doesn't exist
if not os.path.exists('assets'):
    os.makedirs('assets')

# Load data
df = pd.read_csv('loan.csv')

# --- Outlier Removal using IQR ---
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

print(f"Original shape: {df.shape}")
df_clean = remove_outliers(df, 'Annual Income (INR)')
df_clean = remove_outliers(df_clean, 'Approved Loan Amount (INR)')
print(f"Shape after outlier removal: {df_clean.shape}")


# Set style
sns.set_theme(style="whitegrid")

# 1. Loan Status Distribution (Bar Chart)
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df_clean, x='Loan Status (Pass/Rejected)', palette='viridis')
plt.title('Distribution of Loan Status', fontsize=16)
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('assets/loan_status_distribution.png', bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap (Numerical columns only)
plt.figure(figsize=(10, 8))
numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df_clean[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Numerical Features)', fontsize=16)
plt.savefig('assets/correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 3. Loan Amount Distribution (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=df_clean, x='Approved Loan Amount (INR)', bins=30, kde=True, color='skyblue')
plt.title('Distribution of Approved Loan Amounts (Outliers Removed)', fontsize=16)
plt.xlabel('Approved Loan Amount (INR)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('assets/loan_amount_distribution.png', bbox_inches='tight')
plt.close()

print("Charts generated successfully in 'assets/' folder.")
