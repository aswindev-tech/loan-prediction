import json
import os

NOTEBOOK_PATH = "work.ipynb"

# --- Visualization Code to Append ---
# We define them as list of strings, each string is a line.
# Note: In Jupyter cells, lines should end with \n usually, but json dump handles lists.

viz_imports = [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "sns.set_theme(style=\"whitegrid\")\n"
]

outlier_func = [
    "def remove_outliers(data, column):\n",
    "    Q1 = data[column].quantile(0.25)\n",
    "    Q3 = data[column].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower_bound = Q1 - 1.5 * IQR\n",
    "    upper_bound = Q3 + 1.5 * IQR\n",
    "    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]\n"
]

viz_code = [
    "# Pre-processing: Remove Outliers\n",
    "df_clean = df.copy()\n",
    "if 'Annual Income (INR)' in df_clean.columns:\n",
    "    df_clean = remove_outliers(df_clean, 'Annual Income (INR)')\n",
    "if 'Approved Loan Amount (INR)' in df_clean.columns:\n",
    "    df_clean = remove_outliers(df_clean, 'Approved Loan Amount (INR)')\n",
    "\n",
    "# 1. Loan Status Distribution\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.countplot(data=df_clean, x='Loan Status', palette='viridis')\n",
    "plt.title('Distribution of Loan Status (Outliers Removed)', fontsize=16)\n",
    "plt.show()\n",
    "\n",
    "# 2. Loan Amount Distribution\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(data=df_clean, x='Approved Loan Amount (INR)', bins=30, kde=True, color='skyblue')\n",
    "plt.title('Distribution of Approved Loan Amounts (Outliers Removed)', fontsize=16)\n",
    "plt.show()\n",
    "\n",
    "# 3. Correlation Heatmap\n",
    "plt.figure(figsize=(10, 8))\n",
    "numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns\n",
    "corr_matrix = df_clean[numeric_cols].corr()\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n",
    "plt.title('Correlation Heatmap', fontsize=16)\n",
    "plt.show()\n"
]

def create_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Append new cells
        nb['cells'].append(create_code_cell(viz_imports))
        nb['cells'].append(create_code_cell(outlier_func))
        nb['cells'].append(create_code_cell(viz_code))
        
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
        print(f"Successfully appended visualization cells to {NOTEBOOK_PATH}")
        
    except Exception as e:
        print(f"Failed to update notebook: {e}")

if __name__ == "__main__":
    update_notebook()
