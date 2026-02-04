import json
import os

NOTEBOOK_PATH = "work.ipynb"

# Define the new cells to add
cells_to_add = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Model Optimization and Selection\n",
            "In this section, we will:\n",
            "1. Define a preprocessing pipeline.\n",
            "2. Train multiple models (Logistic Regression, Random Forest, XGBoost).\n",
            "3. Use GridSearchCV for hyperparameter tuning.\n",
            "4. Select and save the best performing model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import pickle\n",
            "from sklearn.model_selection import train_test_split, GridSearchCV\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.impute import SimpleImputer\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import accuracy_score, classification_report\n",
            "try:\n",
            "    from xgboost import XGBClassifier\n",
            "except ImportError:\n",
            "    XGBClassifier = None\n",
            "    print(\"XGBoost not found, skipping...\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load Data\n",
            "df = pd.read_csv('loan.csv')\n",
            "\n",
            "# Separate Target and Predictors\n",
            "target = 'Loan Status (Pass/Rejected)'\n",
            "X = df.drop(['Name', target], axis=1, errors='ignore')\n",
            "y = df[target].map({'Pass': 1, 'Rejected': 0})\n",
            "\n",
            "# Identify columns\n",
            "categorical_cols = X.select_dtypes(include=['object']).columns\n",
            "numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns\n",
            "\n",
            "# Split Data\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define Preprocessing Pipeline\n",
            "numeric_transformer = Pipeline(steps=[\n",
            "    ('imputer', SimpleImputer(strategy='median')),\n",
            "    ('scaler', StandardScaler())\n",
            "])\n",
            "\n",
            "categorical_transformer = Pipeline(steps=[\n",
            "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
            "    ('encoder', OneHotEncoder(handle_unknown='ignore'))\n",
            "])\n",
            "\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('num', numeric_transformer, numerical_cols),\n",
            "        ('cat', categorical_transformer, categorical_cols)\n",
            "    ])"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define Models and Hyperparameters\n",
            "models = {\n",
            "    'LogisticRegression': {\n",
            "        'model': LogisticRegression(max_iter=1000),\n",
            "        'params': {\n",
            "            'classifier__C': [0.1, 1.0, 10.0],\n",
            "            'classifier__solver': ['liblinear', 'lbfgs']\n",
            "        }\n",
            "    },\n",
            "    'RandomForest': {\n",
            "        'model': RandomForestClassifier(random_state=42),\n",
            "        'params': {\n",
            "            'classifier__n_estimators': [50, 100, 200],\n",
            "            'classifier__max_depth': [None, 10, 20]\n",
            "        }\n",
            "    }\n",
            "}\n",
            "\n",
            "if XGBClassifier:\n",
            "    models['XGBoost'] = {\n",
            "        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),\n",
            "        'params': {\n",
            "            'classifier__n_estimators': [50, 100],\n",
            "            'classifier__learning_rate': [0.01, 0.1, 0.2],\n",
            "            'classifier__max_depth': [3, 5, 7]\n",
            "        }\n",
            "    }"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train and Optimize\n",
            "best_score = 0\n",
            "best_model = None\n",
            "best_name = \"\"\n",
            "\n",
            "results = []\n",
            "\n",
            "for name, config in models.items():\n",
            "    pipeline = Pipeline(steps=[('preprocessor', preprocessor),\n",
            "                               ('classifier', config['model'])])\n",
            "    \n",
            "    print(f\"Training {name}...\")\n",
            "    grid = GridSearchCV(pipeline, config['params'], cv=5, scoring='accuracy', n_jobs=-1)\n",
            "    grid.fit(X_train, y_train)\n",
            "    \n",
            "    y_pred = grid.predict(X_test)\n",
            "    test_score = accuracy_score(y_test, y_pred)\n",
            "    \n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Best Params': grid.best_params_,\n",
            "        'CV Score': grid.best_score_,\n",
            "        'Test Score': test_score\n",
            "    })\n",
            "    \n",
            "    if test_score > best_score:\n",
            "        best_score = test_score\n",
            "        best_model = grid.best_estimator_\n",
            "        best_name = name\n",
            "\n",
            "results_df = pd.DataFrame(results)\n",
            "print(\"\\nOptimization Results:\")\n",
            "print(results_df)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save Best Model\n",
            "print(f\"\\nSaving Best Model: {best_name}\")\n",
            "with open('best_model.pkl', 'wb') as f:\n",
            "    pickle.dump(best_model, f)\n",
            "\n",
            "# Save Column Info for App\n",
            "col_info = {\n",
            "    'numerical': list(numerical_cols),\n",
            "    'categorical': list(categorical_cols)\n",
            "}\n",
            "with open('model_columns.pkl', 'wb') as f:\n",
            "    pickle.dump(col_info, f)"
        ]
    }
]

def append_to_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print("Notebook not found!")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    nb['cells'].extend(cells_to_add)
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Appended optimization cells to {NOTEBOOK_PATH}")

if __name__ == "__main__":
    append_to_notebook()
