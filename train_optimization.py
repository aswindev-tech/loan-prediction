import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# 1. Load Data
df = pd.read_csv('loan.csv')

# 2. Preprocessing
# Drop Name (ID column)
df.drop('Name', axis=1, inplace=True, errors='ignore')

# Identify columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
# Remove target from predictors
target = 'Loan Status (Pass/Rejected)'
if target in categorical_cols:
    categorical_cols = categorical_cols.drop(target)
if target in numerical_cols:
    numerical_cols = numerical_cols.drop(target)

X = df.drop(target, axis=1)
y = df[target].map({'Pass': 1, 'Rejected': 0}) # Ensure numeric target

# Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 3. Model Pipelines
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
    }
}

if XGBClassifier:
    models['XGBoost'] = {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    }

best_score = 0
best_model = None
best_name = ""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting Model Optimization...")

for name, config in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', config['model'])])
    
    grid = GridSearchCV(pipeline, config['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"\n{name} Best Params: {grid.best_params_}")
    print(f"{name} Training Score: {grid.best_score_:.4f}")
    
    y_pred = grid.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    print(f"{name} Test Score: {test_score:.4f}")
    
    if test_score > best_score:
        best_score = test_score
        best_model = grid.best_estimator_
        best_name = name

print(f"\n---------------------------------------")
print(f"Best Model Selected: {best_name} with Accuracy: {best_score:.4f}")

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Saved best_model.pkl")

# Save column info for app.py
col_info = {
    'numerical': list(numerical_cols),
    'categorical': list(categorical_cols)
}
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(col_info, f)
print("Saved model_columns.pkl")
