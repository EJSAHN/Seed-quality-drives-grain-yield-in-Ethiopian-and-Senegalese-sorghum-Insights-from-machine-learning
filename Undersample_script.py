# This script performs classification on a dataset using various machine learning models.
# It includes data loading, preprocessing (scaling), model training (with hyperparameter tuning using GridSearchCV),
# model evaluation, and reporting of classification metrics.  Several models are compared.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Load data "data/data.csv"
file_path = "path/to/yourfile" # data/data.csv
df_p2017 = pd.read_csv(file_path)


# Subtract 1 from the values in the 'Cultivar' column to ensure the classes start from 0 for compatibility with XGBoost.
df_p2017['Cultivar'] = df_p2017['Cultivar'] - 1


# Separate features and target
X = df_p2017.drop('Cultivar', axis=1)
y = df_p2017['Cultivar']


# Split data (train + validation, test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split data (train, validation)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)


# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': {'model': LogisticRegression(random_state=42), 'params': {'C': [0.1, 1, 10]}},
    'Ridge Classifier': {'model': RidgeClassifier(random_state=42), 'params': {'alpha': [0.1, 1, 10]}},
    'Support Vector Classifier (RBF)': {'model': SVC(kernel='rbf', random_state=42), 'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}},
    'Random Forest': {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}},
    'Decision Tree': {'model': DecisionTreeClassifier(random_state=42), 'params': {'max_depth': [None, 5, 10]}},
    'Boosted Tree (XGBoost)': {'model': XGBClassifier(random_state=42), 'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}},
    'K-Nearest Neighbors': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [3, 5, 7]}}
}


# Train and evaluate models (GridSearchCV and cross-validation)
for name, model_info in models.items():
    model = model_info['model']
    params = model_info['params']

    # GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=StratifiedKFold(n_splits=5), scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val_scaled)
    print(f"{name} (Best Params: {grid_search.best_params_})")
    print("Validation Report:")
    print(classification_report(y_val, y_val_pred))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.3f}\n")

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test_scaled)
    print("Test Report:")
    print(classification_report(y_test, y_test_pred))
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}\n")


# Print cross-validation results (StratifiedKFold)
for name, model_info in models.items():  # Iterate through the values in the models dictionary
    model = model_info['model']  # Access the model object from the model_info dictionary
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in skf.split(X_train_scaled, y_train):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)
        cv_scores.append(accuracy_score(y_val_fold, y_val_pred))

    print(f"{name} Cross-Validation Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")

    # Evaluate on validation set
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)
    print("Validation Report:")
    print(classification_report(y_val, y_val_pred))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.3f}")

    # Evaluate on test set
    y_test_pred = model.predict(X_test_scaled)
    print("Test Report:")
    print(classification_report(y_test, y_test_pred))
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}\n")

