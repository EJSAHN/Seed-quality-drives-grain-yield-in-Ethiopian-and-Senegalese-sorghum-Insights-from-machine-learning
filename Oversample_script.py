# This script performs a classification task on an imbalanced dataset, employing several machine learning models.
# It addresses class imbalance using ADASYN, performs hyperparameter tuning with GridSearchCV,
# and uses Stratified K-Fold cross-validation to evaluate model performance.
# the classification results for each model and identifies the best-performing model based on the macro-average F1-score.

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN
import numpy as np

# 1. Load and preprocess data
file_path = "path/to/yourfile" # data/data.csv
df = pd.read_csv(file_path)
X = df.drop('Cultivar', axis=1)
y = df['Cultivar']


# 2. Address class imbalance (ADASYN)
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)


# 3. Define models and tune hyperparameters
models = {
    "Logistic Regression": (LogisticRegression(random_state=42, max_iter=1000), {'C': [0.001, 0.01, 0.1, 1, 10, 100]}),
    "Ridge Classifier": (RidgeClassifier(random_state=42), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}),
    "SVM": (SVC(random_state=42), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    "Random Forest": (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
    "XGBoost": (GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 10]})
}


# 4. Train and evaluate models (using cross-validation)
results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, param_grid) in models.items():
    all_y_true = []
    all_y_pred = []
    for train_index, test_index in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    report = classification_report(all_y_true, all_y_pred, output_dict=True)
    results[name] = report


# 5. Output results
for name, report in results.items():
    print(f"Model: {name}")
    print(classification_report(all_y_true, all_y_pred))
    print("-" * 50)


# 6. Output the best-performing model
best_model_name = max(results, key=lambda k: results[k]['macro avg']['f1-score'])
print(f"Best Model: {best_model_name}")
print(f"Best Model Report: {classification_report(all_y_true, all_y_pred)}")
    

