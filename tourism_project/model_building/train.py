"""Tune, evaluate, track, and export the Wellness Tourism purchase classifier."""

import os
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")   # quiet MLflow git warning if git is absent

import io
import contextlib
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)

# ---------------------------------------------------------------- Tracking
# The workflow starts an MLflow server on the runner at localhost:5000
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")   # separate from dev experiment

# ---------------------------------------------------------------- Load splits (from workflow artifact)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# ---------------------------------------------------------------- Feature groups
# Continuous measurements -> median + scaling
continuous_features = ["Age", "DurationOfPitch", "NumberOfTrips", "MonthlyIncome"]

# Numeric codes with few levels -> mode + one-hot
discrete_cat_features = [
    "CityTier", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting",
]

# Text categories -> mode + one-hot
nominal_cat_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

continuous_features   = [c for c in continuous_features if c in Xtrain.columns]
discrete_cat_features = [c for c in discrete_cat_features if c in Xtrain.columns]
nominal_cat_features  = [c for c in nominal_cat_features if c in Xtrain.columns]

# ---------------------------------------------------------------- Preprocessor
def categorical_pipe():
    """Mode imputation followed by one-hot encoding (unknown levels -> all zeros)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), continuous_features),
    ("disc", categorical_pipe(), discrete_cat_features),
    ("nom",  categorical_pipe(), nominal_cat_features),
])

# ---------------------------------------------------------------- Model + grid
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42)),
])

param_grid = {
    "classifier__max_depth":         [4, 6, 8, 10, None],
    "classifier__min_samples_split": [2, 10, 20],
    "classifier__min_samples_leaf":  [1, 5, 10],
    "classifier__criterion":         ["gini", "entropy"],
    "classifier__class_weight":      [None, "balanced"],
}

with mlflow.start_run(run_name="decision_tree_gridsearch"):

    # ------------------------------------------------------------ Tune
    # n_jobs=-1 is safe on the Linux runner (uses all cores)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="f1")
    grid_search.fit(Xtrain, ytrain)

    # ------------------------------------------------------------ Log every tuned parameter set
    results = grid_search.cv_results_
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    print("Best params:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # ------------------------------------------------------------ Evaluate
    ytrain_pred = best_model.predict(Xtrain)
    ytest_pred  = best_model.predict(Xtest)
    ytest_proba = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "train_accuracy":  accuracy_score(ytrain, ytrain_pred),
        "test_accuracy":   accuracy_score(ytest, ytest_pred),
        "train_precision": precision_score(ytrain, ytrain_pred, zero_division=0),
        "test_precision":  precision_score(ytest, ytest_pred, zero_division=0),
        "train_recall":    recall_score(ytrain, ytrain_pred, zero_division=0),
        "test_recall":     recall_score(ytest, ytest_pred, zero_division=0),
        "train_f1":        f1_score(ytrain, ytrain_pred, zero_division=0),
        "test_f1":         f1_score(ytest, ytest_pred, zero_division=0),
        "test_roc_auc":    roc_auc_score(ytest, ytest_proba),
    }
    mlflow.log_metrics(metrics)

    for name, value in metrics.items():
        print(f"{name:16s}: {value:.4f}")
    print("\nConfusion matrix (test):")
    print(confusion_matrix(ytest, ytest_pred))
    print("\nClassification report (test):")
    print(classification_report(ytest, ytest_pred, zero_division=0))

    # ------------------------------------------------------------ Save next to app.py
    # The whole pipeline (preprocessing + tree) is saved, so the app can pass raw inputs
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")   # traceability in MLflow
    print(f"Model saved to {model_path}")
