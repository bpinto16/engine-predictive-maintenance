
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# for preprocessing and pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
)

import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

# for experiment tracking
import mlflow
import mlflow.sklearn

# Constants
HF_DATASET_REPO  = "bpinto16/Predictive-Maintenance-HFSpace"
HF_MODEL_REPO    = "bpinto16/predictive-maintenance-mlflow-model"
MODEL_FILENAME   = "best_predictive_maintenance_mlflow_model.joblib"
RANDOM_STATE     = 42
MLFLOW_EXPERIMENT_NAME = "engine-predictive-maintenance-experiment"

# Classification threshold
# For a maintenance ALERT system, a missed fault (false negative) is costlier
# than a false alarm, so we bias the decision threshold toward recall.
CLASSIFICATION_THRESHOLD = 0.45

TARGET_COL   = "Engine Condition"
NUMERIC_COLS = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp",
]
CAP_COLS = ["Coolant temp"] 

# Hugging Face auth
api = HfApi(token=os.getenv("HF_TOKEN"))

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
else:
    mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


# Enable automatic logging for hyperparameter tuning structures
mlflow.sklearn.autolog(max_tuning_runs=None, log_models=False)

# LOAD PREPROCESSED SPLITS
print("Load preprocessed splits from Hugging Face")
BASE = f"hf://datasets/{HF_DATASET_REPO}"

Xtrain = pd.read_csv(f"{BASE}/Xtrain.csv")
Xtest  = pd.read_csv(f"{BASE}/Xtest.csv")
ytrain = pd.read_csv(f"{BASE}/ytrain.csv").squeeze("columns")
ytest  = pd.read_csv(f"{BASE}/ytest.csv").squeeze("columns")

print(f"Xtrain : {Xtrain.shape}   ytrain : {ytrain.shape}")
print(f"Xtest  : {Xtest.shape}    ytest  : {ytest.shape}")


# ---------------------------------------------------------------------
# Shared leakage-safe preprocessor 
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[("scaler", StandardScaler(), NUMERIC_COLS)],
    remainder="drop",
    verbose_feature_names_out=False,
)

# ---------------------------------------------------------------------
# Target is ~63% faulty / 37% normal -> weight the minority (normal=0)
# ---------------------------------------------------------------------
print("Compute class imbalance weight")
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]


 
def build_pipeline(model):
    """capper (per-fold) -> scaler (per-fold) -> model. No leakage."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
 
# ---------------------------------------------------------------------
# Define candidate models + parameter grids
# ---------------------------------------------------------------------
candidates = {
    "DecisionTree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "model__max_depth":        [4, 6, 8, None],
            "model__min_samples_leaf": [1, 5, 20],
            "model__criterion":        ["gini", "entropy"],
        },
    ),
    "Bagging": (
        BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators":  [50, 100],
            "model__max_samples":   [0.7, 1.0],
            "model__max_features":  [0.7, 1.0],
        },
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators":     [100, 200],
            "model__max_depth":        [6, 10, None],
            "model__min_samples_leaf": [1, 5],
        },
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1, 1.0],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators":  [100, 200],
            "model__max_depth":     [3, 4],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample":     [0.8, 1.0],
        },
    ),
    "XGBoost": (
        xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
        ),
        {
            "model__n_estimators":     [150, 250],
            "model__max_depth":        [3, 4, 5],
            "model__learning_rate":    [0.05, 0.1],
            "model__subsample":        [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    ),
}
 
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
 
# representative input row for the model signature
input_example = pd.DataFrame([{c: float(Xtrain[c].median()) for c in Xtrain.columns}])
 

# ---------------------------------------------------------------------
# Tune each model, log to MLflow, track the best by CV ROC-AUC
# ---------------------------------------------------------------------

print("\n-- MLflow tracking --")
leaderboard = []
best_overall = {"name": None, "cv_score": -np.inf, "estimator": None, "params": None}
 
with mlflow.start_run(run_name="engine_maintenance_model_selection") as parent_run:
    mlflow.set_tags({
        "pipeline_step":            "train",
        "dataset":                  HF_DATASET_REPO,
        "classification_threshold": str(CLASSIFICATION_THRESHOLD),
        "training_date":            datetime.now().strftime("%Y-%m-%d"),
        "data_version":             "v1",
        "models_compared":          ", ".join(candidates.keys()),
    })
 
    for name, (model, grid) in candidates.items():
        print(f"\n{'='*60}\nTuning {name}\n{'='*60}")
        with mlflow.start_run(run_name=name, nested=True):
            gs = GridSearchCV(
                estimator=build_pipeline(model),
                param_grid=grid,
                cv=cv_strategy,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            gs.fit(Xtrain, ytrain)
 
            proba_te = gs.best_estimator_.predict_proba(Xtest)[:, 1]
            pred_te  = (proba_te >= CLASSIFICATION_THRESHOLD).astype(int)
 
            row = {
                "model":          name,
                "cv_roc_auc":     gs.best_score_,
                "test_roc_auc":   roc_auc_score(ytest, proba_te),
                "test_pr_auc":    average_precision_score(ytest, proba_te),
                "test_f1":        f1_score(ytest, pred_te),
                "test_recall":    recall_score(ytest, pred_te),
                "test_precision": precision_score(ytest, pred_te, zero_division=0),
                "test_accuracy":  accuracy_score(ytest, pred_te),
            }
            leaderboard.append(row)
 
            # 4a. LOG all tuned/best params + metrics for this model
            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "model"})
 
            print(f"  Best CV ROC-AUC : {gs.best_score_:.4f}")
            print(f"  Test ROC-AUC    : {row['test_roc_auc']:.4f}   Test F1: {row['test_f1']:.4f}")
 
            if gs.best_score_ > best_overall["cv_score"]:
                best_overall.update(name=name, cv_score=gs.best_score_,
                                    estimator=gs.best_estimator_, params=gs.best_params_)
 
    # -----------------------------------------------------------------
    # 5. Leaderboard + full evaluation of the winning model
    # -----------------------------------------------------------------
    lb = pd.DataFrame(leaderboard).sort_values("test_roc_auc", ascending=False)
    lb.to_csv("model_leaderboard.csv", index=False)
    mlflow.log_artifact("model_leaderboard.csv")
    print("\n===== MODEL LEADERBOARD (sorted by test ROC-AUC) =====")
    print(lb.round(4).to_string(index=False))
 
    best_pipeline = best_overall["estimator"]
    print(f"\nBEST MODEL: {best_overall['name']}  (CV ROC-AUC {best_overall['cv_score']:.4f})")
    mlflow.set_tag("best_model", best_overall["name"])
    mlflow.log_params({f"best__{k}": v for k, v in best_overall["params"].items()})
 
    # threshold sweep (recall matters for a maintenance alert system)
    proba_test  = best_pipeline.predict_proba(Xtest)[:, 1]
    proba_train = best_pipeline.predict_proba(Xtrain)[:, 1]
    print(f"\n{'Threshold':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for t in np.arange(0.30, 0.61, 0.05):
        pr = (proba_test >= t).astype(int)
        mark = "  <-- selected" if abs(t - CLASSIFICATION_THRESHOLD) < 1e-6 else ""
        print(f"{t:>10.2f}{precision_score(ytest,pr,zero_division=0):>11.3f}"
              f"{recall_score(ytest,pr):>9.3f}{f1_score(ytest,pr):>8.3f}{mark}")
 
    y_pred_train = (proba_train >= CLASSIFICATION_THRESHOLD).astype(int)
    y_pred_test  = (proba_test  >= CLASSIFICATION_THRESHOLD).astype(int)
 
    train_roc = roc_auc_score(ytrain, proba_train)
    test_roc  = roc_auc_score(ytest,  proba_test)
    overfit_gap = train_roc - test_roc
 
    mlflow.log_metrics({
        "final_train_roc_auc": train_roc,
        "final_test_roc_auc":  test_roc,
        "final_test_pr_auc":   average_precision_score(ytest, proba_test),
        "final_test_f1":       f1_score(ytest, y_pred_test),
        "final_test_recall":   recall_score(ytest, y_pred_test),
        "final_test_precision":precision_score(ytest, y_pred_test, zero_division=0),
        "overfit_gap":         overfit_gap,
    })
 
    print("\n-- Final evaluation (winning model) --")
    print("Test set classification report:")
    print(classification_report(ytest, y_pred_test, target_names=["Normal", "Faulty"]))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(ytest, y_pred_test))
    print(f"\nTrain ROC-AUC: {train_roc:.4f}   Test ROC-AUC: {test_roc:.4f}")
    print(f"Overfit gap  : {overfit_gap:.4f}  "
          f"({'acceptable' if overfit_gap < 0.05 else 'WARNING: possible overfit'})")
 
    # 6. Register the best model in the MLflow registry
    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        name="engine_maintenance_pipeline",
        registered_model_name="EnginePredictiveMaintenanceClassifier",
        input_example=input_example,
    )
    print(f"\nMLflow parent run ID: {parent_run.info.run_id}")
 
# ---------------------------------------------------------------------
# 7. Register the best model in the Hugging Face MODEL hub
# ---------------------------------------------------------------------
print("\n-- Save & upload best model to HF Model Hub --")
joblib.dump(best_pipeline, MODEL_FILENAME)
 
# also ship the metadata card so the deployment app knows what it loaded
metadata = {
    "best_model":               best_overall["name"],
    "best_params":              {k: str(v) for k, v in best_overall["params"].items()},
    "classification_threshold": CLASSIFICATION_THRESHOLD,
    "cv_roc_auc":               float(best_overall["cv_score"]),
    "test_roc_auc":             float(test_roc),
    "feature_order":            list(Xtrain.columns),
    "trained_on":               datetime.now().strftime("%Y-%m-%d"),
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
 
try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False, token=os.getenv("HF_TOKEN"))
 
for fname in [MODEL_FILENAME, "model_metadata.json", "model_leaderboard.csv"]:
    api.upload_file(path_or_fileobj=fname, path_in_repo=fname,
                    repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"  Uploaded: {fname} to {HF_MODEL_REPO}")
 
print("\ntrain.py completed successfully.")
