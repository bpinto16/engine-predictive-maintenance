
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
import dagshub
import mlflow.sklearn

# Constants
HF_DATASET_REPO  = "bpinto16/Predictive-Maintenance-HFSpace"
HF_MODEL_REPO    = "bpinto16/predictive-maintenance-mlflow-model"
MODEL_FILENAME   = "best_predictive_maintenance_mlflow_model.joblib"
RANDOM_STATE     = 42
MLFLOW_EXPERIMENT_NAME = "engine-predictive-maintenance-experiment"


DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name="engine-predictive-maintenance", mlflow=True)


# Classification threshold
# For a maintenance ALERT system, a missed fault (false negative) is costlier
# than a false alarm, so we bias the decision threshold toward recall.
CLASSIFICATION_THRESHOLD = 0.35

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
            "model__max_depth":        [3, 5, 7, None],
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
            "model__n_estimators":     [150, 200, 100],
            "model__max_depth":        [10, 15, None],
            "model__min_samples_leaf": [2, 4],
            "model__min_samples_split":[2,5,10],
            "model__max_features":     ["sqrt",0.6,0.8]
        },
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators":  [150, 200],
            "model__learning_rate": [1.0],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators":  [150, 200],
            "model__max_depth":     [ 3, 4,],
            "model__learning_rate": [0.01, 0.07],
            "model__subsample":     [0.6, 0.8],
            "model__min_samples_leaf":[1,3,5]
        },
    ),
    "XGBoost": (
        xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
        ),
        {
            "model__n_estimators":     [200, 250],
            "model__max_depth":        [3, 4, 5, 6],
            "model__learning_rate":    [ 0.04, 0.06, 0.08],
            "model__gamma":            [0,0.1,0.3],
            "model__min_child_weight": [5, 7],
            "model__subsample":        [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__reg_alpha":        [0,0.1,1],
            "model__reg_lambda":       [1]
        },
    ),
}
 
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
 
# representative input row for the model signature
input_example = pd.DataFrame([{c: float(Xtrain[c].median()) for c in Xtrain.columns}])
 
PRIMARY_METRIC = "pr_auc"               
 
SCORING = {                              # multi-metric CV scoring
    "pr_auc":  "average_precision",
    "roc_auc": "roc_auc",
    "recall":  "recall",
}

# ---------------------------------------------------------------------
# Tune each model, log to MLflow, track the best
# ---------------------------------------------------------------------

print("\n-- MLflow tracking --")
leaderboard = []
best_overall = {"name": None, "cv_pr_auc": -np.inf, "estimator": None, "params": None}

# Start the Parent Run (represents the whole pipeline execution)
parent_name = f"engine_maintenance_model_selection_{datetime.now().strftime('%Y%m%d-%H%M')}"
 
with mlflow.start_run(run_name=parent_name) as parent_run:
    mlflow.set_tags({
        "pipeline_step":            "train",
        "dataset":                  HF_DATASET_REPO,
        "primary_metric":           PRIMARY_METRIC,          
        "classification_threshold": str(CLASSIFICATION_THRESHOLD),
        "training_date":            datetime.now().strftime("%Y-%m-%d"),
        "data_version":             "v1",
        "models_compared":          ", ".join(candidates.keys()),
    })
 
    for name, (model, grid) in candidates.items():
        print(f"\n{'='*60}\nTuning {name}\n{'='*60}")
        with mlflow.start_run(run_name=name, nested=True):
            # Tag every child so DagsHub run search can filter:
            # tags.model_family = "GradientBoosting", etc.
            mlflow.set_tags({"model_family": name, "pipeline_step": "tune"})
 
            gs = GridSearchCV(
                estimator=build_pipeline(model),
                param_grid=grid,
                cv=cv_strategy,
                scoring=SCORING,           
                refit=PRIMARY_METRIC,       # best_estimator_ = best PR-AUC
                n_jobs=-1,
                verbose=0,
                return_train_score=True,    # audit CV overfit later
            )
            gs.fit(Xtrain, ytrain)
 
            # ---- CV metrics at the winning param combo ----
            bi = gs.best_index_
            cv_metrics = {
                f"cv_{m}": float(gs.cv_results_[f"mean_test_{m}"][bi])
                for m in SCORING
            }
            cv_metrics["cv_pr_auc_std"] = float(gs.cv_results_["std_test_pr_auc"][bi])
 
            # ---- Held-out test metrics at the operating threshold ----
            proba_te = gs.best_estimator_.predict_proba(Xtest)[:, 1]
            pred_te  = (proba_te >= CLASSIFICATION_THRESHOLD).astype(int)
            test_metrics = {
                "test_pr_auc":    average_precision_score(ytest, proba_te),
                "test_roc_auc":   roc_auc_score(ytest, proba_te),
                "test_recall":    recall_score(ytest, pred_te),
                "test_precision": precision_score(ytest, pred_te, zero_division=0),
                "test_f1":        f1_score(ytest, pred_te),
                "test_accuracy":  accuracy_score(ytest, pred_te),
            }
 
            leaderboard.append({"model": name, **cv_metrics, **test_metrics})
 
            # ---- Log params + metrics (consistent names across runs) ----
            mlflow.log_params(gs.best_params_)
            mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD) 
            mlflow.log_param("cv_n_splits", cv_strategy.get_n_splits())
            mlflow.log_metrics({**cv_metrics,
                                **{k: float(v) for k, v in test_metrics.items()}})
 
            # ---- threshold sweep as stepped metrics (renders as
            #      a curve in DagsHub; x-axis = threshold * 100) ----
            for t in np.arange(0.30, 0.61, 0.02):
                p = (proba_te >= t).astype(int)
                step = int(round(t * 100))
                mlflow.log_metric("sweep_recall",    recall_score(ytest, p), step=step)
                mlflow.log_metric("sweep_precision", precision_score(ytest, p, zero_division=0), step=step)
                mlflow.log_metric("sweep_f1",        f1_score(ytest, p), step=step)
 
            # ---- full grid results as an auditable artifact ----
            cv_df = pd.DataFrame(gs.cv_results_)
            cv_path = f"cv_results_{name}.csv"
            cv_df.to_csv(cv_path, index=False)
            mlflow.log_artifact(cv_path)
 
            print(f"  Best CV PR-AUC : {cv_metrics['cv_pr_auc']:.4f} "
                  f"(± {cv_metrics['cv_pr_auc_std']:.4f})")
            print(f"  Test PR-AUC    : {test_metrics['test_pr_auc']:.4f}   "
                  f"Test Recall: {test_metrics['test_recall']:.4f}")
 
            # ---- select champion by CV PR-AUC ----
            if cv_metrics["cv_pr_auc"] > best_overall["cv_pr_auc"]:
                best_overall.update(name=name,
                                    cv_pr_auc=cv_metrics["cv_pr_auc"],
                                    estimator=gs.best_estimator_,
                                    params=gs.best_params_)

 
    # -----------------------------------------------------------------
    # 5. Leaderboard + full evaluation of the winning model
    # -----------------------------------------------------------------
    lb = pd.DataFrame(leaderboard).sort_values("test_pr_auc", ascending=False)
    lb.to_csv("model_leaderboard.csv", index=False)
    mlflow.log_artifact("model_leaderboard.csv")
    print("\n===== LEADERBOARD (sorted by test PR-AUC) =====")
    print(lb.round(4).to_string(index=False))
 
    best_pipeline = best_overall["estimator"]
    # metric selected on (PR-AUC).
    print(f"\nBEST MODEL: {best_overall['name']}  (CV PR-AUC {best_overall['cv_pr_auc']:.4f})")
    # tag + best-param logging happens ONCE here
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
    # since PR-AUC is the primary metric, track the overfit gap
    # in PR-AUC too (ROC gap kept for reference).
    train_pr  = average_precision_score(ytrain, proba_train)
    test_pr   = average_precision_score(ytest,  proba_test)
    overfit_gap_roc = train_roc - test_roc
    overfit_gap_pr  = train_pr - test_pr
 
    mlflow.log_metrics({
        "final_train_roc_auc":  train_roc,
        "final_test_roc_auc":   test_roc,
        "final_train_pr_auc":   train_pr,
        "final_test_pr_auc":    test_pr,
        "final_test_f1":        f1_score(ytest, y_pred_test),
        "final_test_recall":    recall_score(ytest, y_pred_test),
        "final_test_precision": precision_score(ytest, y_pred_test, zero_division=0),
        "overfit_gap_roc":      overfit_gap_roc,
        "overfit_gap_pr":       overfit_gap_pr,
    })
 
    print("\n-- Final evaluation (winning model) --")
    print("Test set classification report:")
    print(classification_report(ytest, y_pred_test, target_names=["Normal", "Faulty"]))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(ytest, y_pred_test))
    print(f"\nTrain PR-AUC : {train_pr:.4f}   Test PR-AUC : {test_pr:.4f}")
    print(f"Train ROC-AUC: {train_roc:.4f}   Test ROC-AUC: {test_roc:.4f}")
    print(f"Overfit gap (PR): {overfit_gap_pr:.4f}  "
          f"({'acceptable' if overfit_gap_pr < 0.05 else 'WARNING: possible overfit'})")
 
    # Register the best model in the MLflow registry
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
    "primary_metric":           PRIMARY_METRIC,
    "cv_pr_auc":                float(best_overall["cv_pr_auc"]),
    "test_pr_auc":              float(test_pr),
    "test_roc_auc":             float(test_roc),
    "test_recall_at_threshold": float(recall_score(ytest, y_pred_test)),
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
