import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import optuna
import numpy as np

from autotune.utils.logger import get_logger
from autotune.utils.callbacks import FormattedTrialCallback

logger = get_logger(__name__)

def run_svm_experiment(data_path: str, n_trials: int):
    """
    Runs a full SVM tuning experiment, including baseline comparison and a final summary.
    """
    logger.info(f"Starting SVM Experiment:")
    
    # Data Prep
    df = pd.read_csv(data_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Baseline Model
    baseline_model = SVC(random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    y_pred_baseline = baseline_model.predict(X_test_scaled)
    metrics_before = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "precision": precision_score(y_test, y_pred_baseline, average="macro"),
        "recall": recall_score(y_test, y_pred_baseline, average="macro"),
        "f1_score": f1_score(y_test, y_pred_baseline, average="macro")
    }

    # Objective Function
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        }
        model = SVC(random_state=42, **params)
        score = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="f1_macro").mean()
        return 1.0 - score
        
    # Tuning
    logger.info(f"Starting tuning for {n_trials} trials.")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[FormattedTrialCallback()])
    best_params = study.best_params

    # Final Model
    tuned_model = SVC(random_state=42, **best_params)
    tuned_model.fit(X_train_scaled, y_train)
    y_pred_tuned = tuned_model.predict(X_test_scaled)
    metrics_after = {
        "accuracy": accuracy_score(y_test, y_pred_tuned),
        "precision": precision_score(y_test, y_pred_tuned, average="macro"),
        "recall": recall_score(y_test, y_pred_tuned, average="macro"),
        "f1_score": f1_score(y_test, y_pred_tuned, average="macro")
    }

    # Results
    logger.info(f"SVM Experiment Finished")
    change_percent = {
        metric: (metrics_after[metric] - before) / before * 100 if before != 0 else 0
        for metric, before in metrics_before.items()
    }
    df_results = pd.DataFrame({
        "Metric": list(metrics_before.keys()),
        "Before": list(metrics_before.values()),
        "After": list(metrics_after.values()),
        "Change (%)": [f"{v:+.2f}%" for v in change_percent.values()]
    })
    print("\n=== Metric Change Summary ===\n")
    print(df_results.to_string(index=False))