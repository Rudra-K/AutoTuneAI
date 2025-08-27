import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from autotune.main_tuner import AutoTuneTuner
from autotune.utils.logger import get_logger

logger = get_logger(__name__)

def run_lr_experiment(data_path: str, n_trials: int, adaptive: bool):
    """
    A self-contained function to run the full Logistic Regression tuning experiment,
    """
    logger.info(f"Starting Logistic Regression Experiment:")
    logger.info(f"Dataset: {data_path}, Trials: {n_trials}, Adaptive: {adaptive}")

    try:
        df = pd.read_csv(data_path)
        X = df.drop("Class", axis=1)
        y = df["Class"]
    except Exception as e:
        logger.error(f"Failed to load data. Error: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Before Tuning
    logger.info("Training baseline model for comparison:")
    baseline_model = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, solver='liblinear'
    )
    baseline_model.fit(X_train_scaled, y_train)
    y_pred_baseline = baseline_model.predict(X_test_scaled)

    metrics_before = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "precision": precision_score(y_test, y_pred_baseline, average="macro"),
        "recall": recall_score(y_test, y_pred_baseline, average="macro"),
        "f1_score": f1_score(y_test, y_pred_baseline, average="macro")
    }
    
    def objective(params):
        model = LogisticRegression(
            C=params["C"], max_iter=int(params["max_iter"]),
            class_weight="balanced", random_state=42, solver='liblinear'
        )
        score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1_macro").mean()
        return 1.0 - score

    # Tuning
    model_to_tune = LogisticRegression()
    tuner = AutoTuneTuner(model=model_to_tune, objective_function=objective)
    best_params, best_loss = tuner.tune(n_trials=n_trials, adaptive=adaptive)

    # After Tuning
    logger.info("Training final model with best parameters:")
    tuned_model = LogisticRegression(
        class_weight='balanced', random_state=42, solver='liblinear', **best_params
    )
    tuned_model.fit(X_train_scaled, y_train)
    y_pred_tuned = tuned_model.predict(X_test_scaled)
    
    metrics_after = {
        "accuracy": accuracy_score(y_test, y_pred_tuned),
        "precision": precision_score(y_test, y_pred_tuned, average="macro"),
        "recall": recall_score(y_test, y_pred_tuned, average="macro"),
        "f1_score": f1_score(y_test, y_pred_tuned, average="macro")
    }

    logger.info("Logistic Regression Experiment Finished:")
    tuner.plot_metrics_comparison(metrics_before, metrics_after, save_path="lr_experiment_outputs")