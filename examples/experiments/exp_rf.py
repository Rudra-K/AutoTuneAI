import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging

from autotune.main_tuner import AutoTuneTuner
from autotune.utils.callbacks import FormattedTrialCallback

def run_rf_experiment(data_path: str, n_trials: int, adaptive: bool):
    """
    Runs a full RandomForest tuning experiment using the AutoTuneTuner.
    """
    logging.getLogger("optuna").setLevel(logging.WARNING)
    print(f"\nStarting RandomForest Experiment:")

    # Data Prep
    df = pd.read_csv(data_path)
    #df = df.sample(frac=0.05, random_state=42)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Baseline Model
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    y_pred_baseline = baseline_model.predict(X_test_scaled)
    metrics_before = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "precision": precision_score(y_test, y_pred_baseline, average="macro"),
        "recall": recall_score(y_test, y_pred_baseline, average="macro"),
        "f1_score": f1_score(y_test, y_pred_baseline, average="macro")
    }

    # Objective Function
    def objective(params):
        model = RandomForestClassifier(random_state=42, **params)
        score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1_macro").mean()
        return 1.0 - score
        
    print("\nRunning AutoTuneAI")
    model_to_tune = RandomForestClassifier()
    
    # Create an instance of AutoTune
    tuner = AutoTuneTuner(model=model_to_tune, objective_function=objective)
    
    best_params, best_loss = tuner.tune(
        n_trials=n_trials, 
        adaptive=adaptive,
        callbacks=[FormattedTrialCallback()]
    )

    print("\nTraining Final Tuned Model:")
    tuned_model = RandomForestClassifier(random_state=42, **best_params)
    tuned_model.fit(X_train_scaled, y_train)
    y_pred_tuned = tuned_model.predict(X_test_scaled)
    metrics_after = {
        "accuracy": accuracy_score(y_test, y_pred_tuned),
        "precision": precision_score(y_test, y_pred_tuned, average="macro"),
        "recall": recall_score(y_test, y_pred_tuned, average="macro"),
        "f1_score": f1_score(y_test, y_pred_tuned, average="macro")
    }

    # Results
    print("\nComparing Results:")
    tuner.plot_metrics_comparison(metrics_before, metrics_after)