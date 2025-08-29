import os
import warnings
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging
import optuna

from autotune.main_tuner import AutoTuneTuner
from autotune.utils.callbacks import FormattedTrialCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger("optuna").setLevel(logging.WARNING)


def run_lgbm_experiment(data_path: str, n_trials: int, adaptive: bool):
    """
    Runs a full LightGBM tuning experiment, focusing on the recall metric.
    """
    logging.getLogger("optuna").setLevel(logging.WARNING)
    print(f"\nStarting LightGBM Experiment:")

    # 1. Data Prep
    print("Preparing Data:")
    df = pd.read_csv(data_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 2. Baseline Model
    print("\n--- 2. Training Baseline Model ---")
    baseline_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    metrics_before = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "precision": precision_score(y_test, y_pred_baseline), # Note: not macro for binary
        "recall": recall_score(y_test, y_pred_baseline),
        "f1_score": f1_score(y_test, y_pred_baseline)
    }

    # 3. Objective Function (Focusing on Recall)
    def objective(params):
        neg_count = y_train.value_counts()[0]
        pos_count = y_train.value_counts()[1]
        params['scale_pos_weight'] = neg_count / pos_count
        
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
        
        score = cross_val_score(model, X_train, y_train, cv=3, scoring="average_precision").mean()
        return 1.0 - score
    # 4. Tuning
    print("\nRunning AutoTuneAI:")
    model_to_tune = lgb.LGBMClassifier()
    tuner = AutoTuneTuner(model=model_to_tune, objective_function=objective)
    
    best_params, _best_loss = tuner.tune(
        n_trials=n_trials, 
        adaptive=adaptive,
        callbacks=[FormattedTrialCallback()]
    )

    # 5. Final Model
    print("\nTraining Final Tuned Model:")
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    best_params['scale_pos_weight'] = neg_count / pos_count
    tuned_model = lgb.LGBMClassifier(random_state=42, **best_params)
    tuned_model.fit(X_train, y_train)
    y_pred_tuned = tuned_model.predict(X_test)
    metrics_after = {
        "accuracy": accuracy_score(y_test, y_pred_tuned),
        "precision": precision_score(y_test, y_pred_tuned),
        "recall": recall_score(y_test, y_pred_tuned),
        "f1_score": f1_score(y_test, y_pred_tuned)
    }

    # 6. Results
    print("\nComparing Results:")
    tuner.plot_metrics_comparison(metrics_before, metrics_after)