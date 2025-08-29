import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import optuna
from optuna.integration import TFKerasPruningCallback

from autotune.utils.logger import get_logger
from autotune.utils.callbacks import FormattedTrialCallback
from examples.models.cnn_tf_model import create_cnn_model

logger = get_logger(__name__)

def run_cnn_experiment(n_trials: int, epochs: int):
    """
    Runs a full CNN tuning experiment on CIFAR-10
    """
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    logger.info("Starting CNN Experiment:")

    # Data Prep
    logger.info("Loading and preprocessing CIFAR-10:")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val = x_train[-val_size:]
    y_val = y_train_cat[-val_size:]
    x_train_sub = x_train[:-val_size]
    y_train_sub = y_train_cat[:-val_size]

    # Baselin Model
    logger.info("Training baseline model for comparison:")
    baseline_model = create_cnn_model() # Using default parameters
    baseline_model.compile(optimizer='adam',
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
    baseline_model.fit(x_train_sub, y_train_sub, validation_data=(x_val, y_val),
                       epochs=epochs, batch_size=128, verbose=0)
    
    y_pred_baseline_probs = baseline_model.predict(x_test)
    y_pred_baseline = np.argmax(y_pred_baseline_probs, axis=1)
    
    metrics_before = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "precision": precision_score(y_test, y_pred_baseline, average="macro"),
        "recall": recall_score(y_test, y_pred_baseline, average="macro"),
        "f1_score": f1_score(y_test, y_pred_baseline, average="macro")
    }
    
    # Objective Function
    def objective(trial):
        tf.keras.backend.clear_session()
        params = {
            'num_filters_1': trial.suggest_int('num_filters_1', 16, 64),
            'num_filters_2': trial.suggest_int('num_filters_2', 32, 128),
            'dense_units': trial.suggest_int('dense_units', 256, 1024),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
        }
        model = create_cnn_model(**params)
        optimizer = getattr(tf.keras.optimizers, params['optimizer'])(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
        model.fit(x_train_sub, y_train_sub, validation_data=(x_val, y_val),
                  epochs=epochs, batch_size=128, callbacks=[pruning_callback], verbose=0)
        _, accuracy = model.evaluate(x_val, y_val, verbose=0)
        return accuracy

    # Tuning
    logger.info(f"Starting tuning for {n_trials} trials.")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, callbacks=[FormattedTrialCallback()])
    best_params = study.best_trial.params

    # Final Model
    logger.info("Training final model with best parameters:")
    tuned_model = create_cnn_model(**best_params)
    optimizer = getattr(tf.keras.optimizers, best_params['optimizer'])(learning_rate=best_params['learning_rate'])
    tuned_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
    tuned_model.fit(x_train, y_train_cat, epochs=epochs, batch_size=128, verbose=0)
    
    y_pred_tuned_probs = tuned_model.predict(x_test)
    y_pred_tuned = np.argmax(y_pred_tuned_probs, axis=1)
    
    metrics_after = {
        "accuracy": accuracy_score(y_test, y_pred_tuned),
        "precision": precision_score(y_test, y_pred_tuned, average="macro"),
        "recall": recall_score(y_test, y_pred_tuned, average="macro"),
        "f1_score": f1_score(y_test, y_pred_tuned, average="macro")
    }

    # Results
    logger.info(f"CNN Experiment Finished")
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