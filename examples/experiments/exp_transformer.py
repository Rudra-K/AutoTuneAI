import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.metrics import accuracy_score

from autotune.utils.logger import get_logger
from examples.models.transformer_model import create_transformer_model

logger = get_logger(__name__)

def run_transformer_experiment(n_trials: int, epochs: int):
    logger.info("Starting Transformer Experiment:")
    
    # Data Prep for IMDb dataset
    vocab_size = 20000
    maxlen = 200
    (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_val = pad_sequences(x_val, maxlen=maxlen)

    # Objective Function
    def objective(trial):
        tf.keras.backend.clear_session()
        params = {
            'embed_dim': trial.suggest_categorical('embed_dim', [32, 64, 128]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'ff_dim': trial.suggest_categorical('ff_dim', [32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        }
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        model = create_transformer_model(maxlen=maxlen, vocab_size=vocab_size, **params)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy", metrics=["accuracy"])
        
        pruning_callback = TFKerasPruningCallback(trial, "val_accuracy")
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=epochs, batch_size=256, callbacks=[pruning_callback], verbose=0)
        
        _, accuracy = model.evaluate(x_val, y_val, verbose=0)
        return accuracy

    # Tuning
    logger.info(f"Starting tuning for {n_trials} trials...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
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