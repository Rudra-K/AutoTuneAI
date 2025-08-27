# in examples/experiments/tune_cnn.py

import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from optuna.integration import TFKerasPruningCallback
import optuna

# Import our project's modules
from autotune.main_tuner import AutoTuneTuner
from autotune.utils.logger import get_logger
from examples.models.cnn_tf_model import create_cnn_model

logger = get_logger(__name__)

def run_cnn_experiment(n_trials: int, epochs: int, adaptive: bool):
    """
    A self-contained function to run the CNN tuning experiment on CIFAR-10.
    """
    logger.info("Starting CNN Experiment:")
    
    logger.info("Loading and preprocessing CIFAR-10 data:")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    def objective(trial):
        tf.keras.backend.clear_session()

        params = {
            'num_filters_1': trial.suggest_int('num_filters_1', 16, 64),
            'num_filters_2': trial.suggest_int('num_filters_2', 32, 128),
            'dense_units': trial.suggest_int('dense_units', 256, 1024),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        }

        model = create_cnn_model(**params)
        
        optimizer = getattr(tf.keras.optimizers, params['optimizer'])(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')

        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=128,
            callbacks=[pruning_callback],
            verbose=0 
        )

        _, accuracy = model.evaluate(x_val, y_val, verbose=0)
        return accuracy

    # Tuning
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    study.optimize(objective, n_trials=n_trials)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    logger.info("CNN Experiment Finished")
    logger.info(f"Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (accuracy): {trial.value:.4f}")
    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")