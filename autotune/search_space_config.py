MODEL_TYPE_TO_SPACE = {
    "linear": {
        "C": (0.1, 3.0),
        "max_iter": (100, 500),
        #"learning_rate": (1e-4, 1.0)
    },
    "logistic_regression": {
        "C": (0.01, 10.0),
        "max_iter": (100, 500)
    },
    "decision_tree": {
        "max_depth": (2, 30),
        "min_samples_split": (2, 20)
    },
    "random_forest": {
        "n_estimators": (50, 300),
        "max_depth": (5, 50),
        "min_samples_split": (2, 10)
    },
    "xgboost": {
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "n_estimators": (50, 200)
    },
    "lightgbm": {
        "learning_rate": (0.01, 0.3),
        "num_leaves": (20, 150),
        "n_estimators": (50, 500),
        "max_depth": (3, 15)
    },
    "catboost": {
        "learning_rate": (0.01, 0.3),
        "depth": (3, 10),
        "iterations": (100, 1000)
    },
    "svm": {
        "C": (0.1, 10.0),
        "gamma": (0.0001, 1.0)
    },
    "svc": {
        "C": (0.1, 10.0),
        "gamma": (0.0001, 1.0)
    },
    "mlp": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "alpha": (0.0001, 0.1),
        "learning_rate_init": (0.001, 0.1)
    },
    "mlp_classifier": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "alpha": (0.0001, 0.1),
        "learning_rate_init": (0.001, 0.1)
    },
    "mlp_regressor": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "alpha": (0.0001, 0.1),
        "learning_rate_init": (0.001, 0.1)
    },
    "cnn": {
        "num_filters": (16, 128),
        "kernel_size": (2, 5),
        "learning_rate": (0.0001, 0.01)
    },
    "resnet": {
        "learning_rate": (0.0001, 0.01),
        "batch_size": (16, 128)
    },
    "conv": {
        "num_filters": (16, 128),
        "kernel_size": (2, 5),
        "learning_rate": (0.0001, 0.01)
    },
    "transformer": {
        "num_layers": (2, 12),
        "num_heads": (2, 12),
        "learning_rate": (0.00001, 0.001)
    },
    "bert": {
        "num_layers": (2, 12),
        "num_heads": (2, 12),
        "learning_rate": (0.00001, 0.001)
    },
    "gpt": {
        "num_layers": (2, 24),
        "num_heads": (2, 16),
        "learning_rate": (0.00001, 0.001)
    },
    "rnn": {
        "hidden_size": (32, 256),
        "num_layers": (1, 3),
        "learning_rate": (0.0001, 0.01)
    },
    "lstm": {
        "hidden_size": (32, 256),
        "num_layers": (1, 3),
        "learning_rate": (0.0001, 0.01)
    },
    "gru": {
        "hidden_size": (32, 256),
        "num_layers": (1, 3),
        "learning_rate": (0.0001, 0.01)
    },
    "dqn": {
        "learning_rate": (0.0001, 0.01),
        "gamma": (0.8, 0.99),
        "epsilon": (0.01, 0.2)
    },
    "ppo": {
        "learning_rate": (0.0001, 0.01),
        "clip_range": (0.1, 0.3),
        "entropy_coef": (0.001, 0.05)
    },
    "a2c": {
        "learning_rate": (0.0001, 0.01),
        "value_loss_coef": (0.25, 1.0),
        "entropy_coef": (0.001, 0.05)
    }
}
