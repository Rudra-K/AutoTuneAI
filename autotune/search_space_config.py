MODEL_TYPE_TO_SPACE = {

    "logistic_regression": {
        "_base": {
            "C": ("float", 0.01, 100.0, "log"),
            "max_iter": ("int", 200, 2000)
        }
    },

    "random_forest": {
        "_base": {
            "n_estimators": ("int", 50, 500),    # Number of trees in the forest.
            "max_depth": ("int", 5, 50),         # Max depth of each tree.
            "min_samples_split": ("int", 2, 20)
        },
    },

    "lightgbm": {
        "_base": {
            "n_estimators": ("int", 50, 1000),
            "learning_rate": ("float", 0.01, 0.3, "log"),
            "num_leaves": ("int", 20, 150) # Controls the complexity of the tree.
        }
    },
    
    "svm": {
        "_base": {
            "C": ("float", 0.1, 100.0, "log"),
            "gamma": ("float", 0.0001, 1.0, "log")
        }
    },

    "cnn": {
        "_base": {
            "num_filters": ("int", 16, 128),
            "kernel_size": ("int", 3, 7),
            "dropout_rate": ("float", 0.1, 0.5, "linear")
        },
        "cv": {
            "learning_rate": ("float", 1e-5, 1e-2, "log"),
            "optimizer": ("categorical", ["adam", "sgd", "rmsprop"]),
            "batch_size": ("int", 16, 128)
        }
    },

    "transformer": {
        "_base": {
            "num_layers": ("int", 2, 12),
            "num_heads": ("int", 2, 16),      # Number of attention heads.
            "ff_dim": ("int", 512, 2048)    # feed-forward network's dimension.
        },
        "nlp": {
            "learning_rate": ("float", 1e-6, 1e-4, "log"),
            "optimizer": ("categorical", ["adamw", "sgd"]),
            "batch_size": ("int", 8, 64)
        }
    }
}