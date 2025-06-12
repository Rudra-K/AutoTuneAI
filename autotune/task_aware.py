import numpy as np
import random
import psutil
import torch

def detect_task(model):
    """Simple heuristic to detect model type: CNN, Transformer, RNN, RL."""
    model_name = model.__class__.__name__.lower()
    if "cnn" in model_name or "resnet" in model_name:
        return "cv"
    elif "transformer" in model_name or "bert" in model_name:
        return "nlp"
    elif "rnn" in model_name or "lstm" in model_name:
        return "rnn"
    elif "dqn" in model_name or "ppo" in model_name:
        return "rl"
    return "unknown"

class TaskAwareTuner:
    def __init__(self, model):
        self.task = detect_task(model)
    
    def get_task_params(self):
        """Suggests hyperparameter ranges based on detected task type."""
        if self.task == "cv":
            return {"learning_rate": (1e-4, 1e-2), "batch_size": (16, 128), "augmentations": ["flip", "rotate"]}
        elif self.task == "nlp":
            return {"learning_rate": (5e-5, 5e-3), "sequence_length": (128, 512), "tokenizer": ["BPE", "WordPiece"]}
        elif self.task == "rnn":
            return {"learning_rate": (1e-3, 1e-1), "hidden_size": (128, 1024), "num_layers": (1, 4)}
        elif self.task == "rl":
            return {"learning_rate": (1e-5, 1e-2), "gamma": (0.9, 0.99), "exploration": (0.1, 1.0)}
        else:
            return {"learning_rate": (1e-5, 1e-2), "batch_size": (16, 128)}  # Default case

class HyperparameterTuner:
    def __init__(self, objective_function, model, search_method="bayesian", n_trials=50):
        self.objective_function = objective_function
        self.search_method = search_method
        self.n_trials = n_trials
        self.trial_results = []
        self.task_tuner = TaskAwareTuner(model)

    def suggest_params(self):
        """Suggests hyperparameters based on task-aware tuning and search method."""
        task_params = self.task_tuner.get_task_params()
        
        if self.search_method == "random":
            return {key: random.uniform(*value) if isinstance(value, tuple) else random.choice(value) for key, value in task_params.items()}
        elif self.search_method == "bayesian":
            return {key: np.random.uniform(*value) if isinstance(value, tuple) else random.choice(value) for key, value in task_params.items()}
        else:
            raise ValueError("Invalid search method")

    def optimize(self):
        """Runs the hyperparameter tuning process."""
        for trial in range(self.n_trials):
            params = self.suggest_params()
            score = self.objective_function(params)
            self.trial_results.append((params, score))
            print(f"Trial {trial+1}/{self.n_trials}: Params={params}, Score={score}")
        
        best_params = min(self.trial_results, key=lambda x: x[1])
        return best_params
