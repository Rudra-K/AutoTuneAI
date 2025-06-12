import numpy as np
import random
import psutil
import torch

class ResourceAwareTuner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.ram_gb = round(psutil.virtual_memory().total / 1e9, 2)

        print("[ResourceAware] Resource adaptation complete.")
        print(f"[ResourceAware] Detected specs: CPUs: {self.cpu_cores}, RAM: {self.ram_gb} GB, GPU: {self.device}")

    def get_optimal_params(self):
        if self.device == "cuda":
            batch_size = 128 if self.ram_gb > 16 else 64
            try:
                major_capability = torch.cuda.get_device_capability(0)[0]
                precision = "float16" if major_capability >= 7 else "float32"
            except Exception as e:
                print("[ResourceAware] Failed to get GPU capability, defaulting to float32")
                precision = "float32"
        else:  
            batch_size = 32 if self.cpu_cores >= 8 else 16
            precision = "float32"
        
        return {"batch_size": batch_size, "precision": precision}
    
    def adjust(self, search_space):
        adjusted_space = {}
        for param, value_range in search_space.items():
            if isinstance(value_range, tuple) and all(isinstance(v, (int, float)) for v in value_range):
                min_val, max_val = value_range

                if self.cpu_cores < 4 or self.ram_gb < 8:
                     # shrink the range by 40% if resources are limited
                    center = (min_val + max_val) / 2
                    half_range = (max_val - min_val) * 0.3
                    new_min = max(min_val, center - half_range)
                    new_max = min(max_val, center + half_range)
                    adjusted_space[param] = (new_min, new_max)
                else:
                    adjusted_space[param] = value_range
            else:
                adjusted_space[param] = value_range  # non-numeric param (e.g., categorical)

        return adjusted_space


class HyperparameterTuner:
    def __init__(self, objective_function, search_method="bayesian", n_trials=50):
        self.objective_function = objective_function
        self.search_method = search_method
        self.n_trials = n_trials
        self.trial_results = []
        self.resource_tuner = ResourceAwareTuner()

    def suggest_params(self):
        resource_params = self.resource_tuner.get_optimal_params()
        if self.search_method == "random":
            return {
                "learning_rate": 10 ** random.uniform(-5, -2),
                "batch_size": resource_params["batch_size"],
                "precision": resource_params["precision"],
            }
        elif self.search_method == "bayesian":
            return {
                "learning_rate": np.random.uniform(1e-5, 1e-2),
                "batch_size": resource_params["batch_size"],
                "precision": resource_params["precision"],
            }
        else:
            raise ValueError("Invalid search method")

    def optimize(self):
        for trial in range(self.n_trials):
            params = self.suggest_params()
            score = self.objective_function(params)
            self.trial_results.append((params, score))
            print(f"Trial {trial+1}/{self.n_trials}: Params={params}, Score={score}")
        
        best_params = min(self.trial_results, key=lambda x: x[1])
        return best_params

