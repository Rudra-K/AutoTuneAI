from autotune.auto_detect import detect_model_type, detect_task_type
from autotune.hybrid_optimization import HybridOptimizer
from autotune.search_space import AdaptiveSearchSpace
from autotune.resource_aware import ResourceAwareTuner
from autotune.task_aware import TaskAwareTuner
from autotune.search_space_config import MODEL_TYPE_TO_SPACE
from autotune.sampler_conv import make_sampler
import matplotlib.pyplot as plt
import pandas as pd
import os

class AutoTuneTuner:
    def __init__(self, model, objective_function, config=None, search_method="bayesian", n_trials=50):
        self.model = model
        self.objective_function = objective_function
        self.search_method = search_method
        self.config = config
        self.n_trials = config["tuning"]["n_trials"] if config and "tuning" in config else n_trials

        self.model_type = detect_model_type(model)
        if self.model_type == "unknown":
            self.model_type = input("Couldn't auto-detect model type. Please enter it manually: ")

        self.task_type = detect_task_type(model)
        if self.task_type == "unknown":
            self.task_type = input("Couldn't auto-detect task type. Please enter it manually: ")

    def objective(self, params):
        return self.objective_function(params)

    def tune(self):
        print(f"[Tuner] ✅ Detected model type: {self.model_type}, task type: {self.task_type}")

        initial_space = MODEL_TYPE_TO_SPACE.get(self.model_type, {})
        adaptive_space = AdaptiveSearchSpace(initial_space)
        base_search_space = adaptive_space.current_space
        
        task_aware_space = TaskAwareTuner(self.model).get_task_params()

        # Combine base search space and task-aware adjustments
        combined_space = {**base_search_space, **task_aware_space}

        # Resource-aware tuning adjustments
        resource_tuner = ResourceAwareTuner()
        final_space = resource_tuner.adjust(combined_space)

        def search_space_structure():
            return {
                name: config[1:] if config[0] in ["float", "int"] else config[1]
                for name, config in final_space.items()
            }


        def search_space_sampler(trial):
            params = {}
            for name, config in final_space.items():
                param_type = config[0]
                if param_type == "float":
                    params[name] = trial.suggest_float(name, config[1], config[2])
                elif param_type == "int":
                    params[name] = trial.suggest_int(name, config[1], config[2])
                elif param_type == "categorical":
                    params[name] = trial.suggest_categorical(name, config[1])
            return params


        print("[Tuner] 🚀 Starting hybrid optimization...")
        print("[Debug] Final search space:", final_space)

        model_type = self.model_type  # dynamically detected earlier
        sampler_fn = make_sampler(model_type)

        raw_space = search_space_structure()

        optimizer = HybridOptimizer(
            objective_function=self.objective,
            search_space_fn=search_space_structure,
            sampler_fn=sampler_fn,
            raw_space=raw_space,
            n_trials=self.n_trials
        )

        best_params, best_score = optimizer.optimize()

        print(f"[Tuner] 🏆 Best score: {best_score}")
        print(f"[Tuner] 🧠 Best parameters: {best_params}")

        return best_params, best_score
    
    def plot_metrics_comparison(self, metrics_before, metrics_after, save_path=None):
        metric_names = list(metrics_before.keys())
        before_scores = [metrics_before[m] for m in metric_names]
        after_scores = [metrics_after[m] for m in metric_names]

        for i, metric in enumerate(metric_names):
            plt.figure(figsize=(6, 4))
            plt.plot(['Before Tuning', 'After Tuning'],
                    [before_scores[i], after_scores[i]],
                    marker='o', linestyle='-', color='blue', label='Before')
            plt.plot(['Before Tuning', 'After Tuning'],
                    [before_scores[i], after_scores[i]],
                    marker='x', linestyle='--', color='red', label='After')
            plt.title(f"{metric.capitalize()} Comparison")
            plt.ylabel(metric.capitalize())
            plt.grid(True)
            plt.legend()

            # 🔍 Dynamic Y-axis zoom based on values
            delta = abs(before_scores[i] - after_scores[i])
            if delta < 1e-4:  # very small difference
                delta = 0.01
            min_val = min(before_scores[i], after_scores[i]) - delta * 0.3
            max_val = max(before_scores[i], after_scores[i]) + delta * 0.3
            plt.ylim(max(0, min_val), min(1, max_val))

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/{metric}_comparison_zoom.png", dpi=300)
            plt.show()

        # Metric change table
        change_percent = [(after - before) / before * 100 if before != 0 else 0
                        for before, after in zip(before_scores, after_scores)]
        df = pd.DataFrame({
            "Metric": metric_names,
            "Before": before_scores,
            "After": after_scores,
            "Change (%)": [f"{v:+.2f}%" for v in change_percent]
        })

        print("\n=== Metric Change Summary ===\n")
        print(df.to_string(index=False))


