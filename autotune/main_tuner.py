import os
import optuna
import pandas as pd
import matplotlib.pyplot as plt

from autotune.auto_detect import detect_framework, detect_model_type, detect_task_type
from autotune.hybrid_optimization import HybridOptimizer
from autotune.adaptive_search import AdaptiveSearchSpace
from autotune.resource_aware import ResourceAwareTuner
from autotune.search_space_config import MODEL_TYPE_TO_SPACE

from autotune.utils.logger import get_logger
logger = get_logger(__name__)

class AutoTuneTuner:
    """
    The main user-facing class for managing the tuning process.
    """
    def __init__(self, model, objective_function):
        """
        A Constructor that stores the user's model and objective.
        """
        if not callable(objective_function):
            raise TypeError("The objective_function must be a callable function.")
        
        self.model = model
        self.objective_function = objective_function
        self.results = {}

    def tune(self, n_trials=50, adaptive=True, callbacks=None):
        # Stage 1: Detection:
        framework = detect_framework(self.model)
        model_type = detect_model_type(self.model)
        task_type = detect_task_type(self.model, framework=framework)

        if model_type == "unknown":
            raise ValueError("Model type could not be auto-detected. Please use a supported model.")
        
        logger.info(f"Detected model: {model_type}, task: {task_type}, framework: {framework}")

        # Stage 2: Build Initial Search Space:
        model_config = MODEL_TYPE_TO_SPACE.get(model_type, {})
        base_space = model_config.get("_base", {})
        task_specific_space = model_config.get(task_type, {})
        initial_space = {**base_space, **task_specific_space}

        if not initial_space:
            raise ValueError(f"No search space found for model '{model_type}' and task '{task_type}'.")
        
        # Stage 3: Adjust Space for Hardware:
        resource_tuner = ResourceAwareTuner()
        adjusted_space = resource_tuner.adjust(initial_space)
        
        # Stage 4: Dynamic Search
        exploration_trials = max(10, int(n_trials * 0.4))

        adaptive_search = AdaptiveSearchSpace(
            adjusted_space,
            top_k=exploration_trials,
            shrink_factor=0.95,       
            elite_fraction=0.4        
        )
        
        optimizer = HybridOptimizer(
            objective_function=self.objective_function,
            search_space=adaptive_search.current_space,
            n_trials=n_trials
        )

        logger.info(f"Starting optimization for {n_trials} trials.")
        logger.info(f"Adaptive Search enabled: {adaptive}")
        logger.info(f"Initial search space: {adaptive_search.current_space}")

        # Stage 5: Run Optimization with Callback(True/False)
        callbacks_to_run = []
        if adaptive:
            def adaptive_callback(study, _frozen_trial):
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                trial_results = [(t.params, t.value) for t in completed_trials]
                adaptive_search.update_space(trial_results)
                optimizer.search_space = adaptive_search.current_space
            callbacks_to_run.append(adaptive_callback)

        if callbacks:
            callbacks_to_run.extend(callbacks)

        best_params, best_score = optimizer.optimize(callbacks=callbacks_to_run)

        # Stage 6: Store and Display Results
        self.results = {
            "best_parameters": best_params,
            "best_score": best_score
        }
        logger.info(f"Optimization finished:")
        logger.info(f"Best Score (loss): {best_score:.4f}")
        logger.info(f"Best Parameters: {best_params}")

        return best_params, best_score

    def plot_metrics_comparison(self, metrics_before, metrics_after, save_path=None):
        metric_names = list(metrics_before.keys())
        before_scores = list(metrics_before.values())
        after_scores = list(metrics_after.values())

        for i, metric in enumerate(metric_names):
            plt.figure(figsize=(6, 4))
            
            plt.plot(['Before Tuning'], [before_scores[i]],
                     marker='o', linestyle='-', color='blue', label='Before Tuning')
            plt.plot(['After Tuning'], [after_scores[i]],
                     marker='x', linestyle='-', color='red', label='After Tuning')
            
            plt.plot(['Before Tuning', 'After Tuning'], [before_scores[i], after_scores[i]],
                     linestyle='--', color='gray', alpha=0.7, zorder=0)
            
            plt.title(f"{metric.capitalize()} Comparison")
            plt.ylabel(metric.capitalize())
            plt.grid(True)
            plt.legend()
            
            min_val = min(before_scores[i], after_scores[i])
            max_val = max(before_scores[i], after_scores[i])
            
            padding = (max_val - min_val) * 0.1
            if padding == 0:
                padding = 0.0001
            
            plt.ylim(bottom=max(0, min_val - padding), top=max_val + padding)
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/{metric}_comparison.png", dpi=300)
            plt.show()

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