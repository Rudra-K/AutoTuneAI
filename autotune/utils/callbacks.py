import optuna
from autotune.utils.logger import get_logger

logger = get_logger(__name__)

class FormattedTrialCallback:
    """A custom Optuna callback for clean, formatted trial output."""
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        latest_trial = study.trials[-1]
        
        print(f"\n[Trial {latest_trial.number} Finished]")
        print(f"  Value (loss): {latest_trial.value:.6f}")
        print("  Params: {")
        for key, value in latest_trial.params.items():
            print(f"    '{key}': {value},")
        print("  }")
        
        if latest_trial.value is not None and latest_trial.value == study.best_value:
            print(f"  Status: New Best!")
        
        print("-" * 50)