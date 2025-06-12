from autotune.search_space_config import MODEL_TYPE_TO_SPACE

def make_sampler(model_type):
    if model_type not in MODEL_TYPE_TO_SPACE:
        raise ValueError(f"Model type '{model_type}' not found in MODEL_TYPE_TO_SPACE")

    def sampler(trial):
        space = MODEL_TYPE_TO_SPACE[model_type]
        params = {}
        for name, val_range in space.items():
            if isinstance(val_range, tuple) and len(val_range) == 2:
                low, high = val_range
                if isinstance(low, float) or isinstance(high, float):
                    # Log sampling for small continuous ranges like learning_rate
                    if "lr" in name.lower() or "learning_rate" in name.lower():
                        params[name] = trial.suggest_float(name, low, high, log=True)
                    else:
                        params[name] = trial.suggest_float(name, low, high)
                else:
                    params[name] = trial.suggest_int(name, low, high)
            else:
                raise ValueError(f"Unsupported range for {name}: {val_range}")
        return params

    return sampler
