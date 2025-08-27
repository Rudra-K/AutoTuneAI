class AdaptiveSearchSpace:
    """
    Manages a search space that can dynamically shrink to focus on
    more promising regions as the tuning process progresses.
    """
    def __init__(self, initial_space, top_k=5, shrink_factor=0.8, elite_fraction=0.5):
        self.original_space = initial_space
        self.current_space = initial_space.copy()
        self.top_k = top_k
        self.shrink_factor = shrink_factor
        self.elite_fraction = elite_fraction # The fraction of the top_k to consider

    def update_space(self, trial_results):
        if len(trial_results) < self.top_k:
            return

        print(f"[AdaptiveSearch] Evaluating top {self.top_k} trials to shrink space:")

        top_trials = sorted(trial_results, key=lambda x: x[1])[:self.top_k]
        
        elite_count = max(1, int(len(top_trials) * self.elite_fraction))
        elite_trials = top_trials[:elite_count]
        
        top_params = [trial[0] for trial in elite_trials]

        new_space = {}
        for param_name, config in self.original_space.items():
            param_type = config[0]
            
            #for Numerical Parameters
            if param_type in ["float", "int"]:
                original_min, original_max = config[1], config[2]
                
                values = [p[param_name] for p in top_params if param_name in p]
                if not values:
                    new_space[param_name] = self.current_space[param_name]
                    continue

                min_val, max_val = min(values), max(values)
                
                #shrink the range around the best-performing values
                range_center = (min_val + max_val) / 2
                range_half = (max_val - min_val) / 2 * self.shrink_factor
                
                new_min = max(range_center - range_half, original_min)
                new_max = min(range_center + range_half, original_max)
                
                new_config = (param_type, new_min, new_max)
                if len(config) == 4:
                    new_config += (config[3],)
                
                new_space[param_name] = new_config
            
            #for Categorical Parameters
            elif param_type == "categorical":
                top_categories = {p[param_name] for p in top_params if param_name in p}
                if top_categories:
                    new_space[param_name] = (param_type, list(top_categories))
                else:
                    new_space[param_name] = self.current_space[param_name]
            else:
                new_space[param_name] = self.current_space[param_name]

        self.current_space = new_space
        print(f"[AdaptiveSearch] New shrunken space: {self.current_space}")

    def reset(self):
        """Resets the search space to its original state."""
        self.current_space = self.original_space.copy()