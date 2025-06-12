import numpy as np

class AdaptiveSearchSpace:
    def __init__(self, initial_space, top_k=5, shrink_factor=0.8):
        self.original_space = initial_space
        self.current_space = initial_space.copy()
        self.top_k = top_k
        self.shrink_factor = shrink_factor

    def update_space(self, trial_results):
        if len(trial_results) < self.top_k:
            return self.current_space

        top_trials = sorted(trial_results, key=lambda x: x[1])[:self.top_k]

        new_space = {}
        for param in self.original_space:
            values = [trial[0][param] for trial in top_trials if isinstance(self.original_space[param], tuple)]
            if not values:
                new_space[param] = self.original_space[param]
                continue

            min_val = min(values)
            max_val = max(values)
            range_center = (min_val + max_val) / 2
            range_half = (max_val - min_val) / 2 * self.shrink_factor

            new_min = max(range_center - range_half, self.original_space[param][0])
            new_max = min(range_center + range_half, self.original_space[param][1])

            new_space[param] = (new_min, new_max)

        self.current_space = new_space
        return new_space

    def reset(self):
        self.current_space = self.original_space.copy()
