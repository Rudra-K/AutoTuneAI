import optuna
import random

class BayesianOptimizer:
    def __init__(self, objective_function, n_trials=20):
        self.objective_function = objective_function
        self.n_trials = n_trials

    def optimize(self, search_space_sampler):
        def objective(trial):
            params = search_space_sampler(trial)
            return self.objective_function(params)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_trial


class RLAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        return random.choice(self.action_space)

    def update(self, action, reward):
        # Placeholder for RL training update
        pass


class HybridOptimizer:
    def __init__(self, objective_function, search_space_fn, sampler_fn, raw_space, n_trials=10):
        self.objective_function = objective_function
        self.search_space_fn = search_space_fn
        self.sampler_fn = sampler_fn
        self.raw_space = raw_space
        self.n_trials = n_trials


    def optimize(self):
        print("[HybridOptimizer] Starting with Bayesian Optimization...")
        bayes = BayesianOptimizer(self.objective_function, n_trials=self.n_trials // 2)
        bayes_result = bayes.optimize(self.sampler_fn)

        print("[HybridOptimizer] Switching to RL Agent for exploration...")

        # TEMPORARILY COMMENT OUT RL PART
        # raw_space = self.raw_space
        #
        # action_space = []
        # for _ in range(5):
        #     sample = {}
        #     for name, val_range in raw_space.items():
        #         if isinstance(val_range, tuple) and all(isinstance(v, (int, float)) for v in val_range):
        #             low, high = val_range
        #             sample[name] = random.uniform(low, high) if isinstance(low, float) else random.randint(low, high)
        #         elif isinstance(val_range, list):
        #             sample[name] = random.choice(val_range)
        #         else:
        #             raise ValueError(f"Unsupported parameter type for {name}: {val_range}")
        #     action_space.append(sample)
        #
        # rl_agent = RLAgent(action_space)
        #
        # best_score = bayes_result.value
        # best_params = bayes_result.params
        #
        # for _ in range(self.n_trials // 2):
        #     params = rl_agent.select_action()
        #     score = self.objective_function(params)
        #     rl_agent.update(params, score)
        #
        #     if score < best_score:
        #         best_score = score
        #         best_params = params

        # RETURN ONLY BAYESIAN RESULT
        return bayes_result.params, bayes_result.value


        #return best_params, best_score
