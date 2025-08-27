import psutil
import torch

from autotune.utils.logger import get_logger
logger = get_logger(__name__)

class ResourceAwareTuner:
    """
    Inspects the user's hardware and adjusts the hyperparameter search space
    to be more efficient and to avoid overwhelming the machine.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.ram_gb = round(psutil.virtual_memory().total / 1e9, 2)
        self.is_low_spec = self.cpu_cores < 4 or self.ram_gb < 8

        logger.info(f"[ResourceAware] Detected specs: CPUs: {self.cpu_cores}, RAM: {self.ram_gb} GB, GPU: {self.device}")

    def adjust(self, search_space):
        """
        Adjusts the search space based on available system resources.
        """
        adjusted_space = search_space.copy()

        if 'batch_size' in adjusted_space:
            if self.device == "cuda":
                max_bs = 64 if self.ram_gb < 16 else 128
            else:
                max_bs = 16 if self.is_low_spec else 32
            
            bs_config = adjusted_space['batch_size']
            original_max = bs_config[2]

            # Use of min() to ensure we don't go above the user's original max
            adjusted_space['batch_size'] = (bs_config[0], bs_config[1], min(original_max, max_bs))
            logger.info(f"[ResourceAware] Adjusted 'batch_size' range to a max of {min(original_max, max_bs)}")

        # Shrink all numeric ranges if the system is low-spec
        if self.is_low_spec:
            logger.info("[ResourceAware] Low-spec system detected. Shrinking general search ranges.")
            for param, config in adjusted_space.items():
                if config[0] in ['float', 'int']:
                    param_type, min_val, max_val = config[0], config[1], config[2]
                    
                    center = (min_val + max_val) / 2
                    half_range = (max_val - min_val) * 0.3 # New half-range is 30% of original
                    
                    new_min = max(min_val, center - half_range)
                    new_max = min(max_val, center + half_range)

                    new_config = (param_type, new_min, new_max)
                    if len(config) == 4:
                        new_config += (config[3],)
                    
                    adjusted_space[param] = new_config
        
        return adjusted_space