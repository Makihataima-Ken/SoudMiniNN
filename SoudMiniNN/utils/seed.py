import random
import numpy as np # type: ignore

def set_seed(seed: int) -> None:
    """Set python + numpy seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
