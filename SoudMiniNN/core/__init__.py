import numpy as np

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    # Kaiming/He normal
    return np.random.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(2.0 / fan_in)

def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    # Xavier/Glorot normal
    return np.random.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(1.0 / fan_in)
