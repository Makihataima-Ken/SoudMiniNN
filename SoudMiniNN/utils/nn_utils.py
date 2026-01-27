import numpy as np # type: ignore
from ..core.parameter import Parameter
from typing import Iterable

def clip_grad_norm_(parameters: Iterable[Parameter], max_norm: float, eps: float = 1e-12) -> float:
    """
    In-place gradient clipping by global L2 norm (PyTorch-like).

    Returns:
        total_norm (float): L2 norm of all gradients before clipping.
    """
    params = list(parameters)
    grads = []
    for p in params:
        if p is None or not isinstance(p, Parameter):
            continue
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        grads.append(p.grad)

    if not grads:
        return 0.0

    # flatten and accumulate squared norms
    total_norm = np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads))

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for g in grads:
            g *= clip_coef

    return float(total_norm)
