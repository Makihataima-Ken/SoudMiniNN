import numpy as np # type: ignore
from typing import Iterable
from ..core.parameter import Parameter

def clip_grad_norm_(parameters: Iterable[Parameter], max_norm: float, eps: float = 1e-12) -> float:
    """
    Clip gradients in-place to have global norm <= max_norm.
    Returns the (pre-clipping) total norm.

    Similar to torch.nn.utils.clip_grad_norm_ (educational).
    """
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0

    total_sq = 0.0
    for p in params:
        g = p.grad
        total_sq += float(np.sum(g * g))

    total_norm = float(np.sqrt(total_sq))
    if total_norm <= max_norm:
        return total_norm

    scale = max_norm / (total_norm + eps)
    for p in params:
        p.grad[...] = p.grad * scale
    return total_norm
