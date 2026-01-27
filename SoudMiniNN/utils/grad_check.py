import numpy as np
from ..core.module import Module

def _relative_error(a: float, b: float, eps: float = 1e-12) -> float:
    return abs(a - b) / max(eps, abs(a) + abs(b))

def grad_check_module(
    module: Module,
    x: np.ndarray,
    dout: np.ndarray | None = None,
    eps: float = 1e-5,
    num_checks_per_param: int = 10,
    seed: int = 0
) -> dict:
    """
    Numerical gradient checking for Modules that implement forward/backward.

    This checks gradients of trainable Parameters only.
    It approximates d/dtheta sum(out * dout) using finite differences.

    Notes:
    - Avoid running grad-check with Dropout/BatchNorm in training mode.
      Prefer module.eval() for those, or test them separately.
    - This is a debugging tool, not used in training.

    Returns:
        report dict with max_relative_error and per-parameter details.
    """
    rng = np.random.default_rng(seed)

    out = module.forward(x)
    if dout is None:
        dout = rng.standard_normal(out.shape).astype(out.dtype)

    # analytic grads
    module.zero_grad()
    module.backward(dout)

    def scalar_out(o: np.ndarray) -> float:
        return float(np.sum(o * dout))

    details = []
    max_rel = 0.0

    for p in module.parameters():
        # pick random indices to check (keeps it fast)
        flat = p.data.reshape(-1)
        grad_flat = p.grad.reshape(-1)
        if flat.size == 0:
            continue

        idxs = rng.choice(flat.size, size=min(num_checks_per_param, flat.size), replace=False)
        for idx in idxs:
            old = float(flat[idx])

            flat[idx] = old + eps
            plus = scalar_out(module.forward(x))

            flat[idx] = old - eps
            minus = scalar_out(module.forward(x))

            flat[idx] = old  # restore

            num = (plus - minus) / (2 * eps)
            ana = float(grad_flat[idx])

            rel = _relative_error(num, ana)
            max_rel = max(max_rel, rel)
            details.append({
                "param_shape": p.data.shape,
                "index": int(idx),
                "numeric": num,
                "analytic": ana,
                "relative_error": rel
            })

    return {
        "max_relative_error": max_rel,
        "checks": details
    }
