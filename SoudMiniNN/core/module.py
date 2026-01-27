from __future__ import annotations
import numpy as np  # type: ignore
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union
from .parameter import Parameter

def _iter_params(obj: Any) -> Iterator[Tuple[str, Parameter]]:
    """
    Recursively iterate Parameters inside objects.
    Supports:
      - Parameter
      - Module
      - list/tuple
      - dict
    """

    if isinstance(obj, Parameter):
        yield ("", obj)
    elif isinstance(obj, Module):
        for n, p in obj.named_parameters():
            yield (n, p)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            for n, p in _iter_params(item):
                yield (f"{i}{('.' + n) if n else ''}", p)
    elif isinstance(obj, dict):
        for k, item in obj.items():
            for n, p in _iter_params(item):
                yield (f"{k}{('.' + n) if n else ''}", p)

class Module:
    """
    PyTorch-like module (educational).
    - forward(x): compute output
    - backward(dout): compute gradient w.r.t. input and fill Parameter.grad
    - parameters(): list of trainable Parameters (recursive)
    """

    def __init__(self) -> None:
        self.training: bool = True

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def train(self) -> "Module":
        self.training = True
        for _, m in self.named_modules():
            m.training = True
        return self

    def eval(self) -> "Module":
        self.training = False
        for _, m in self.named_modules():
            m.training = False
        return self

    def named_modules(self) -> Iterator[Tuple[str, "Module"]]:
        """
        Yields (name, module) for child modules (recursive), excluding self.
        """
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield (name, value)
                for sub_name, sub_m in value.named_modules():
                    yield (f"{name}.{sub_name}", sub_m)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield (f"{name}.{i}", item)
                        for sub_name, sub_m in item.named_modules():
                            yield (f"{name}.{i}.{sub_name}", sub_m)
            elif isinstance(value, dict):
                for k, item in value.items():
                    if isinstance(item, Module):
                        yield (f"{name}.{k}", item)
                        for sub_name, sub_m in item.named_modules():
                            yield (f"{name}.{k}.{sub_name}", sub_m)

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """
        Yields (name, Parameter) for all parameters (recursive).
        """
        # Direct parameters on this module
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                yield (name, value)
            elif isinstance(value, (list, tuple, dict, Module)):
                # recurse: we treat the container itself as a subtree
                for sub_name, p in _iter_params(value):
                    if sub_name == "":
                        # value is a Parameter already (shouldn't happen here) or Module without name
                        continue
                    yield (f"{name}.{sub_name}", p)

    def parameters(self) -> List[Parameter]:
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # --- Serialization helpers (PyTorch-like) ---
    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Return a mapping from parameter name to a *copy* of its data.
        Only trainable Parameters are included (buffers stay external).
        """
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None:
        """
        Load weights from a state dict into existing Parameters.

        Args:
            state: dict of name -> ndarray (matching shapes)
            strict: if True, raise on missing/unexpected keys; otherwise ignore them.
        """
        current: Dict[str, Parameter] = {name: p for name, p in self.named_parameters()}

        missing = [k for k in current.keys() if k not in state]
        unexpected = [k for k in state.keys() if k not in current]

        if strict and (missing or unexpected):
            raise KeyError(f"load_state_dict strict mismatch. missing={missing}, unexpected={unexpected}")

        for name, param in current.items():
            if name not in state:
                continue  # non-strict: allow missing
            arr = np.asarray(state[name])
            if arr.shape != param.data.shape:
                raise ValueError(f"Shape mismatch for '{name}': state {arr.shape} vs param {param.data.shape}")
            # copy into existing array to keep optimizer references intact
            np.copyto(param.data, arr.astype(param.data.dtype, copy=False))

        # ignore unexpected keys if non-strict


def state_dict(self) -> Dict[str, np.ndarray]:
    """
    Return a dictionary mapping parameter names -> numpy arrays.
    Similar idea to torch.nn.Module.state_dict (educational).
    """
    sd: Dict[str, np.ndarray] = {}
    for name, p in self.named_parameters():
        sd[name] = p.data.copy()
    return sd

def load_state_dict(self, state: Dict[str, "np.ndarray"], strict: bool = True) -> None:
    """
    Load parameters from a state dict produced by state_dict().

    Args:
        state: mapping name -> array
        strict: if True, error on missing/unexpected keys.
    """
    current = dict(self.named_parameters())
    missing = []
    unexpected = []

    for k, arr in state.items():
        if k in current:
            p = current[k]
            if p.data.shape != arr.shape:
                raise ValueError(f"Shape mismatch for '{k}': expected {p.data.shape}, got {arr.shape}")
            p.data[...] = arr
        else:
            unexpected.append(k)

    for k in current.keys():
        if k not in state:
            missing.append(k)

    if strict and (missing or unexpected):
        raise KeyError(f"load_state_dict strict=True: missing={missing}, unexpected={unexpected}")
