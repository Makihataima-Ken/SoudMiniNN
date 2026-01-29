from __future__ import annotations
import numpy as np  # type: ignore
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union
from .parameter import Parameter


#forward-backward _ collect parms _train/eval mode _ zero_grad _ state_dict / load_state_dict 

def _iter_params(obj: Any) -> Iterator[Tuple[str, Parameter]]: 
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

class Module: # make the Libe same pytorch

    def __init__(self) -> None:
        self.training: bool = True # for check if the NN on eva phase or train why?? ex: Dropout on evaluate should be off

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x, *args, **kwargs): # like exactly pytorch / out = model(x) ==out = model.forward(x)
        return self.forward(x, *args, **kwargs)

    def train(self) -> "Module": # change sub_modules training status to ture train
        self.training = True
        for _, m in self.named_modules():
            m.training = True
        return self

    def eval(self) -> "Module": # like train but for eval
        self.training = False
        for _, m in self.named_modules():
            m.training = False
        return self

    def named_modules(self) -> Iterator[Tuple[str, "Module"]]: # to passeges for all sub_modules  to get all trees train/eval
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

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]: # get all parms in model because all opt need know the wight and parms
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

    def parameters(self) -> List[Parameter]: # return list of parms  
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None: # make all parms zero
        for p in self.parameters():
            p.zero_grad()

    # --- Serialization helpers (PyTorch-like) --- save
    def state_dict(self) -> Dict[str, np.ndarray]: # name : wight(data)
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None: #
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


