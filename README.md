SoudMiniNN — Minimal NumPy Neural Network (package: `soudmininn`)

Compact, educational neural-network library implemented with NumPy. The project is intended for learning and experimentation and favors clarity and simplicity over production performance.

**Repository layout**
- Package: [soudmininn](soudmininn)
  - `network.py` — model assembly and orchestration
  - `trainer.py` — simple training loop utilities
  - `layers/` — layer modules (`dense.py`, `dropout.py`, `batchnorm.py`, `activations.py`, `base_layer.py`)
  - `losses/` — loss modules (`mse_func.py`, `softmax_func.py`, `base_loss.py`)
  - `optimizers/` — optimizers (`sgd.py`, `momentum.py`, `adagrad.py`, `adam.py`, `base_optimizer.py`)
- `examples/` — example scripts (see [examples/iris_classification.py](examples/iris_classification.py))
- Project metadata at root: [setup.py](setup.py), `pyproject.toml`, `requirements.txt`

Note: the distribution name in `setup.py` is `SoudMiniNN` (project/distribution name), while the importable package folder is `soudmininn`.

**Key features**
- NumPy-only implementation (dependency: `numpy`)
- Modular layers, loss functions, and optimizers for experimentation
- Implemented optimizers: SGD, Momentum, Adagrad, Adam
- Trainer utilities and an iris classification example included

**Install & run**
Create and activate a virtual environment, then install dependencies.

# SoudMiniNN

Minimal educational neural-network library implemented with NumPy.
This repository provides a compact, from-scratch implementation useful for learning and experimentation.

## Quick start

1. Create and activate a virtual environment:

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run an example (iris classification):
```powershell
python examples/iris_classification.py
```

## What’s in this repo

- Package: `soudmininn`
  - `network.py` — model and forward/backward orchestration
  - `trainer.py` — basic training loop utilities
  - `layers/` — layer implementations:
    - `dense.py`, `dropout.py`, `batchnorm.py`, `activations.py`, `base_layer.py`
  - `losses/` — loss implementations: `mse_func.py`, `softmax_func.py`, `base_loss.py`
  - `optimizers/` — optimizers: `sgd.py`, `momentum.py`, `adagrad.py`, `adam.py`, `base_optimizer.py`
- `examples/` — runnable example scripts demonstrating classification and regression (see `examples/iris_classification.py`, `examples/mnist_classification.py`, etc.)
- Project metadata: `setup.py`, `pyproject.toml`, `requirements.txt`

Note: the distribution name in `setup.py` is `SoudMiniNN`, while the importable package is `soudmininn`.

## Highlights

- NumPy-only implementation (dependency: `numpy`).
- Modular design: add new layers, losses, or optimizers by following the existing module patterns.
- Included optimizers: SGD, Momentum, Adagrad, Adam.

## Example usage

Typical imports:

```python
from soudmininn.network import Network
from soudmininn.trainer import Trainer
from soudmininn.layers.dense import Dense
from soudmininn.layers.activations import ReLU, Softmax
from soudmininn.losses.mse_func import MSE
from soudmininn.optimizers.sgd import SGD
```

See the `examples/` folder for end-to-end scripts demonstrating how to build models, train them, and evaluate results.

## Development

- Add unit tests when extending functionality (recommended: `pytest`).
- Follow existing module patterns when adding layers, losses, or optimizers.

## License

See the `LICENSE` file at the project root.
