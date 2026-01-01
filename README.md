SoudMiniNN — Minimal NumPy Neural Network (package: `soudmininn`)

Compact, educational neural-network library implemented with NumPy. The project is intended for learning and experimentation and favors clarity and simplicity over production performance.

**Repository layout**
# SoudMiniNN

Minimal, educational neural-network framework implemented from scratch using NumPy.
This repository is intended for learning and experimenting with core neural-network concepts.

## Install

Create and activate a virtual environment and install dependencies:

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or install the package in editable mode for development:
```powershell
pip install -e .
```

## Project layout

- Package: `soudmininn`
  - `network.py` — model assembly and orchestration
  - `trainer.py` — training loop utilities
  - `layers/` — layer implementations (`dense.py`, `dropout.py`, `batchnorm.py`, `activations.py`, `base_layer.py`)
  - `losses/` — loss functions (`mse_func.py`, `softmax_func.py`, `base_loss.py`, `softmax_func.py`)
  - `optimizers/` — optimizers (`sgd.py`, `momentum.py`, `adagrad.py`, `adam.py`, `base_optimizer.py`)
- `examples/` — example scripts: `iris_classification.py`, `mnist_classification.py`, `digit_classification.py`, `binary_classification.py`, `regression_california.py`
- Project metadata: `setup.py`, `pyproject.toml`, `requirements.txt`

The distribution name in `setup.py` is `SoudMiniNN`, while the importable package is `soudmininn`.

## Quick usage

1. Build a small model using the provided layers.
2. Choose a loss and optimizer and run the training loop in `Trainer`.

Example (see `examples/` for full scripts):
```python
from soudmininn.network import Network
from soudmininn.trainer import Trainer
from soudmininn.layers.dense import Dense
from soudmininn.layers.activations import ReLU, Softmax
from soudmininn.losses.softmax_func import SoftmaxCrossEntropy
from soudmininn.optimizers.sgd import SGD

# build model, trainer, then train using examples as a reference
```

## Examples

Run an example script from the repository root:
```powershell
python examples/iris_classification.py
python examples/mnist_classification.py
```

## Development notes

- Dependency: `numpy` (listed in `requirements.txt`)
- Follow module patterns when adding new layers, losses, or optimizers.
- Consider adding tests (`pytest`) when extending functionality.

## License

See the `LICENSE` file at the project root.
