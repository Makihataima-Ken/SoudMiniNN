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

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the iris example:
```powershell
python examples/iris_classification.py
```

**Usage (imports)**
```python
from soudmininn.network import Network
from soudmininn.trainer import Trainer
from soudmininn.layers.dense import Dense
from soudmininn.layers.activations import ReLU, Softmax
from soudmininn.losses.mse_func import MSE
from soudmininn.optimizers.sgd import SGD
```

**Development notes**
- Add new layers, losses, or optimizers by following existing module patterns.
- Consider adding `pytest` tests for layers and optimizers when extending the codebase.