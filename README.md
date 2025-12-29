SoudMiniNN — Minimal NumPy-based Neural Network Framework
=========================================================

Project overview
----------------

SoudMiniNN is a compact, educational neural-network framework implemented from scratch using NumPy. It provides a small set of composable layers, activation functions, loss functions, and optimizers so you can build, train, and experiment with simple feed-forward networks while keeping the implementation clear and readable.

This project is intended for learning and experimentation — it favors simplicity and clarity over performance or feature completeness.

Key features
------------

- Lightweight, dependency-minimal implementation (only NumPy).
- Modular layer and activation design for easy experimentation.
- Several optimization algorithms (SGD, Momentum, Adagrad, Adam).
- Basic trainers and loss functions (MSE, Softmax cross-entropy).
- Clear code structure suitable for teaching and extension.

Repository layout
-----------------

- `SoudMiniNN/` — main package
	- `network.py` — network/container utilities (model assembly and forward/backward orchestration).
	- `trainer.py` — training loop utilities and helpers.
	- `layers/` — layer implementations and activation modules:
		- `dense.py` — fully-connected (dense) layers.
		- `dropout.py` — dropout layer for regularization.
		### SoudMiniNN — Minimal NumPy-based Neural Network (package: `soudmininn`)

		Compact, educational neural-network library implemented from scratch with NumPy. Intended for learning and experimentation; clarity and simplicity are prioritized over performance.

		**Repository layout**
		- Package: [soudmininn](soudmininn)
			- `network.py` — model assembly and orchestration
			- `trainer.py` — simple training loop utilities
			- `layers/` — layer modules (`dense.py`, `dropout.py`, `batchnorm.py`, `activations.py`, `base_layer.py`)
			- `losses/` — loss modules (`mse_func.py`, `softmax_func.py`, `base_loss.py`)
			- `optimizers/` — optimizers (`sgd.py`, `momentum.py`, `adagrad.py`, `adam.py`, `base_optimizer.py`)
		- `examples/` — example scripts (see [examples/iris_classification.py](examples/iris_classification.py))
		- Root metadata: `pyproject.toml`, `requirements.txt`, `setup.py`

		**Key features**
		- NumPy-only implementation (dependency: `numpy`)
		- Modular layers, loss functions, and optimizers for experimentation
		- Implemented optimizers: SGD, Momentum, Adagrad, Adam
		- Trainer utilities and an iris classification example included

		**Install & run**
		Create and activate a virtual environment and install dependencies:

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
		- Follow existing module patterns when adding new layers, losses, or optimizers.
		- Consider adding `pytest` tests for layers and optimizers when extending the codebase.

		If you'd like, I can run the iris example here and confirm it executes.