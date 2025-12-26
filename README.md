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
		- `batchnorm.py` — batch normalization layer.
		- `activations.py` — activation functions (ReLU, Sigmoid, Softmax helpers, etc.).
		- `base_layer.py` — shared layer base class and utilities.
	- `losses/` — loss implementations and helpers (MSE, softmax cross-entropy, base loss class).
	- `optimizers/` — optimizer implementations (SGD, Momentum, Adagrad, Adam and a base optimizer interface).

Installation
------------

Install from source in a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e .
```

Or install dependencies manually:

```bash
pip install numpy
```

Quick start (illustrative)
--------------------------

The following example shows a typical high-level workflow. The exact class and function names are in the `SoudMiniNN` modules; use this snippet as a template and consult the module docstrings for precise APIs.

```python
import numpy as np
from SoudMiniNN.network import Network
from SoudMiniNN.trainer import Trainer
from SoudMiniNN.layers.dense import Dense
from SoudMiniNN.layers.activations import ReLU, Softmax
from SoudMiniNN.losses.mse_func import MSE
from SoudMiniNN.optimizers.sgd import SGD

# Create a small dataset
X = np.random.randn(100, 20)
y = np.random.randint(0, 2, size=(100, 1))

# Build a simple model
model = Network([
		Dense(20, 64), ReLU(),
		Dense(64, 32), ReLU(),
		Dense(32, 1),
])

# Choose loss and optimizer
loss = MSE()
opt = SGD(lr=0.01)

# Train with a trainer helper
trainer = Trainer(model, loss, opt)
trainer.fit(X, y, epochs=100, batch_size=16)
```

Notes on API and extensibility
------------------------------

- Layers implement forward and backward methods and expose parameters and gradients for optimizers.
- Loss classes compute scalar loss and gradients with respect to outputs.
- Optimizers take model parameters and update them using their internal state.
- The codebase is intentionally small so you can add layers, new regularizers, or alternative optimizers easily.

Contributing and development
----------------------------

- Please open issues or pull requests for bug fixes, new features, or documentation improvements.
- Follow standard Python development flows: create a branch, add focused commits, and include tests where relevant.
- To run examples or quick experiments, run Python scripts from the repository root while your virtual environment is active.

Testing and formatting
----------------------

This repository is minimal and may not include an automated test harness. For development, consider adding `pytest` and basic unit tests for layers, losses, and optimizers.

License
-------

This project does not include an explicit license file. If you intend to publish or share, add a `LICENSE` file (MIT, Apache 2.0, or another preferred license) to make reuse terms explicit.

Further reading
---------------

- Study the implementations in `SoudMiniNN/layers`, `SoudMiniNN/losses`, and `SoudMiniNN/optimizers` to understand the mechanics of forward/backward propagation and parameter updates.
- Use the package as a learning tool before moving to feature-rich frameworks like PyTorch or TensorFlow for production workloads.

Questions or next steps
----------------------

If you want, I can:
- add runnable example scripts under `examples/` that demonstrate training on toy datasets,
- create unit tests for core components, or
- add a `requirements.txt` and CI configuration for automated testing.