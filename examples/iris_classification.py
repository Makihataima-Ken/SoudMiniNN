from soudmininn import (
    Dense, Sigmoid, ReLU, BatchNorm,
    NeuralNetwork, Trainer, SGD, SoftmaxCrossEntropy
)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = NeuralNetwork(
    layers=[
        Dense(4, 16),
        Sigmoid(),
        BatchNorm(16,16),
        Dense(16, 8),
        ReLU(),
        Dense(8, 3)
    ],
    loss=SoftmaxCrossEntropy()
)

trainer = Trainer(model, SGD(lr=0.1))
trainer.fit(X_train, y_train, epochs=300)
