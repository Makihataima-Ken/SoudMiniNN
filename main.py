from sklearn.datasets import load_iris
from .layers.dense import Dense
from .layers.batchnorm import BatchNorm
from .layers.activations import ReLU, Sigmoid
from .losses.softmax_func import SoftmaxCrossEntropy
from .optimizers.sgd import SGD
from .network import NeuralNetwork
from .trainer import Trainer

def main():

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    model = NeuralNetwork(
        layers=[
            Dense(4, 16),
            Sigmoid(),
            BatchNorm(16),
            Dense(16, 8),
            ReLU(),
            Dense(8, 3)
        ],
        loss=SoftmaxCrossEntropy()
    )

    trainer = Trainer(model, SGD(lr=0.1))
    trainer.fit(X, y, epochs=200)
    
    train_acc = trainer.accuracy(model, X_train, y_train)
    test_acc = trainer.accuracy(model, X_test, y_test)

    print(f"Train Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    main()
