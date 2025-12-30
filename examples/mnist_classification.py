from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from soudmininn import (
    Dense, ReLU, BatchNorm,
    NeuralNetwork, Trainer, SGD, SoftmaxCrossEntropy
)

def main():

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    model = NeuralNetwork(
    layers=[
        Dense(784, 256),
        ReLU(),
        BatchNorm(256),
        Dense(256, 128),
        ReLU(),
        BatchNorm(128),
        Dense(128, 10),
    ],
    loss=SoftmaxCrossEntropy()
    )                   

    trainer = Trainer(model, SGD(lr=0.1))
    trainer.fit(X_train, y_train, epochs=50)

    print("Accuracy:", trainer.accuracy(X_test, y_test))
    

if __name__ == "__main__":
    main()