import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from soudmininn import (
    Dense, ReLU, Dropout, BatchNorm,
    NeuralNetwork, Trainer, AdaGrad, SoftmaxCrossEntropy
)

def main():
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = NeuralNetwork(
        layers=[
            Dense(X_train.shape[1], 128),
            ReLU(),
            Dropout(0.5),
            Dense(128, 64),
            ReLU(),
            Dropout(0.3),
            Dense(64, 10)
        ],
        loss=SoftmaxCrossEntropy()
    )
    
    trainer = Trainer(model, AdaGrad(lr=0.01))

    print("Starting training...")
    trainer.fit(X_train, y_train, epochs=200, batch_size=32)
    print("Training complete.")

    accuracy = trainer.accuracy(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
