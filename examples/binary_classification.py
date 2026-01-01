from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from soudmininn import (
    Dense, Sigmoid, ReLU,
    NeuralNetwork, Trainer, Adam, MeanSquaredError
)

def main():
    X, y = load_breast_cancer(return_X_y=True)
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = NeuralNetwork(
        layers=[
            Dense(X_train.shape[1], 32),
            ReLU(),
            Dense(32, 16),
            ReLU(),
            Dense(16, 1,  "xavier"),
            Sigmoid()
        ],
        loss=MeanSquaredError()
    )

    trainer = Trainer(model, Adam(lr=0.001))

    print("Starting training...")
    trainer.fit(X_train, y_train, epochs=100, batch_size=32)
    print("Training complete.")
    
    accuracy = trainer.accuracy(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

