import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from soudmininn import (
    Dense, ReLU,
    NeuralNetwork, Trainer, Momentum, MeanSquaredError
)

def main():

    X, y = fetch_california_housing(return_X_y=True)

    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    model = NeuralNetwork(
        layers=[
            Dense(X_train.shape[1], 64),
            ReLU(),
            Dense(64, 32),
            ReLU(),
            Dense(32, 1)
        ],
        loss=MeanSquaredError()
    )

    trainer = Trainer(model, Momentum(lr=0.001, momentum=0.9))

    print("Starting training...")
    trainer.fit(X_train, y_train, epochs=50, batch_size=64) 
    print("Training complete.")

    print("\nEvaluating model on test set...")
    scores = trainer.regression_score(X_test, y_test, scaler_y=scaler_y)

    print(f"  R-squared: {scores['r2_score']:.4f}")
    print(f"  Mean Absolute Error: ${scores['mae']:,.2f}")
    print(f"  Mean Squared Error: {scores['mse']:.4f}")


if __name__ == "__main__":
    main()
