import numpy as np
from .optimizers.base_optimizer import Optimizer
from .network import NeuralNetwork

class Trainer:
    def __init__(self, model:NeuralNetwork, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model.forward(x)
        loss = self.model.loss_func.forward(logits, y)
        grad = self.model.loss_func.backward()
        self.model.backward(grad)

        params = self.model.get_params()
        grads = self.model.get_grads()
        self.optimizer.update(params, grads)

        return loss
                
    def fit(self, x, y, epochs=100, batch_size=32):
        num_samples = x.shape[0]
        
        for e in range(epochs):
            epoch_loss = 0
            
            permutation = np.random.permutation(num_samples)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss * x_batch.shape[0] 

            avg_epoch_loss = epoch_loss / num_samples
            if e % 10 == 0:
                print(f"Epoch {e}, Loss: {avg_epoch_loss:.4f}")
                
    def accuracy(self, X, y):
        logits = self.model.forward(X, training=False)
        if logits.shape[1] == 1:
            predictions = (logits > 0.5).astype(int)
        else:
            predictions = np.argmax(logits, axis=1)
        return np.mean(predictions == y)

    def predict(self, X):
        logits = self.model.forward(X, training=False)
        if logits.shape[1] == 1:
            predictions = (logits > 0.5).astype(int)
        else:
            predictions = np.argmax(logits, axis=1)
        return predictions
    
    def regression_score(self, X_test, y_test, scaler_y=None):
        """
        Calculates and returns common regression metrics (MAE, MSE, R^2)
        using the from-scratch implementations above.
        """
        y_pred = self.predict(X_test)

        y_true_final = y_test
        y_pred_final = y_pred

        if scaler_y is not None:
            y_true_final = scaler_y.inverse_transform(y_test)
            y_pred_final = scaler_y.inverse_transform(y_pred)
        else:
            print("Warning: No y-scaler provided. Metrics are calculated on scaled data.")

        scores = {
            'mae': self._mean_absolute_error(y_true_final, y_pred_final),
            'mse': self._mean_squared_error(y_true_final, y_pred_final),
            'r2_score': self._r2_score(y_true_final, y_pred_final)
        }
        return scores
    
    def _mean_squared_error(self,y_true, y_pred):
        """
        Calculates the Mean Squared Error from scratch using NumPy.
        Formula: (1/n) * Σ(y_true - y_pred)^2
        """
        return np.mean((y_true - y_pred) ** 2)

    def _mean_absolute_error(self,y_true, y_pred):
        """
        Calculates the Mean Absolute Error from scratch using NumPy.
        Formula: (1/n) * Σ|y_true - y_pred|
        """
        return np.mean(np.abs(y_true - y_pred))

    def _r2_score(self,y_true, y_pred):
        """
        Calculates the R-squared (Coefficient of Determination) score 
        from scratch using NumPy. A score of 1.0 is a perfect fit.
        Formula: 1 - (sum_squares_res / sum_squares_total)
        """
        sum_squares_res = np.sum((y_true - y_pred) ** 2)

        sum_squares_total = np.sum((y_true - np.mean(y_true)) ** 2)
    
        if sum_squares_total == 0:
            return 1.0 if sum_squares_res == 0 else 0.0
            
        return 1 - (sum_squares_res / sum_squares_total)
