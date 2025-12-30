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

    def fit(self, x, y, epochs=100):
        for e in range(epochs):
            loss = self.train_step(x, y)
            if e % 10 == 0:
                print(f"Epoch {e}, Loss: {loss:.4f}")
                
    def fit(self, x, y, epochs=100, batch_size=32):
        num_samples = x.shape[0]
        
        for e in range(epochs):
            epoch_loss = 0
            
            # Shuffle the data at the start of each epoch
            permutation = np.random.permutation(num_samples)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]
            
            # Iterate over mini-batches
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss * x_batch.shape[0] # Weight loss by batch size

            # Print average loss for the epoch
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