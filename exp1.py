import numpy as np
from sklearn.metrics import confusion_matrix

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0  # Step function

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias term
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = y[i] - y_pred
                self.weights += self.lr * error * x_i  # Update rule

    def evaluate(self, X, y):
        y_pred = [self.predict(x) for x in X]
        accuracy = np.mean(np.array(y_pred) == np.array(y)) * 100
        conf_matrix = confusion_matrix(y, y_pred)
        return accuracy, conf_matrix

# NAND Truth Table
X_NAND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_NAND = np.array([1, 1, 1, 0])  # NAND output

# XOR Truth Table
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([0, 1, 1, 0])  # XOR output

# Train Perceptron for NAND
perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X_NAND, y_NAND)
accuracy_nand, conf_matrix_nand = perceptron_nand.evaluate(X_NAND, y_NAND)

# Train Perceptron for XOR
perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X_XOR, y_XOR)
accuracy_xor, conf_matrix_xor = perceptron_xor.evaluate(X_XOR, y_XOR)

# Display Results
print(f"NAND Perceptron Accuracy: {accuracy_nand}%")
print("Confusion Matrix for NAND:\n", conf_matrix_nand)

print(f"\nXOR Perceptron Accuracy (Expected to fail): {accuracy_xor}%")
print("Confusion Matrix for XOR:\n", conf_matrix_xor)

# Test Predictions
print("\nTesting NAND Perceptron:")
for x in X_NAND:
    print(f"Input: {x}, Output: {perceptron_nand.predict(x)}")

print("\nTesting XOR Perceptron:")
for x in X_XOR:
    print(f"Input: {x}, Output: {perceptron_xor.predict(x)}")
