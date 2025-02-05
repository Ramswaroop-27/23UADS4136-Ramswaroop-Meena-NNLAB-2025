import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        """
        Initialize the perceptron.

        Parameters:
            input_size (int): Number of input features.
            learning_rate (float): Learning rate for weight updates.
            epochs (int): Number of training iterations.
        """
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        """
        Step activation function.

        Parameters:
            x (float): Weighted sum of inputs.

        Returns:
            int: 1 if x >= 0, else 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, x):
        """
        Make a prediction for a single input.

        Parameters:
            x (array-like): Input features.

        Returns:
            int: Predicted class label (0 or 1).
        """
        z = np.dot(x, self.weights[1:]) + self.weights[0]  # Weighted sum + bias
        return self.activation_function(z)

    def fit(self, X, y):
        """
        Train the perceptron using the training data.

        Parameters:
            X (array-like): Feature matrix (shape: [n_samples, n_features]).
            y (array-like): Target labels (shape: [n_samples]).
        """
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                xi = X[i]
                target = y[i]
                prediction = self.predict(xi)

                # Update weights if prediction is incorrect
                update = self.learning_rate * (target - prediction)
                self.weights[1:] += update * xi  # Update weights
                self.weights[0] += update       # Update bias

    def evaluate(self, X, y):
        """
        Evaluate the perceptron on test data.

        Parameters:
            X (array-like): Feature matrix (shape: [n_samples, n_features]).
            y (array-like): True labels (shape: [n_samples]).

        Returns:
            float: Accuracy of the perceptron.
        """
        predictions = [self.predict(xi) for xi in X]
        accuracy = np.mean(predictions == y)
        return accuracy

# Example usage
if __name__ == "__main__":
    # Input data: AND gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    # Initialize and train perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
    perceptron.fit(X, y)

    # Test perceptron
    for xi, target in zip(X, y):
        prediction = perceptron.predict(xi)
        print(f"Input: {xi}, Target: {target}, Prediction: {prediction}")

    # Evaluate accuracy
    accuracy = perceptron.evaluate(X, y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
