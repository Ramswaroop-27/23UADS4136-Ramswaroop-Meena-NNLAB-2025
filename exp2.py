import numpy as np 
import pandas as pd 
 
# Perceptron class with Gradient Descent 
class PerceptronGD: 
    def __init__(self, input_size, learning_rate=0.01, epochs=1000): 
        self.weights = np.zeros(input_size)  # Initialize weights as zeros 
        self.bias = 0  # Initialize bias as 0 
        self.learning_rate = learning_rate 
        self.epochs = epochs 
 
    # Activation function (Step function) 
    def activation(self, x): 
        return 1 if x >= 0 else 0 
 
    # Compute predictions for inputs X 
    def predict(self, X): 
        return np.array([self.activation(np.dot(inputs, self.weights) + self.bias) for inputs 
in X]) 
 
    # Training the perceptron using Gradient Descent 
    def train(self, X, y): 
        for _ in range(self.epochs): 
            for inputs, label in zip(X, y): 
                weighted_sum = np.dot(inputs, self.weights) + self.bias 
                prediction = self.activation(weighted_sum) 
                error = label - prediction 
 
                # Gradient descent update rule 
                self.weights += self.learning_rate * error * inputs 
                self.bias += self.learning_rate * error 
 
    # Convert to a more Pandas-friendly format for easy manipulation 
    def train_pandas(self, X, y): 
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]) 
        data['label'] = y 
 
        for _ in range(self.epochs): 
            for index, row in data.iterrows(): 
                inputs = row.drop('label').values 
                label = row['label'] 
                weighted_sum = np.dot(inputs, self.weights) + self.bias 
                prediction = self.activation(weighted_sum) 
                error = label - prediction 
 
                # Gradient descent update rule 
                self.weights += self.learning_rate * error * inputs 
                self.bias += self.learning_rate * error 
 
# Example usage 
if __name__ == "__main__": 
    # Training data (AND logic gate) 
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
    y = np.array([0, 0, 0, 1]) 
 
    # Create and train the perceptron using Gradient Descent 
    perceptron_gd = PerceptronGD(input_size=2) 
    perceptron_gd.train(X, y)  # Standard training using NumPy 
 
    # Test the perceptron 
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
    predictions = perceptron_gd.predict(test_input) 
 
    print("Predictions using NumPy:") 
    for i, prediction in enumerate(predictions): 
        print(f"Input: {test_input[i]} => Prediction: {prediction}") 
 
    # Create and train the perceptron using Gradient Descent with Pandas 
    perceptron_gd_pandas = PerceptronGD(input_size=2) 
    perceptron_gd_pandas.train_pandas(X, y)  # Training using Pandas 
 
    # Test the perceptron with Pandas approach 
    predictions_pandas = perceptron_gd_pandas.predict(test_input) 
 
    print("\nPredictions using Pandas:") 
    for i, prediction in enumerate(predictions_pandas): 
        print(f"Input: {test_input[i]} => Prediction: {prediction}")