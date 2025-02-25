import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X[i]))
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]

    def predict(self, X, y):
        y_pred = []
        for x in X:
            x_with_bias = np.insert(x, 0, 1)  # Add bias term
            prediction = self.activation(np.dot(self.weights, x_with_bias))
            y_pred.append(prediction)
        return y_pred
# Define functions dynamically
Hidden_ly_output = [
    np.array([0, 0, 0, 1]),
    np.array([0, 0, 1, 0]),
    np.array([0, 1, 0, 0]),
    np.array([1, 0, 0, 0])
]

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = []

# Train perceptrons for each function dynamically i.e 4 neurons for 4 different inputs
for y in Hidden_ly_output:
    perceptron = Perceptron(input_size=2,epochs=15)
    perceptron.train(X, y)
    y_pred = perceptron.predict(X, y)
    predictions.append(y_pred)

# Convert predictions into input for final perceptron
final_X = np.array(predictions)

final_y = np.array([0, 1, 1, 0]) # XOR output

# Train final perceptron
final_perceptron = Perceptron(input_size=len(final_X),epochs=15)
final_perceptron.train(final_X, final_y)
final_predictions = final_perceptron.predict(final_X, final_y)

# Display XOR truth table with predictions
print("\nXOR Truth Table Predictions:")
print(" X1  X2 |  y_actual  y_pred")
print("---------------------------")
for i in range(len(X)):
    print(f" {X[i][0]}   {X[i][1]}  |     {final_y[i]}        {final_predictions[i]}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


accuracy = accuracy_score(final_y, final_predictions)
print(f"Final Perceptron Accuracy: {accuracy * 100:.2f}%")
print()

cm = confusion_matrix(final_y, final_predictions)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix for XOR using MLP")
plt.show()
print()


