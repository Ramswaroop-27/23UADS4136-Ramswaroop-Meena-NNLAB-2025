import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data for XOR function
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Expected output
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Set seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Random weights and biases
W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
b1 = np.random.uniform(-1, 1, (1, hidden_neurons))
W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
b2 = np.random.uniform(-1, 1, (1, output_neurons))

# Learning rate
learning_rate = 0.5

# Training iterations
epochs = 10000
errors = []

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    
    # Compute error
    error = Y - final_output
    errors.append(np.mean(np.abs(error)))
    
    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)
    
    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

# Testing the trained model
final_output = np.round(final_output)
print("\nFinal Outputs:")
print(final_output)

# Compute and print confusion matrix
cm = confusion_matrix(Y, final_output)
print("\nConfusion Matrix:")
print(cm)

# Plot loss curve
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Loss Curve')
plt.show()
