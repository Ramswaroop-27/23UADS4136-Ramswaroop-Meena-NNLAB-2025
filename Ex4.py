import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, accuracy_score

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape input data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)  # Flattening 28x28 images
x_test = x_test.reshape(-1, 784)

# One-hot encoding for labels
y_train_oh = np.eye(10)[y_train]
y_test_oh = np.eye(10)[y_test]

# Hyperparameter Configurations
hidden_layer_configs = [[160, 100], [100, 100], [100, 160], [60, 60], [100, 60]]
learning_rates = [0.001, 0.1, 1]
epochs = 10
batch_size = 32

# Function to create model
def create_model(hidden_layers):
    weights = []
    biases = []
    
    prev_layer_size = 784  # Input layer
    
    for layer_size in hidden_layers:
        weights.append(tf.Variable(tf.random.normal([prev_layer_size, layer_size], stddev=0.1)))
        biases.append(tf.Variable(tf.zeros([layer_size])))
        prev_layer_size = layer_size

    # Output layer
    weights.append(tf.Variable(tf.random.normal([prev_layer_size, 10], stddev=0.1)))
    biases.append(tf.Variable(tf.zeros([10])))

    return weights, biases

# Function for forward propagation
def forward_propagation(X, weights, biases):
    layer = X
    for i in range(len(weights) - 1):
        layer = tf.nn.relu(tf.matmul(layer, weights[i]) + biases[i])

    output = tf.matmul(layer, weights[-1]) + biases[-1]
    return output

# Loop through each configuration
results = {}

for layers in hidden_layer_configs:
    for lr in learning_rates:
        tf.compat.v1.reset_default_graph()
        
        # Define placeholders
        X = tf.compat.v1.placeholder(tf.float32, [None, 784])
        Y = tf.compat.v1.placeholder(tf.float32, [None, 10])

        # Create model
        weights, biases = create_model(layers)
        logits = forward_propagation(X, weights, biases)

        # Loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)

        # Accuracy metric
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Training
        loss_history = []
        accuracy_history = []
        start_time = time.time()

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(epochs):
                epoch_loss = 0
                epoch_acc = 0
                num_batches = x_train.shape[0] // batch_size
                
                for i in range(0, x_train.shape[0], batch_size):
                    batch_x = x_train[i:i+batch_size]
                    batch_y = y_train_oh[i:i+batch_size]
                    _, batch_loss, batch_acc = sess.run([optimizer, loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    epoch_loss += batch_loss
                    epoch_acc += batch_acc
                
                loss_history.append(epoch_loss / num_batches)
                accuracy_history.append(epoch_acc / num_batches)
                print(f"Layers: {layers}, LR: {lr}, Epoch {epoch+1}, Loss: {epoch_loss/num_batches:.4f}, Acc: {epoch_acc/num_batches:.4f}")

            # Evaluate on test set
            test_logits = sess.run(logits, feed_dict={X: x_test})
            test_preds = np.argmax(test_logits, axis=1)
            test_acc = accuracy_score(y_test, test_preds)

            # Confusion Matrix
            cm = confusion_matrix(y_test, test_preds)

        end_time = time.time()
        execution_time = end_time - start_time

        # Store results
        key = f"Layers: {layers}, LR: {lr}"
        results[key] = {
            "loss_history": loss_history,
            "accuracy_history": accuracy_history,
            "test_accuracy": test_acc,
            "execution_time": execution_time,
            "confusion_matrix": cm
        }

        print(f"\nTest Accuracy: {test_acc:.4f}, Execution Time: {execution_time:.2f} sec\n")

        # Plot Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Loss Curve', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss ({key})')
        plt.legend()
        plt.show()

        # Plot Accuracy Curve
        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_history, label='Accuracy Curve', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Training Accuracy ({key})')
        plt.legend()
        plt.show()

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix ({key})')
        plt.show()

# Print final results
print("\n==== FINAL RESULTS ====")
for key, values in results.items():
    print(f"{key}: Test Accuracy = {values['test_accuracy']:.4f}, Execution Time = {values['execution_time']:.2f} sec")
