
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Load the dataset
url = 'airline-passengers.csv'
data = pd.read_csv(url)

# Only 'Passengers' column needed
passengers = data['Passengers'].values.astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
passengers = scaler.fit_transform(passengers.reshape(-1, 1))

# 2. Create sequences
def create_dataset(series, seq_length):
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_dataset(passengers, seq_length)

# Convert to tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
X = X.view(X.size(0), seq_length, 1)

# 3. Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Model setup
input_size = 1
hidden_size = 32
num_layers = 1
output_size = 1
model = RNNModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Train the model
num_epochs = 100
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    optimizer.zero_grad()
    loss = criterion(outputs.squeeze(), y.squeeze())
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # Calculate custom accuracy
    predicted = outputs.detach().numpy()
    actual = y.detach().numpy()

    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(actual)

    mae = np.mean(np.abs(predicted.flatten() - actual.flatten()))
    mean_actual = np.mean(actual.flatten())
    accuracy = 1 - (mae / mean_actual)

    accuracies.append(accuracy * 100)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Accuracy: {accuracy*100:.2f}%')

# 5. Final Evaluation
model.eval()
predicted = model(X).detach().numpy()
actual = y.detach().numpy()

predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(actual)

final_loss = np.mean((predicted.flatten() - actual.flatten())**2)
final_mae = np.mean(np.abs(predicted.flatten() - actual.flatten()))
final_mean_actual = np.mean(actual.flatten())
final_accuracy = 1 - (final_mae / final_mean_actual)

print(f"\nFinal MSE Loss: {final_loss:.4f}")
print(f"Final Custom Accuracy: {final_accuracy*100:.2f}%")

# 6. Plot everything
plt.figure(figsize=(14,8))

# Plot Predictions
plt.subplot(3,1,1)
plt.plot(actual, label='Actual Passengers')
plt.plot(predicted, label='Predicted Passengers')
plt.title('International Airline Passengers Prediction')
plt.xlabel('Months')
plt.ylabel('Passengers')
plt.legend()

# Plot Loss Curve
plt.subplot(3,1,2)
plt.plot(losses, label='Training Loss', color='red')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

# Plot Accuracy Curve
plt.subplot(3,1,3)
plt.plot(accuracies, label='Training Accuracy', color='green')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
