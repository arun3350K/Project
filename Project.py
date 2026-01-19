import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Generate Multivariate Time Series Dataset
np.random.seed(42)
n_samples = 1000
n_features = 5  # Multivariate (e.g., temp, humidity, pressure, wind, etc.)

# Generate correlated time series with trend + seasonality + noise
time = np.arange(n_samples)
data = np.zeros((n_samples, n_features))

for i in range(n_features):
    # Trend + seasonality + noise
    trend = 0.01 * time
    seasonal = 10 * np.sin(2 * np.pi * time / 365)
    noise = np.random.normal(0, 1, n_samples)
    data[:, i] = trend + seasonal + noise + i * 0.5  # Feature correlation

df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
df['target'] = data[:, 0] * 0.8 + data[:, 1] * 0.2 + np.random.normal(0, 0.5, n_samples)
print("Dataset shape:", df.shape)[file:61]

# Data Preprocessing
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict first feature (target)
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data_scaled, seq_length)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train), torch.FloatTensor(y_train)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_test), torch.FloatTensor(y_test)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)[file:61]

# LSTM + Attention Model
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size=1):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

# Initialize model
model = LSTMAttentionModel(input_size=n_features+1, hidden_dim=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)[file:61]

# Training with Hyperparameter Tuning
epochs = 50
train_losses = []

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred, _ = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

plt.plot(train_losses)
plt.title('LSTM-Attention Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()[file:61]

# Evaluation (MAE, RMSE)
model.eval()
y_pred_list, y_true_list = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred, attn_weights = model(X_batch)
        y_pred_list.extend(y_pred.squeeze().numpy())
        y_true_list.extend(y_batch.numpy())

y_pred = np.array(y_pred_list)
y_true = np.array(y_true_list)

# Inverse scale for evaluation
dummy = np.zeros((len(y_true), n_features+1))
dummy[:, 0] = y_true
y_true_inv = scaler.inverse_transform(dummy)[:, 0]

dummy[:, 0] = y_pred
y_pred_inv = scaler.inverse_transform(dummy)[:, 0]

mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y_true_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Predictions vs Actual')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Attention Weights (Sample)')
plt.imshow(attn_weights[0].numpy(), cmap='hot', aspect='auto')
plt.colorbar()
plt.show()

print("Complete LSTM-Attention time series forecasting pipeline executed successfully!")
print(f"Final Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}")[file:61]
