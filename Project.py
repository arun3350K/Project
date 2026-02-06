import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import math

# 1. Data Generation/Loading (Synthetic for demo; replace with pd.read_csv('sp500.csv')['Close'])
np.random.seed(42)
time = np.arange(1000)
data = np.sin(0.01 * time) * 20 + np.cumsum(np.random.normal(0, 1, 1000)) + 100  # Financial-like
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

def create_sequences(data, seq_len=60, pred_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

seq_len, pred_len = 60, 5
X, y = create_sequences(data_scaled)
split = int(0.8 * len(X))
X_train, X_test = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
y_train, y_test = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, 64, shuffle=True)
test_loader = DataLoader(test_ds, 64, shuffle=False)

# 2. LSTM Baseline
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# 3. Seq2Seq Attention Model
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Seq2SeqAttention(nn.Module):
    def __init__(self, hidden_size=50, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size * 2 + 1, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size)
        self.fc_out = nn.Linear(hidden_size * 3, 1)  # For multi-step, loop decoder
        self.hidden_size = hidden_size * 2  # Bidirectional
    
    def forward(self, src, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = torch.zeros(batch_size, 1, 1).to(src.device)
        outputs = torch.zeros(batch_size, pred_len, 1).to(src.device)
        for t in range(pred_len):
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            dec_input = torch.cat((context.unsqueeze(1), decoder_input), dim=2)
            dec_output, (hidden, cell) = self.decoder(dec_input, (hidden, cell))
            pred = self.fc_out(torch.cat((dec_output.squeeze(1), context), dim=1)).unsqueeze(2)
            outputs[:, t:t+1, :] = pred
            decoder_input = pred if np.random.random() < teacher_forcing_ratio else pred
        return outputs.squeeze(2), attn_weights

# 4. Training Function
def train_model(model, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            if isinstance(model, LSTMModel):
                y_pred = model(X_batch)
            else:
                y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader)

# 5. Evaluation
def evaluate(model):
    model.eval()
    preds, trues = [], []
    attn_ws = [] if hasattr(model, 'attention') else None
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if isinstance(model, LSTMModel):
                y_pred = model(X_batch)
            else:
                y_pred, attn = model(X_batch)
                if attn_ws is not None:
                    attn_ws.append(attn.mean(0).cpu())
            preds.append(y_pred.cpu())
            trues.append(y_batch.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    mape = np.mean(np.abs((trues - preds) / trues)) * 100
    return {'MSE': mse, 'MAE': mae, 'MAPE': mape}, preds, trues, attn_ws

# 6. Hyperparameter Tuning with Optuna
def objective(trial, model_class):
    hidden = trial.suggest_int('hidden_size', 32, 128)
    layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    model = model_class(hidden_size=hidden, num_layers=layers)
    train_model(model, epochs=30, lr=lr)
    metrics, _, _, _ = evaluate(model)
    return metrics['MSE']

# Run tuning (50 trials)
study_lstm = optuna.create_study()
study_lstm.optimize(lambda trial: objective(trial, LSTMModel), n_trials=20)
best_lstm = LSTMModel(**study_lstm.best_params)
train_model(best_lstm, lr=study_lstm.best_params['lr'])

study_attn = optuna.create_study()
study_attn.optimize(lambda trial: objective(trial, Seq2SeqAttention), n_trials=20)
best_attn = Seq2SeqAttention(**study_attn.best_params)
train_model(best_attn, lr=study_attn.best_params['lr'])

# 7. Results
lstm_metrics, lstm_pred, _, _ = evaluate(best_lstm)
attn_metrics, attn_pred, attn_true, attn_ws = evaluate(best_attn)

print("LSTM Metrics:", lstm_metrics)
print("Attention Metrics:", attn_metrics)

# 8. Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Predictions
axes[0,0].plot(attn_true.flatten()[:200], label='True')
axes[0,0].plot(lstm_pred[:200], label='LSTM')
axes[0,0].plot(attn_pred[:200], label='Attention')
axes[0,0].legend(); axes[0,0].set_title('Forecast Comparison')

# Metrics Table (print as text)
metrics_df = pd.DataFrame({**lstm_metrics, **{'LSTM_'+k: v for k,v in lstm_metrics.items()},
                           **attn_metrics, **{'Attn_'+k: v for k,v in attn_metrics.items()}})
print(metrics_df)

# Attention Heatmap (avg over batch)
if attn_ws:
    avg_attn = torch.stack(attn_ws).mean(0).numpy()
    sns.heatmap(avg_attn[:20, :], ax=axes[0,1], cmap='Blues')
    axes[0,1].set_title('Attention Weights')

plt.tight_layout()
plt.savefig('project_results.png')
plt.show()
