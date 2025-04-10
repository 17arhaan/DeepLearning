import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the dataset
df = pd.read_csv("./data/NaturalGasPrice/daily.csv")
df = df.dropna()
y = df['Price'].values
print(f"Total number of price records: {len(y)}")

# Normalize the prices to range [0, 1]
minm = y.min()
maxm = y.max()
print(f"Min price: {minm}, Max price: {maxm}")
y_norm = (y - minm) / (maxm - minm)

# Create sequences (last 10 days → predict next)
sequence_length = 10
X, Y = [], []
num_samples = min(5900, len(y_norm) - sequence_length - 1)
for i in range(num_samples):
    X.append(y_norm[i:i + sequence_length])
    Y.append(y_norm[i + sequence_length])
X = np.array(X)
Y = np.array(Y)
print(f"Total samples created: {len(X)}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)
print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

# Dataset class
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Loaders
batch_size = 256
train_loader = DataLoader(NGTimeSeries(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(NGTimeSeries(x_test, y_test), batch_size=batch_size, shuffle=False)

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc1(torch.relu(out))
        return out

model = RNNModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 1500

# Train
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.view(-1, sequence_length, 1).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {running_loss/len(train_loader):.6f}")

print("Training complete.")

# Evaluate on test data
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.view(-1, sequence_length, 1).to(device)
        outputs = model(inputs).view(-1)
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Plot: Normalized
plt.figure(figsize=(12, 6))
plt.plot(all_preds, label='Predicted (Normalized)')
plt.plot(all_targets, label='Actual (Normalized)')
plt.title('Test Set: Normalized Price Prediction')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()

# Plot: Original scale
all_preds_orig = np.array(all_preds) * (maxm - minm) + minm
all_targets_orig = np.array(all_targets) * (maxm - minm) + minm

plt.figure(figsize=(12, 6))
plt.plot(all_preds_orig, label='Predicted Price')
plt.plot(all_targets_orig, label='Actual Price')
plt.title('Test Set: Natural Gas Price Prediction (Original Scale)')
plt.xlabel('Sample Index')
plt.ylabel('Price (Nominal Dollars)')
plt.legend()
plt.show()
