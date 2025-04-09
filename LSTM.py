import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Sample text (replace with your own text if desired)
text = "Hello there. This is a simple LSTM next character predictor. Enjoy!"

# Build vocabulary and mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

# Convert text to indices
data = np.array([char2idx[ch] for ch in text])

# Create sequences: use 10 characters to predict the next one
seq_length = 10
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)

X, Y = create_sequences(data, seq_length)

# Define a simple Dataset
class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = TextDataset(X, Y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a minimal LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=16, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # use output of last time-step
        return out, hidden

model = LSTMModel(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
model.train()
for epoch in range(100):
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output, _ = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
print("Training complete.")

# Text generation function
def generate_text(model, seed, length=100, temperature=1.0):
    model.eval()
    generated = seed
    # Ensure the seed is at least seq_length long
    inp = [char2idx[ch] for ch in seed[-seq_length:]]
    inp = torch.tensor(inp, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    for _ in range(length):
        output, hidden = model(inp, hidden)
        output = output / temperature
        probs = torch.softmax(output, dim=-1).cpu().detach().numpy().squeeze()
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx2char[next_idx]
        generated += next_char
        # Update input sequence
        new_input = torch.tensor([[next_idx]], dtype=torch.long).to(device)
        inp = torch.cat([inp[:, 1:], new_input], dim=1)
    return generated

seed_text = "Hello ther"  # ensure seed is at least `seq_length` characters long
print("\nGenerated Text:\n", generate_text(model, seed_text, length=100, temperature=0.8))
