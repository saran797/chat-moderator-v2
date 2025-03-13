
import torch
import torch.nn as nn

class BiLSTMFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTMFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # No classification layer
        features = lstm_out[:, -1, :]  # Extract features from the last time step
        return features  # Features shape: [batch_size, hidden_dim * 2]



