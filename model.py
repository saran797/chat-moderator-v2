import torch

class BiLSTM(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # Add batch dimension
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        output = self.fc(lstm_out)
        return output
