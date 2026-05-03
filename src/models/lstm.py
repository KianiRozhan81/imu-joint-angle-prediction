import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        # bidirectional doubles the output size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)           # [batch, seq_len, hidden_size * 2]
        out = self.dropout(out)
        out = self.fc(out)              # [batch, seq_len, output_size]
        return out