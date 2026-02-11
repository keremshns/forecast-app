import torch
import torch.nn as nn


class SalesForecastLSTM(nn.Module):
    """Conv1D + Stacked LSTM + Dense layers for monthly sales forecasting.

    Architecture (adapted from reference):
        Input (seq_len, input_size)
        → Conv1D(30, kernel_size=2)
        → LSTM(30, return_sequences=True)
        → LSTM(30)
        → Dense(30, relu)
        → Dense(10, relu)
        → Dense(1, linear)
    """

    def __init__(self, input_size: int = 13):
        super().__init__()

        # Conv1D: extracts local patterns from the sequence
        # PyTorch Conv1d expects (batch, channels, seq_len) so we permute in forward()
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=30,
            kernel_size=2,
        )

        # Stacked LSTM: first layer returns full sequence, second takes it
        self.lstm1 = nn.LSTM(input_size=30, hidden_size=30, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=30, hidden_size=30, batch_first=True)

        # Dense head
        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)

        # Conv1D expects (batch, channels, seq_len)
        out = x.permute(0, 2, 1)
        out = self.conv1d(out)            # (batch, 30, seq_len - 1)
        out = out.permute(0, 2, 1)        # (batch, seq_len - 1, 30)

        # LSTM layer 1 (return full sequence)
        out, _ = self.lstm1(out)           # (batch, seq_len - 1, 30)

        # LSTM layer 2 (take last hidden state)
        out, _ = self.lstm2(out)           # (batch, seq_len - 1, 30)
        out = out[:, -1, :]               # (batch, 30)

        # Dense layers
        out = self.relu(self.fc1(out))    # (batch, 30)
        out = self.relu(self.fc2(out))    # (batch, 10)
        out = self.fc3(out)               # (batch, 1)

        return out
