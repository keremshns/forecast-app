import torch
import torch.nn as nn


class SalesForecastLSTM(nn.Module):
    """Conv1D + Stacked LSTM + Dense layers for monthly sales forecasting.

    Direct multi-step: predicts all 12 future months at once (no recursive loop).

    Architecture:
        Input (seq_len, input_size)
        → Conv1D(30, kernel_size=2)
        → LSTM(30, return_sequences=True)
        → LSTM(30)
        → Dense(30, relu)
        → Dense(10, relu)
        → Dense(max_horizon, linear)
    """

    def __init__(self, input_size: int = 13, max_horizon: int = 12, dropout: float = 0.2):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=30,
            kernel_size=2,
        )

        self.lstm1 = nn.LSTM(input_size=30, hidden_size=30, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=30, hidden_size=30, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, max_horizon)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.permute(0, 2, 1)
        out = self.conv1d(out)
        out = out.permute(0, 2, 1)

        out, _ = self.lstm1(out)
        out = self.dropout(out)

        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout(out)

        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out
