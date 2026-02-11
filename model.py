import torch
import torch.nn as nn


class SalesForecastLSTM(nn.Module):
    """2-layer stacked LSTM for monthly sales forecasting.

    Takes a sequence of feature vectors and predicts the next month's sales.
    """

    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output
