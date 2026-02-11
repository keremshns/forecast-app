import torch
import torch.nn as nn


class SalesForecastLSTM(nn.Module):
    """Encoder-Decoder LSTM for monthly sales forecasting.

    Encoder (Conv1D + stacked LSTM) processes historical sequence.
    Decoder (LSTMCell) generates future months autoregressively with
    future month features (month_sin, month_cos) for seasonal awareness.

    Architecture:
        Encoder:
            Input (seq_len, input_size)
            → Conv1D(64, kernel_size=2)
            → LSTM(64) → LSTM(64)
        Decoder:
            LSTMCell(3, 64) × max_horizon steps
            Input per step: [prev_prediction, month_sin, month_cos]
            → Linear(64, 1) per step
    """

    def __init__(self, input_size: int = 13, max_horizon: int = 12, dropout: float = 0.2):
        super().__init__()

        hidden_size = 64
        self.max_horizon = max_horizon

        # Encoder
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=2,
        )

        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        # Decoder: input = prev prediction (1) + month_sin (1) + month_cos (1) = 3
        self.decoder_cell = nn.LSTMCell(input_size=3, hidden_size=hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, future_month_feats: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) - historical features
            future_month_feats: (batch, max_horizon, 2) - month_sin/cos for future months
        """
        # Encode
        out = x.permute(0, 2, 1)
        out = self.conv1d(out)
        out = out.permute(0, 2, 1)

        out, _ = self.lstm1(out)
        out = self.dropout(out)

        _, (h, c) = self.lstm2(out)

        # Decode step-by-step
        h_dec = h.squeeze(0)  # (batch, hidden)
        c_dec = c.squeeze(0)  # (batch, hidden)

        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, 1, device=x.device)

        outputs = []
        for step in range(self.max_horizon):
            if future_month_feats is not None:
                month_feats = future_month_feats[:, step, :]  # (batch, 2)
            else:
                month_feats = torch.zeros(batch_size, 2, device=x.device)

            decoder_in = torch.cat([decoder_input, month_feats], dim=1)  # (batch, 3)

            h_dec, c_dec = self.decoder_cell(decoder_in, (h_dec, c_dec))
            h_dec = self.dropout(h_dec)
            pred = self.fc_out(h_dec)  # (batch, 1)
            outputs.append(pred)
            decoder_input = pred  # feed prediction as next input

        return torch.cat(outputs, dim=1)  # (batch, max_horizon)
