import torch
import torch.nn as nn
from model import SalesForecastLSTM
from preprocessing import FEATURE_COLUMNS


def train_model(
    data: dict,
    max_epochs: int = 150,
    lr: float = 0.001,
    weight_decay: float = 1e-3,
    early_stop_patience: int = 10,
    progress_callback=None,
) -> dict:
    """Train the LSTM model with Huber loss and AdamW optimizer.

    Args:
        data: Output from prepare_training_data()
        max_epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: AdamW weight decay
        early_stop_patience: Stop if val loss doesn't improve for N epochs
        progress_callback: Optional callable(epoch, max_epochs, train_loss, val_loss)

    Returns:
        Dict with trained model, loss histories, and best epoch info.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    input_size = X_train.shape[2]
    model = SalesForecastLSTM(input_size=input_size)

    criterion = nn.SmoothL1Loss()  # Huber Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        scheduler.step(val_loss.item())

        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if progress_callback:
            progress_callback(epoch, max_epochs, train_loss.item(), val_loss.item())

        if patience_counter >= early_stop_patience:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(train_losses),
    }


def forecast(
    model: SalesForecastLSTM,
    df,
    data: dict,
    horizon: int,
) -> list[float]:
    """Direct multi-step forecast.

    Single forward pass predicts all 12 months, then slice first `horizon` months.
    Horizon can be 1, 3, 6, or 12.
    """
    model.eval()
    feature_scaler = data["feature_scaler"]
    target_scaler = data["target_scaler"]
    seq_len = data["sequence_length"]

    # Get the last sequence_length rows of scaled features
    features = df[FEATURE_COLUMNS].values
    scaled_features = feature_scaler.transform(features)
    window = scaled_features[-seq_len:]

    with torch.no_grad():
        x = torch.FloatTensor(window).unsqueeze(0)
        scaled_preds = model(x).numpy()[0]  # shape: (12,)

    # Inverse transform each prediction
    predicted_sales = []
    for val in scaled_preds[:horizon]:
        actual = target_scaler.inverse_transform([[val]])[0, 0]
        actual = max(actual, 0)  # sales can't be negative
        predicted_sales.append(actual)

    return predicted_sales
