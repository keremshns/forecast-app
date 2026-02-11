import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import SalesForecastLSTM
from preprocessing import FEATURE_COLUMNS, build_forecast_features


def train_model(
    data: dict,
    max_epochs: int = 300,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 20,
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
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
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


def walk_forward_cv(
    df,
    n_folds: int = 3,
    max_epochs: int = 200,
    progress_callback=None,
) -> list[dict]:
    """Walk-forward (expanding window) cross-validation for time-series.

    Splits data chronologically into expanding train sets with fixed-size
    validation windows. Each fold trains a fresh model.

    Returns list of per-fold result dicts.
    """
    n_rows = len(df)

    # Determine sequence length (same logic as prepare_training_data)
    if n_rows >= 36:
        seq_len = 12
    elif n_rows >= 18:
        seq_len = 6
    else:
        seq_len = 3

    features = df[FEATURE_COLUMNS].values
    target = df[["sales"]].values

    # We need at least seq_len + 2 rows per fold (1 train sample + 1 val sample)
    min_train_size = seq_len + 2
    available = n_rows - min_train_size
    if available < n_folds:
        n_folds = max(available, 1)

    val_size = max(available // n_folds, 1)
    fold_results = []
    total_folds = n_folds

    for fold in range(n_folds):
        # Expanding window: train on everything up to val start
        val_end = n_rows - (n_folds - fold - 1) * val_size
        val_start = val_end - val_size
        train_end = val_start

        if train_end < min_train_size:
            continue

        # Scale on training portion only
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        train_features = feature_scaler.fit_transform(features[:train_end])
        train_target = target_scaler.fit_transform(target[:train_end])
        val_features = feature_scaler.transform(features[val_start:val_end])
        val_target = target_scaler.transform(target[val_start:val_end])

        # Build sequences for training
        X_train, y_train = [], []
        for i in range(seq_len, len(train_features)):
            X_train.append(train_features[i - seq_len : i])
            y_train.append(train_target[i])

        # Build sequences for validation (using training data as lookback)
        all_features = np.vstack([train_features, val_features])
        all_target = np.vstack([train_target, val_target])
        X_val, y_val = [], []
        for i in range(train_end, train_end + len(val_features)):
            if i >= seq_len:
                X_val.append(all_features[i - seq_len : i])
                y_val.append(all_target[i])

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        data = {
            "X_train": torch.FloatTensor(np.array(X_train)),
            "y_train": torch.FloatTensor(np.array(y_train)),
            "X_val": torch.FloatTensor(np.array(X_val)),
            "y_val": torch.FloatTensor(np.array(y_val)),
        }

        def fold_progress(epoch, max_ep, tl, vl):
            if progress_callback:
                progress_callback(fold, total_folds, epoch, max_ep, tl, vl)

        result = train_model(data, max_epochs=max_epochs, progress_callback=fold_progress)
        result["fold"] = fold + 1
        result["train_size"] = len(X_train)
        result["val_size"] = len(X_val)
        fold_results.append(result)

    return fold_results


def forecast(
    model: SalesForecastLSTM,
    df,
    data: dict,
    horizon: int,
) -> list[float]:
    """Generate recursive multi-step forecast.

    Predicts one month at a time, feeding each prediction back as input
    for the next step.
    """
    model.eval()
    feature_scaler = data["feature_scaler"]
    target_scaler = data["target_scaler"]
    seq_len = data["sequence_length"]

    # Get the last sequence_length rows of scaled features as the starting window
    features = df[FEATURE_COLUMNS].values
    scaled_features = feature_scaler.transform(features)
    window = list(scaled_features[-seq_len:])

    last_date = df["date"].iloc[-1]
    predicted_sales = []

    with torch.no_grad():
        for step in range(horizon):
            # Prepare input tensor from current window
            x = torch.FloatTensor(np.array(window[-seq_len:])).unsqueeze(0)
            scaled_pred = model(x).numpy()[0, 0]

            # Inverse transform to get actual sales value
            actual_pred = target_scaler.inverse_transform([[scaled_pred]])[0, 0]
            actual_pred = max(actual_pred, 0)  # sales can't be negative
            predicted_sales.append(actual_pred)

            # Build feature vector for next step
            next_features = build_forecast_features(
                df, last_date, predicted_sales, horizon, step
            )
            scaled_next = feature_scaler.transform(next_features)
            window.append(scaled_next[0])

    return predicted_sales
