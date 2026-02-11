import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def validate_and_parse_csv(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Validate uploaded CSV and extract date + sales columns.

    Expects exactly 2 columns: one parseable as dates, one numeric.
    Returns (dates, sales) as pandas Series.
    """
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (date and sales amount).")

    # Try to auto-detect date and sales columns
    date_col = None
    sales_col = None

    for col in df.columns:
        if date_col is None:
            try:
                pd.to_datetime(df[col])
                date_col = col
                continue
            except (ValueError, TypeError):
                pass

        if sales_col is None:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > len(df) * 0.5:
                sales_col = col

    if date_col is None:
        raise ValueError("Could not detect a date column. Ensure one column contains dates (e.g., 2024-01, Jan 2024).")
    if sales_col is None:
        raise ValueError("Could not detect a numeric sales column.")

    dates = pd.to_datetime(df[date_col])
    sales = pd.to_numeric(df[sales_col], errors="coerce")

    # Drop rows where either is NaN
    valid = dates.notna() & sales.notna()
    dates = dates[valid].reset_index(drop=True)
    sales = sales[valid].reset_index(drop=True)

    # Sort by date
    sort_idx = dates.argsort()
    dates = dates.iloc[sort_idx].reset_index(drop=True)
    sales = sales.iloc[sort_idx].reset_index(drop=True)

    if len(dates) < 24:
        raise ValueError(f"Need at least 24 months of data for reliable forecasting. Found {len(dates)} months.")

    return dates, sales


def engineer_features(dates: pd.Series, sales: pd.Series) -> pd.DataFrame:
    """Generate features from date and sales data.

    Returns a DataFrame with all features + the target 'sales' column.
    """
    df = pd.DataFrame({"date": dates, "sales": sales})

    # Cyclical month encoding (captures January-December seasonality)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # Cyclical quarter encoding
    df["quarter_sin"] = np.sin(2 * np.pi * df["date"].dt.quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["date"].dt.quarter / 4)

    # Normalized year (trend component)
    min_year = df["date"].dt.year.min()
    max_year = df["date"].dt.year.max()
    year_range = max(max_year - min_year, 1)
    df["year_norm"] = (df["date"].dt.year - min_year) / year_range

    # Lag features (backfill early rows with earliest available value)
    df["lag_1"] = df["sales"].shift(1)
    df["lag_3"] = df["sales"].shift(3)
    df["lag_6"] = df["sales"].shift(6)
    df["lag_12"] = df["sales"].shift(12)

    # Rolling mean features
    df["rolling_mean_3"] = df["sales"].rolling(window=3, min_periods=1).mean().shift(1)
    df["rolling_mean_6"] = df["sales"].rolling(window=6, min_periods=1).mean().shift(1)
    df["rolling_mean_12"] = df["sales"].rolling(window=12, min_periods=1).mean().shift(1)

    # Month-over-month growth rate
    df["mom_growth"] = df["sales"].pct_change(1)

    # Fill NaN in lag/rolling columns with earliest available value per column
    # (instead of dropping rows, which loses too much data with small datasets)
    for col in ["lag_1", "lag_3", "lag_6", "lag_12",
                "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
                "mom_growth"]:
        df[col] = df[col].bfill()

    # Only the very first row might still have NaN (mom_growth), fill with 0
    df = df.fillna(0).reset_index(drop=True)

    return df


FEATURE_COLUMNS = [
    "month_sin", "month_cos",
    "quarter_sin", "quarter_cos",
    "year_norm",
    "lag_1", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
    "mom_growth",
]


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int | None = None,
    val_ratio: float = 0.25,
) -> dict:
    """Scale features and create sliding window sequences for LSTM training.

    Sequence length adapts to dataset size if not specified:
      - 12 if 36+ rows, 6 if 18+ rows, 3 as minimum fallback.
    """
    n_rows = len(df)
    if sequence_length is None:
        if n_rows >= 36:
            sequence_length = 12
        elif n_rows >= 18:
            sequence_length = 6
        else:
            sequence_length = 3

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features = df[FEATURE_COLUMNS].values
    target = df[["sales"]].values

    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)

    max_horizon = 12  # model always predicts 12 months ahead
    # User selects horizon (1/3/6/12) at inference → we slice first N from 12

    # Precompute month_sin/month_cos for all rows
    months = df["date"].dt.month.values
    all_month_sin = np.sin(2 * np.pi * months / 12)
    all_month_cos = np.cos(2 * np.pi * months / 12)

    # Create sliding window sequences with 12-step targets + future month features
    X, y, future_feats = [], [], []
    for i in range(sequence_length, len(scaled_features) - max_horizon + 1):
        X.append(scaled_features[i - sequence_length : i])
        y.append(scaled_target[i : i + max_horizon].flatten())
        # Future month features for the 12 target months
        fm_sin = all_month_sin[i : i + max_horizon]
        fm_cos = all_month_cos[i : i + max_horizon]
        future_feats.append(np.stack([fm_sin, fm_cos], axis=1))  # (12, 2)

    X = np.array(X)
    y = np.array(y)
    future_feats = np.array(future_feats)

    # Train/validation split (last val_ratio as validation)
    split_idx = int(len(X) * (1 - val_ratio))
    split_idx = max(split_idx, 1)  # ensure at least 1 training sample

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    ff_train, ff_val = future_feats[:split_idx], future_feats[split_idx:]

    return {
        "X_train": torch.FloatTensor(X_train),
        "y_train": torch.FloatTensor(y_train),
        "X_val": torch.FloatTensor(X_val),
        "y_val": torch.FloatTensor(y_val),
        "ff_train": torch.FloatTensor(ff_train),  # (N, 12, 2) future month features
        "ff_val": torch.FloatTensor(ff_val),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "sequence_length": sequence_length,
    }


