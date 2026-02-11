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

    # Lag features
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

    # Drop rows with NaN from lag/rolling features (first 12 rows)
    df = df.dropna().reset_index(drop=True)

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
    sequence_length: int = 12,
    val_ratio: float = 0.2,
) -> dict:
    """Scale features and create sliding window sequences for LSTM training.

    Returns dict with train/val tensors and the fitted scalers.
    """
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features = df[FEATURE_COLUMNS].values
    target = df[["sales"]].values

    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)

    # Create sliding window sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length : i])
        y.append(scaled_target[i])

    X = np.array(X)
    y = np.array(y)

    # Train/validation split (last val_ratio as validation)
    split_idx = int(len(X) * (1 - val_ratio))
    split_idx = max(split_idx, 1)  # ensure at least 1 training sample

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return {
        "X_train": torch.FloatTensor(X_train),
        "y_train": torch.FloatTensor(y_train),
        "X_val": torch.FloatTensor(X_val),
        "y_val": torch.FloatTensor(y_val),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "sequence_length": sequence_length,
    }


def build_forecast_features(
    df: pd.DataFrame,
    last_date: pd.Timestamp,
    predicted_sales: list[float],
    horizon: int,
    step: int,
) -> np.ndarray:
    """Build feature vector for the next forecast step.

    Uses actual historical data + already predicted values to construct
    the feature row for the next month to predict.
    """
    # Combine historical sales with predictions so far
    all_sales = list(df["sales"].values) + predicted_sales
    all_dates = list(df["date"].values)

    next_date = last_date + pd.DateOffset(months=step + 1)
    month = next_date.month
    quarter = next_date.quarter
    year = next_date.year

    # Cyclical encodings
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    quarter_sin = np.sin(2 * np.pi * quarter / 4)
    quarter_cos = np.cos(2 * np.pi * quarter / 4)

    # Year normalization (extend from training range)
    min_year = pd.Timestamp(all_dates[0]).year if len(all_dates) > 0 else year
    max_year = max(pd.Timestamp(all_dates[-1]).year if len(all_dates) > 0 else year, year)
    year_range = max(max_year - min_year, 1)
    year_norm = (year - min_year) / year_range

    n = len(all_sales)

    # Lag features (from combined actual + predicted)
    lag_1 = all_sales[-1] if n >= 1 else 0
    lag_3 = all_sales[-3] if n >= 3 else all_sales[0]
    lag_6 = all_sales[-6] if n >= 6 else all_sales[0]
    lag_12 = all_sales[-12] if n >= 12 else all_sales[0]

    # Rolling means
    rolling_3 = np.mean(all_sales[-3:]) if n >= 3 else np.mean(all_sales)
    rolling_6 = np.mean(all_sales[-6:]) if n >= 6 else np.mean(all_sales)
    rolling_12 = np.mean(all_sales[-12:]) if n >= 12 else np.mean(all_sales)

    # Month-over-month growth
    if n >= 2 and all_sales[-2] != 0:
        mom_growth = (all_sales[-1] - all_sales[-2]) / abs(all_sales[-2])
    else:
        mom_growth = 0.0

    return np.array([
        month_sin, month_cos,
        quarter_sin, quarter_cos,
        year_norm,
        lag_1, lag_3, lag_6, lag_12,
        rolling_3, rolling_6, rolling_12,
        mom_growth,
    ]).reshape(1, -1)
