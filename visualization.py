import pandas as pd
import plotly.graph_objects as go


CURRENCY_SYMBOLS = {"USD": "$", "TRY": "\u20BA"}

# Colors
ICE_BLUE = "#00BFFF"
MAGENTA = "#FF00FF"
DARK_BG = "#1E1E1E"
PLOT_BG = "#2D2D2D"
TEXT_CLR = "#E0E0E0"
GRID_CLR = "#3D3D3D"

CHART_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT_CLR),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
)


def format_currency(value: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, "$")
    return f"{symbol}{value:,.0f}"


def plot_forecast(
    dates: pd.Series,
    actual_sales: pd.Series,
    forecast_dates: list,
    forecast_values: list[float],
    currency: str = "USD",
) -> go.Figure:
    """Plot historical sales and forecasted values."""
    symbol = CURRENCY_SYMBOLS.get(currency, "$")

    fig = go.Figure()

    # Historical sales
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_sales,
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color=ICE_BLUE, width=2),
        marker=dict(size=5),
        hovertemplate=f"Date: %{{x|%b %Y}}<br>Sales: {symbol}%{{y:,.0f}}<extra></extra>",
    ))

    # Connection line (last actual point to first forecast point)
    fig.add_trace(go.Scatter(
        x=[dates.iloc[-1], forecast_dates[0]],
        y=[actual_sales.iloc[-1], forecast_values[0]],
        mode="lines",
        line=dict(color=MAGENTA, width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast",
        line=dict(color=MAGENTA, width=2, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate=f"Date: %{{x|%b %Y}}<br>Forecast: {symbol}%{{y:,.0f}}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title=f"Sales ({currency})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )

    return fig


def plot_loss_curves(train_losses: list[float], val_losses: list[float]) -> go.Figure:
    """Plot training and validation loss curves."""
    epochs = list(range(1, len(train_losses) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode="lines",
        name="Training Loss",
        line=dict(color=ICE_BLUE, width=2),
        hovertemplate="Epoch: %{x}<br>Train Loss: %{y:.6f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode="lines",
        name="Validation Loss",
        line=dict(color=MAGENTA, width=2),
        hovertemplate="Epoch: %{x}<br>Val Loss: %{y:.6f}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Training Loss (Huber / SmoothL1)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )

    return fig


def plot_seasonal_pattern(
    dates: pd.Series,
    sales: pd.Series,
    currency: str = "USD",
) -> go.Figure:
    """Plot average sales by month to show seasonal patterns."""
    symbol = CURRENCY_SYMBOLS.get(currency, "$")

    monthly = pd.DataFrame({"month": dates.dt.month, "sales": sales})
    avg_by_month = monthly.groupby("month")["sales"].mean()

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    months_present = avg_by_month.index.tolist()
    labels = [month_names[m - 1] for m in months_present]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=avg_by_month.values,
        marker_color=ICE_BLUE,
        hovertemplate=f"Month: %{{x}}<br>Avg Sales: {symbol}%{{y:,.0f}}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Seasonal Pattern (Average Sales by Month)",
        xaxis_title="Month",
        yaxis_title=f"Average Sales ({currency})",
        height=400,
    )

    return fig
