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


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

QUARTER_COLORS = {
    "Q1": "#4FC3F7",
    "Q2": "#81C784",
    "Q3": "#FFB74D",
    "Q4": "#E57373",
}

YEAR_PALETTE = [
    "#00BFFF", "#FF00FF", "#00FF7F", "#FFD700",
    "#FF6347", "#7B68EE", "#00CED1", "#FF69B4",
]


def plot_seasonal_pattern(
    dates: pd.Series,
    sales: pd.Series,
    currency: str = "USD",
) -> go.Figure:
    """Plot average sales by month with min/max range and overall average line."""
    symbol = CURRENCY_SYMBOLS.get(currency, "$")

    monthly = pd.DataFrame({"month": dates.dt.month, "sales": sales})
    stats = monthly.groupby("month")["sales"].agg(["mean", "min", "max", "std"])

    months_present = stats.index.tolist()
    labels = [MONTH_NAMES[m - 1] for m in months_present]
    quarter_for_month = [(m - 1) // 3 for m in months_present]
    bar_colors = [list(QUARTER_COLORS.values())[q] for q in quarter_for_month]

    overall_avg = sales.mean()

    fig = go.Figure()

    # Min-max range band
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=list(stats["max"].values) + list(stats["min"].values[::-1]),
        fill="toself",
        fillcolor="rgba(0,191,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Min–Max Range",
        hoverinfo="skip",
    ))

    # Average bars colored by quarter
    fig.add_trace(go.Bar(
        x=labels,
        y=stats["mean"].values,
        marker_color=bar_colors,
        name="Monthly Average",
        hovertemplate=(
            f"Month: %{{x}}<br>"
            f"Avg: {symbol}%{{y:,.0f}}<br>"
            f"Min: {symbol}%{{customdata[0]:,.0f}}<br>"
            f"Max: {symbol}%{{customdata[1]:,.0f}}"
            "<extra></extra>"
        ),
        customdata=list(zip(stats["min"].values, stats["max"].values)),
    ))

    # Overall average reference line
    fig.add_hline(
        y=overall_avg,
        line_dash="dot",
        line_color="#FFFFFF",
        line_width=1,
        annotation_text=f"Overall Avg: {symbol}{overall_avg:,.0f}",
        annotation_position="top right",
        annotation_font_color=TEXT_CLR,
    )

    fig.update_layout(
        **CHART_LAYOUT,
        title="Monthly Seasonal Pattern",
        xaxis_title="Month",
        yaxis_title=f"Sales ({currency})",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.25,
    )

    return fig


def plot_yearly_comparison(
    dates: pd.Series,
    sales: pd.Series,
    currency: str = "USD",
) -> go.Figure:
    """Overlay each year's monthly sales as separate lines for YoY comparison."""
    symbol = CURRENCY_SYMBOLS.get(currency, "$")

    df = pd.DataFrame({"date": dates, "sales": sales})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    years = sorted(df["year"].unique())

    fig = go.Figure()

    for i, year in enumerate(years):
        year_data = df[df["year"] == year].sort_values("month")
        color = YEAR_PALETTE[i % len(YEAR_PALETTE)]
        labels = [MONTH_NAMES[m - 1] for m in year_data["month"]]

        fig.add_trace(go.Scatter(
            x=labels,
            y=year_data["sales"].values,
            mode="lines+markers",
            name=str(year),
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"{year}<br>"
                f"Month: %{{x}}<br>"
                f"Sales: {symbol}%{{y:,.0f}}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="Year-over-Year Monthly Comparison",
        xaxis_title="Month",
        yaxis_title=f"Sales ({currency})",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def compute_seasonal_insights(
    dates: pd.Series,
    sales: pd.Series,
) -> dict:
    """Compute seasonal insights from the data."""
    df = pd.DataFrame({"date": dates, "sales": sales})
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = (df["month"] - 1) // 3 + 1

    # Monthly averages
    monthly_avg = df.groupby("month")["sales"].mean()
    overall_avg = sales.mean()

    peak_month_num = int(monthly_avg.idxmax())
    low_month_num = int(monthly_avg.idxmin())
    peak_month_name = MONTH_NAMES[peak_month_num - 1]
    low_month_name = MONTH_NAMES[low_month_num - 1]
    peak_value = monthly_avg.max()
    low_value = monthly_avg.min()

    # Seasonal strength: coefficient of variation of monthly averages (0-100%)
    seasonal_strength = (monthly_avg.std() / overall_avg * 100) if overall_avg != 0 else 0

    # Quarter analysis
    quarter_avg = df.groupby("quarter")["sales"].mean()
    quarter_labels = {1: "Q1 (Jan–Mar)", 2: "Q2 (Apr–Jun)", 3: "Q3 (Jul–Sep)", 4: "Q4 (Oct–Dec)"}
    best_quarter = int(quarter_avg.idxmax())
    worst_quarter = int(quarter_avg.idxmin())

    quarter_data = []
    for q in sorted(quarter_avg.index):
        avg = quarter_avg[q]
        pct_diff = ((avg - overall_avg) / overall_avg * 100) if overall_avg != 0 else 0
        quarter_data.append({
            "quarter": quarter_labels.get(q, f"Q{q}"),
            "avg_sales": avg,
            "vs_overall": pct_diff,
        })

    # Year-over-year growth
    yearly_totals = df.groupby("year")["sales"].sum().sort_index()
    yoy_growth = []
    years = list(yearly_totals.index)
    for i in range(1, len(years)):
        prev = yearly_totals.iloc[i - 1]
        curr = yearly_totals.iloc[i]
        growth_pct = ((curr - prev) / prev * 100) if prev != 0 else 0
        yoy_growth.append({
            "period": f"{years[i-1]}→{years[i]}",
            "growth_pct": growth_pct,
            "prev_total": prev,
            "curr_total": curr,
        })

    # Month-over-month volatility (std of pct changes)
    pct_changes = sales.pct_change().dropna()
    mom_volatility = pct_changes.std() * 100

    return {
        "peak_month": peak_month_name,
        "peak_value": peak_value,
        "low_month": low_month_name,
        "low_value": low_value,
        "seasonal_strength": seasonal_strength,
        "overall_avg": overall_avg,
        "peak_to_low_ratio": (peak_value / low_value) if low_value != 0 else 0,
        "best_quarter": quarter_labels.get(best_quarter, ""),
        "worst_quarter": quarter_labels.get(worst_quarter, ""),
        "quarter_data": quarter_data,
        "yoy_growth": yoy_growth,
        "mom_volatility": mom_volatility,
        "num_years": len(years),
    }
