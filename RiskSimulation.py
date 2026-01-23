# RiskSimulation.py
# This script performs risk simulation for a portfolio using historical data and Monte Carlo methods.
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# 1. Parameters
# -----------------------------
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

portfolio_weights = np.array([0.25, 0.25, 0.25, 0.25])
initial_portfolio_value = 1_000_000

confidence_levels = [0.95, 0.99]   # 95% and 99%
horizons = [1, 5, 10, 20]          # in days

# -----------------------------
# 2. Download data and compute daily portfolio returns
# -----------------------------
prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()

weights = pd.Series(portfolio_weights, index=tickers)
daily_portfolio_returns = (returns @ weights).dropna()

# -----------------------------
# 3. Helper functions
# -----------------------------
def var_historical(returns_series, alpha=0.95):
    """
    Historical VaR (positive number = loss).
    """
    losses = -returns_series
    return np.quantile(losses, alpha)

def es_historical(returns_series, alpha=0.95):
    """
    Historical Expected Shortfall (Conditional VaR).
    """
    losses = -returns_series
    var_level = np.quantile(losses, alpha)
    tail_losses = losses[losses >= var_level]
    return tail_losses.mean()

def horizon_returns(daily_returns, h):
    """
    Convert daily simple returns to h-day simple returns:
    (1+r1)...(1+rh) - 1, using rolling window.
    """
    return (1 + daily_returns).rolling(h).apply(np.prod, raw=True) - 1

# -----------------------------
# 4. Compute VaR and ES across horizons and confidence levels
# -----------------------------
records = []

for alpha in confidence_levels:
    for h in horizons:
        h_ret = horizon_returns(daily_portfolio_returns, h).dropna()
        var_h = var_historical(h_ret, alpha)
        es_h = es_historical(h_ret, alpha)

        records.append({
            "confidence": alpha,
            "horizon_days": h,
            "VaR_return": var_h,
            "ES_return": es_h,
            "VaR_value": var_h * initial_portfolio_value,
            "ES_value": es_h * initial_portfolio_value,
            "ES_minus_VaR_value": (es_h - var_h) * initial_portfolio_value,
            "ES_over_VaR": es_h / var_h if var_h != 0 else np.nan
        })

results = pd.DataFrame(records)

print("=== Summary table (losses as positive numbers) ===")
print(results.sort_values(["confidence", "horizon_days"]))

# -----------------------------
# 5. Visualization
# -----------------------------
plt.style.use("seaborn-v0_8")

for alpha in confidence_levels:
    df_alpha = results[results["confidence"] == alpha].sort_values("horizon_days")

    # Plot VaR vs ES (in monetary terms)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(df_alpha["horizon_days"], df_alpha["VaR_value"],
               marker="o", label=f"VaR {int(alpha*100)}%")
    ax[0].plot(df_alpha["horizon_days"], df_alpha["ES_value"],
               marker="o", label=f"ES {int(alpha*100)}%")
    ax[0].set_title(f"VaR vs ES across horizons (α={int(alpha*100)}%)")
    ax[0].set_xlabel("Horizon (days)")
    ax[0].set_ylabel("Loss (currency units)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot “how much more conservative” ES is
    # (difference in value and ratio)
    ax[1].bar(df_alpha["horizon_days"] - 0.8,
              df_alpha["ES_minus_VaR_value"],
              width=1.6, alpha=0.7, label="ES − VaR (value)")
    ax2 = ax[1].twinx()
    ax2.plot(df_alpha["horizon_days"], df_alpha["ES_over_VaR"],
             color="red", marker="o", label="ES / VaR (ratio)")

    ax[1].set_title(f"Extra conservativeness of ES vs VaR (α={int(alpha*100)}%)")
    ax[1].set_xlabel("Horizon (days)")
    ax[1].set_ylabel("ES − VaR (currency units)")
    ax2.set_ylabel("ES / VaR (ratio)")

    # Combine legends
    lines1, labels1 = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()