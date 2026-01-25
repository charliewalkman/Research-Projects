import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize

# ---------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------
tickers = ["RIO.L", "AAPL", "SHEL.L", "BA.L"]
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()

# ---------------------------------------------------------
# 2. Compute CAGR and Max Drawdown
# ---------------------------------------------------------
def compute_cagr(prices):
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    return (end_price / start_price)**(1/n_years) - 1

def compute_max_drawdown(prices):
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax
    return dd.min()  # negative number

# Equal weights for MAR calculation (or choose your own)
w0 = np.repeat(1/len(tickers), len(tickers))

portfolio_prices = (prices * w0).sum(axis=1)

CAGR = compute_cagr(portfolio_prices)      # annual
MaxDD = abs(compute_max_drawdown(portfolio_prices))  # fraction, e.g. 0.3

MAR_annual = CAGR / MaxDD
MAR_daily = (1 + MAR_annual)**(1/252) - 1   # convert to daily rate

MAR = MAR_daily
print("CAGR:", CAGR)
print("MaxDD:", MaxDD)
print("MAR (annual):", MAR_annual)
print("MAR (daily):", MAR)

# ---------------------------------------------------------
# 3. PMPT Metrics: Downside deviation and Sortino Ratio
# ---------------------------------------------------------
def portfolio_stats(weights, returns, mar):
    weights = np.array(weights)
    port_ret = returns @ weights

    mean_ret = port_ret.mean()

    # Downside deviation relative to MAR
    downside = np.minimum(0, port_ret - mar)
    downside_dev = np.sqrt((downside**2).mean())

    sortino = (mean_ret - mar) / downside_dev if downside_dev > 0 else -np.inf

    return mean_ret, downside_dev, sortino

# ---------------------------------------------------------
# 4. Optimize Sortino Ratio
# ---------------------------------------------------------
def neg_sortino(weights, returns, mar):
    _, _, sortino = portfolio_stats(weights, returns, mar)
    return -sortino

n_assets = len(tickers)
bounds = tuple((0, 1) for _ in range(n_assets))
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
w_init = np.repeat(1/n_assets, n_assets)

opt = minimize(
    neg_sortino,
    w_init,
    args=(returns, MAR),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

w_opt = opt.x
mean_ret, downside_dev, sortino = portfolio_stats(w_opt, returns, MAR)

print("\n=== Sortino-Optimised Portfolio (PMPT) ===")
for t, w in zip(tickers, w_opt):
    print(f"{t}: {w:.3f}")

print(f"\nMean daily return: {mean_ret:.6f}")
print(f"Downside deviation: {downside_dev:.6f}")
print(f"Sortino ratio: {sortino:.3f}")
print(f"MAR used: {MAR:.6f}")

# ---------------------------------------------------------
# 5. visualization Monte Carlo PMPT Space + Sortino Optimum
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-bright")


def random_weights(n_assets):
    w = np.random.rand(n_assets)
    return w / w.sum()

n_assets = len(tickers)
n_portfolios = 20_000

mc_mean = []
mc_down = []
mc_sortino = []
mc_weights = []

for _ in range(n_portfolios):
    w = random_weights(n_assets)
    m, d, s = portfolio_stats(w, returns, MAR)
    mc_mean.append(m)
    mc_down.append(d)
    mc_sortino.append(s)
    mc_weights.append(w)

mc_mean = np.array(mc_mean)
mc_down = np.array(mc_down)
mc_sortino = np.array(mc_sortino)
mc_weights = np.array(mc_weights)

# Stats for Sortino-optimised portfolio
opt_mean, opt_down, opt_sortino = portfolio_stats(w_opt, returns, MAR)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(mc_down, mc_mean, c=mc_sortino, cmap="viridis", s=8, alpha=0.7)
plt.colorbar(sc, label="Sortino ratio")

ax.scatter(opt_down, opt_mean, color="red", s=80, marker="*", label="Sortino optimum")

ax.set_xlabel("Downside deviation (daily)")
ax.set_ylabel("Mean daily return")
ax.set_title("PMPT Monte Carlo: mean vs downside (Sortino-coloured)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Efficient Frontier using downside deviation (PMPT Frontier)
# ---------------------------------------------------------
from scipy.optimize import minimize

def downside_only(weights, returns, mar):
    _, d, _ = portfolio_stats(weights, returns, mar)
    return d

def port_mean(weights, returns):
    weights = np.array(weights)
    return (returns @ weights).mean()

n_assets = len(tickers)
bounds = tuple((0, 1) for _ in range(n_assets))
eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

# Use Monte Carlo cloud to pick a reasonable range of target returns
target_returns = np.linspace(mc_mean.min(), mc_mean.max(), 30)

frontier_down = []
frontier_mean = []
frontier_weights = []

for target in target_returns:
    cons = (
        eq_constraint,
        {"type": "ineq", "fun": lambda w, t=target: port_mean(w, returns) - t},
    )
    w0 = np.repeat(1/n_assets, n_assets)

    res = minimize(
        downside_only,
        w0,
        args=(returns, MAR),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )

    if res.success:
        w_star = res.x
        m, d, _ = portfolio_stats(w_star, returns, MAR)
        frontier_mean.append(m)
        frontier_down.append(d)
        frontier_weights.append(w_star)

frontier_mean = np.array(frontier_mean)
frontier_down = np.array(frontier_down)
frontier_weights = np.array(frontier_weights)

# Plot Efficient Frontier vs Monte Carlo cloud
fig, ax = plt.subplots(figsize=(8, 6))

# Monte Carlo cloud
ax.scatter(mc_down, mc_mean, c="lightgray", s=8, alpha=0.5, label="Random portfolios")

# PMPT efficient frontier
ax.plot(frontier_down, frontier_mean, color="blue", linewidth=2, label="Downside-efficient frontier")

# Sortino optimum
ax.scatter(opt_down, opt_mean, color="red", s=80, marker="*", label="Sortino optimum")

ax.set_xlabel("Downside deviation (daily)")
ax.set_ylabel("Mean daily return")
ax.set_title("PMPT efficient frontier (downside deviation vs expected return)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()