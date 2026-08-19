import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize


# ---------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------
tickers = ["GSK", "RR.L", "GOOGL", "BP"]
benchmark = "^GSPC"
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# Download assets + benchmark together to ensure aligned dates
all_symbols = tickers + [benchmark]
prices_all = yf.download(all_symbols, start=start_date, end=end_date, auto_adjust=True)["Close"]
prices = prices_all[tickers].copy()
benchmark_prices = prices_all[benchmark].copy()

returns = prices.pct_change().dropna()
benchmark_returns = benchmark_prices.pct_change().dropna()

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

# Equal weights for MAR calculation
w0 = np.repeat(1/len(tickers), len(tickers))

norm_prices = prices / prices.iloc[0]  # normalize to 1 at start
portfolio_prices = (norm_prices * w0).sum(axis=1)

CAGR = compute_cagr(portfolio_prices)      # annual
MaxDD = abs(compute_max_drawdown(portfolio_prices))  # fraction, e.g. 0.3

# guard against zero max drawdown to avoid division by zero
if MaxDD == 0:
    MaxDD = 1e-8

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
    port_ret = (returns @ weights).dropna()
    if isinstance(port_ret, pd.DataFrame):
        port_ret = port_ret.iloc[:, 0]

    # defensive: if no overlapping return observations, return NaNs
    if port_ret.size == 0:
        return np.nan, np.nan, np.nan

    mean_ret = float(port_ret.mean())

    # Downside deviation relative to MAR
    downside = np.minimum(0.0, port_ret - mar)
    downside_dev = float(np.sqrt((downside**2).mean())) if downside.size > 0 else np.nan

    # return NaN (instead of -inf) when downside_dev is zero/undefined
    if downside_dev <= 0 or np.isnan(downside_dev):
        sortino = np.nan
    else:
        sortino = (mean_ret - mar) / downside_dev

    return mean_ret, downside_dev, sortino

# ---------------------------------------------------------
# 4. Optimize Sortino Ratio
# ---------------------------------------------------------
def neg_sortino(weights, returns, mar):
    _, _, sortino = portfolio_stats(weights, returns, mar)
    if np.isnan(sortino):
        return 1e6
    return -sortino


def neg_sortino_reg(weights, returns, mar, lam=1e-3):
    """Negative Sortino with a small L2 regularization to discourage
    extreme concentration. `lam` controls the strength of the penalty."""
    _, _, sortino = portfolio_stats(weights, returns, mar)
    reg = lam * np.sum(np.array(weights) ** 2)
    if np.isnan(sortino):
        return 1e6 + reg
    return -sortino + reg

n_assets = len(tickers)

# Regularization and weight caps to avoid corner solutions
REG_LAM = 1e-2  # increase if solution still concentrates
MIN_W = 1e-4
MAX_W = 0.6

# bounds ensure each weight is between MIN_W and MAX_W
bounds = tuple((MIN_W, MAX_W) for _ in range(n_assets))
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
w_init = np.repeat(1/n_assets, n_assets)

# Diagnostics: compute single-asset Sortino-like metrics to see which asset dominates
print("\nSingle-asset metrics:")
for i, t in enumerate(tickers):
    w = np.zeros(n_assets)
    w[i] = 1.0
    m, d, s = portfolio_stats(w, returns, MAR)
    print(f"{t}: mean={m:.6f}, downside={d:.6f}, sortino={s:.3f}")

opt = minimize(
    neg_sortino_reg,
    w_init,
    args=(returns, MAR, REG_LAM),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000}
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

# -------------------------
# Comparative metrics vs benchmark (^GSPC)
# -------------------------
def compute_stats_from_prices(prices_series):
    cagr = compute_cagr(prices_series)
    maxdd = abs(compute_max_drawdown(prices_series))
    if maxdd == 0:
        maxdd = 1e-8
    mar_ann = cagr / maxdd
    mar_daily_local = (1 + mar_ann) ** (1/252) - 1
    return cagr, maxdd, mar_ann, mar_daily_local

def sortino_from_returns(returns_series, mar):
    rs = returns_series.dropna()
    if isinstance(rs, pd.DataFrame):
        rs = rs.iloc[:, 0]
    if rs.size == 0:
        return np.nan, np.nan, np.nan

    downside = np.minimum(0.0, rs - mar)
    downside_dev = float(np.sqrt((downside ** 2).mean())) if downside.size > 0 else np.nan
    mean_ret = float(rs.mean())
    sortino = (mean_ret - mar) / downside_dev if (downside_dev > 0 and not np.isnan(downside_dev)) else np.nan
    return mean_ret, downside_dev, sortino

# Build portfolio price series for equal-weight and optimised weights
norm_prices = prices / prices.iloc[0]
portfolio_prices_eq = (norm_prices * w0).sum(axis=1)
portfolio_prices_opt = (norm_prices * w_opt).sum(axis=1)

# Align dates
common_idx = portfolio_prices_opt.index.intersection(benchmark_prices.index)
pp_opt = portfolio_prices_opt.loc[common_idx]
pp_eq = portfolio_prices_eq.loc[common_idx]
bp = benchmark_prices.loc[common_idx]

port_opt_returns = pp_opt.pct_change().dropna()
port_eq_returns = pp_eq.pct_change().dropna()
bench_ret_aligned = bp.pct_change().dropna()

# Compute price-based stats (CAGR, MaxDD, MAR)
cagr_b, mdd_b, mar_ann_b, mar_b = compute_stats_from_prices(bp)
cagr_opt, mdd_opt, mar_ann_opt, mar_opt = compute_stats_from_prices(pp_opt)
cagr_eq, mdd_eq, mar_ann_eq, mar_eq = compute_stats_from_prices(pp_eq)

def annualize_sortino(mean_daily, downside_dev_daily, mar_ann):
    if np.isnan(mean_daily) or np.isnan(downside_dev_daily) or downside_dev_daily <= 0:
        return np.nan
    mean_ann = float(mean_daily) * 252
    downside_ann = float(downside_dev_daily) * np.sqrt(252)
    target_ann = float(mar_ann)
    return (mean_ann - target_ann) / downside_ann if downside_ann > 0 else np.nan

# Diagnostic MAR-based Sortino values for reference only
m_ret_b, d_b, s_b = sortino_from_returns(bench_ret_aligned.reindex(port_opt_returns.index), mar_b)
m_ret_opt, d_opt, s_opt = sortino_from_returns(port_opt_returns, mar_opt)
m_ret_eq, d_eq, s_eq = sortino_from_returns(port_eq_returns, mar_eq)

s_b_ann = annualize_sortino(m_ret_b, d_b, mar_ann_b)
s_eq_ann = annualize_sortino(m_ret_eq, d_eq, mar_ann_eq)
s_opt_ann = annualize_sortino(m_ret_opt, d_opt, mar_ann_opt)

# Compare with the standard benchmark convention: zero target (or risk-free target)
# MAR-based Sortino is kept only for diagnostics; headline benchmark output should use zero target.
zero_target = 0.0
mb0, db0, sb0 = sortino_from_returns(bench_ret_aligned.reindex(port_opt_returns.index), zero_target)
me0, de0, se0 = sortino_from_returns(port_eq_returns, zero_target)
mo0, do0, so0 = sortino_from_returns(port_opt_returns, zero_target)

sb0_ann = annualize_sortino(mb0, db0, 0.0)
se0_ann = annualize_sortino(me0, de0, 0.0)
so0_ann = annualize_sortino(mo0, do0, 0.0)

print("\n=== Comparative summary vs ^GSPC (standard Sortino, zero target) ===")
print(f"Benchmark (^GSPC) - CAGR: {cagr_b:.3%}, MaxDD: {mdd_b:.3%}, Sortino (daily): {sb0:.3f}, Sortino (ann): {sb0_ann:.3f}")
print(f"Equal-weight portfolio - CAGR: {cagr_eq:.3%}, MaxDD: {mdd_eq:.3%}, Sortino (daily): {se0:.3f}, Sortino (ann): {se0_ann:.3f}")
print(f"Optimised portfolio - CAGR: {cagr_opt:.3%}, MaxDD: {mdd_opt:.3%}, Sortino (daily): {so0:.3f}, Sortino (ann): {so0_ann:.3f}")

print("\nDiagnostic MAR-based Sortino (not used for headline benchmark comparison):")
print(f"^GSPC - Sortino (daily): {s_b:.3f}, Sortino (ann): {s_b_ann:.3f}")
print(f"Equal-weight - Sortino (daily): {s_eq:.3f}, Sortino (ann): {s_eq_ann:.3f}")
print(f"Optimised - Sortino (daily): {s_opt:.3f}, Sortino (ann): {s_opt_ann:.3f}")

# Additional annualised volatility & Sharpe (assume rf=0)
def ann_vol(returns_series):
    return returns_series.std() * np.sqrt(252)

def sharpe(returns_series, rf=0.0):
    mean_ann = returns_series.mean() * 252
    vol_ann = ann_vol(returns_series)
    return (mean_ann - rf) / vol_ann if vol_ann > 0 else np.nan

print(f"\nAnnualised volatility: ^GSPC={ann_vol(bench_ret_aligned):.3%}, EQ={ann_vol(port_eq_returns):.3%}, OPT={ann_vol(port_opt_returns):.3%}")
print(f"Sharpe (rf=0): ^GSPC={sharpe(bench_ret_aligned):.3f}, EQ={sharpe(port_eq_returns):.3f}, OPT={sharpe(port_opt_returns):.3f}")

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

fig, ax = plt.subplots(figsize=(10, 7))
# Monte Carlo cloud colored by Sortino
sc = ax.scatter(mc_down, mc_mean, c=mc_sortino, cmap="viridis", s=14, alpha=0.35, rasterized=True)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Sortino ratio", rotation=270, labelpad=15)

# Annotate and highlight the Sortino-optimised portfolio
ax.scatter(opt_down, opt_mean, color="red", s=160, marker="*", edgecolors='k', linewidths=0.6, label="Sortino optimum")
opt_weights_str = ", ".join([f"{t}:{w:.2f}" for t, w in zip(tickers, w_opt)])
ax.annotate(f"Opt weights:\n{opt_weights_str}\nSortino: {opt_sortino:.2f}", xy=(opt_down, opt_mean), xytext=(0.98, 0.02), textcoords='axes fraction',
            ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.85))

# Equal-weight marker
eq_mean, eq_down, _ = portfolio_stats(w0, returns, MAR)
ax.scatter(eq_down, eq_mean, color='navy', s=90, marker='D', label='Equal weight')

ax.set_xlabel("Downside deviation (daily)")
ax.set_ylabel("Mean daily return")
ax.set_title("PMPT Monte Carlo: mean vs downside (Sortino-coloured)")
ax.legend(framealpha=0.9)
ax.grid(alpha=0.25)
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
fig, ax = plt.subplots(figsize=(10, 7))

# Monte Carlo cloud (light)
ax.scatter(mc_down, mc_mean, c="lightgray", s=10, alpha=0.45, label="Random portfolios")

# Ensure frontier is sorted by downside for plotting
if len(frontier_down) > 0:
    idx = np.argsort(frontier_down)
    fd = frontier_down[idx]
    fm = frontier_mean[idx]
    ax.plot(fd, fm, color="tab:blue", linewidth=2.2, label="Downside-efficient frontier")

# Plot single-asset points and label them
for i, t in enumerate(tickers):
    w = np.zeros(n_assets)
    w[i] = 1.0
    m, d, s = portfolio_stats(w, returns, MAR)
    ax.scatter(d, m, s=80, marker='X', label=f"{t} (single)")
    ax.annotate(t, xy=(d, m), xytext=(6, -6), textcoords='offset points', fontsize=9)

# Sortino optimum
ax.scatter(opt_down, opt_mean, color="red", s=160, marker="*", edgecolors='k', linewidths=0.6, label="Sortino optimum")
ax.annotate(f"Opt: {opt_sortino:.2f}", xy=(opt_down, opt_mean), xytext=(10, 10), textcoords='offset points', fontsize=9, color='red')

ax.set_xlabel("Downside deviation (daily)")
ax.set_ylabel("Mean daily return")
ax.set_title("PMPT efficient frontier (downside deviation vs expected return)")
ax.legend(ncol=2, framealpha=0.9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()