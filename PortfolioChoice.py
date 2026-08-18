import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
import plotly.express as px  

tickers = ["GSK", "RR.L", "GOOGL", "BP"]
benchmark = "^GSPC"
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

all_symbols = tickers + [benchmark]
prices_all = yf.download(all_symbols, start=start_date, end=end_date, auto_adjust=True)["Close"]
prices = prices_all[tickers].copy()
benchmark_prices = prices_all[benchmark].copy()

returns = prices.pct_change().dropna()
benchmark_returns = benchmark_prices.pct_change().dropna()


fig = px.line(prices,
              title='stock returns',
              labels={"index": "Date", "value": "Return"}
)

fig.update_layout(template='plotly_white',
                  title_x=0.5)

fig.show()

expected_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
print(f"\nExpected annual returns:")
print(expected_returns)

print("\nCovariance matrix:")
print(cov_matrix)

fig = px.bar(
    expected_returns,
    x=expected_returns.index,
    y=expected_returns.values,
    color=expected_returns.index,
    title="Expected Returns",
    labels={"x": "Asset", "y": "Expected Return"}
)

fig.update_layout(
    template="plotly_white",
    title_x=0.5,             # centre title
    showlegend=False
)

fig.show()


fig = px.imshow(
    cov_matrix,
    title="Covariance Matrix Heatmap",
    labels=dict(x="", y=""),
    text_auto=True,          # shows numbers in each cell
    color_continuous_scale="Reds",  # Shades of red color
    aspect="auto"
)


fig.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig.show()

# Define Objective Function: Maximising the Shape Ratio is the same as Minimising the -ve of the Sharpe Ratio
def neg_sharpe_ratio(weights, expected_returns, cov_matrix):

    portfolio_return = np.dot(weights, expected_returns)

    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    return - portfolio_return / portfolio_risk

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# No short selling: weights between 0 and 1
bounds = tuple((0,1) for _ in tickers)

# Initial guess: equal distribution
init_guess = np.array(len(tickers) * [1. / len(tickers)])

# Minimise function
result = minimize(neg_sharpe_ratio, init_guess, args=(expected_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

result

# Get weights which maximise the Sharpe Ratio
sharpe_weights = result.x

# Plotly bar chart
fig = px.bar(sharpe_weights,
            x=tickers,
            y=sharpe_weights,
            color=tickers,
            title="Optimal Portfolio Weights (Max Sharpe Ratio)",
            labels={"x": "Asset", "y": "Weight"})

fig.update_layout(
    template="plotly_white",
    title_x=0.5,             # centre title
    showlegend=False
)

fig.show()

# Portfolio return: R* = w^T * mu
sharpe_return = np.dot(sharpe_weights, expected_returns)

# Portfolio risk: σ* = sqrt(w^T * Ω * w)
sharpe_risk = np.sqrt(np.dot(sharpe_weights.T, np.dot(cov_matrix, sharpe_weights)))

print(f"\nOptimal Portfolio Expected Return: {sharpe_return:.2%}")
print(f"\nOptimal Portfolio Risk (Std Dev): {sharpe_risk:.2%}")
print(f"\nOptimal Portfolio Sharpe Ratio: {sharpe_return/sharpe_risk:.2f}")

# Generate a range of target returns to explore
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
num_assets = len(tickers)

for i in range(num_portfolios):
    # Generate random weights that sum to 1
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    # Calculate portfolio return and risk
    port_return = np.dot(weights, expected_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Store results
    results[0,i] = port_risk
    results[1,i] = port_return
    results[2,i] = (port_return) / port_risk  # Sharpe ratio (risk-free rate not subtracted)


# Plotly scatter plot
fig = px.scatter(
    x=results[0,:],
    y=results[1,:],
    color=results[2,:],
    color_continuous_scale='Viridis',
    title='Efficient Frontier',
    labels={'x': 'Risk (Std Dev)', 'y': 'Portfolio Return', 'color': 'Sharpe Ratio'}
)

fig.add_scatter(
    x=[sharpe_risk],
    y=[sharpe_return],
    mode='markers',
    marker=dict(size=14, color='red', symbol='star'),
    name='Optimal Portfolio'
)

fig.update_layout(template='plotly_white',
                  title_x=0.5,
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()