# Table of Projects
- [Portfolio Choice](#portfolio-choice)
- [Portfolio Optimisation](#portfolio-optimisation)
- [Risk Simulation](#risk-simulation)

## Installation
1. Clone the repository:
git clone https://github.com/charliewalkman/Research-Projects.git
cd Research-Projects

2. Install dependencies:
pip install -r requirements.txt

----------------------------------------------------------------------------------------------
# Portfolio Choice 
This project walks through the core workflow of quantitative portfolio construction, from raw price data to optimal asset allocation.

## Table of Contents
- [Topics Covered](#topics-covered)
- [Usage](#usage)

## Topics Covered
- Data fundamentals - understanding the inputs to a portfolio problem: assets, time period, prices, and returns
- Return & risk metrics - computing daily and annualised returns, and constructing the covariance matrix
- Optimisation - formulating and solving mean-variance optimisation with realistic constraints (weights summing to 1, no short-selling)
- Interpretation - analysing optimal portfolio weights, risk (variance), and expected return, with visualisations of the results
- Tangency portfolio - identifying the tangency portfolio by maximising the Sharpe ratio and comparing it against other allocations

## Usage 
To run the project, use:
python PortfolioChoice.py
----------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------
# Portfolio Optimisation
This project is a more complex quantitative portfolio analysis, using ideas from [Portfolio Choice](#portfolio-choice), but allocating assets through PMPT, using Sortino Ratio and downside risk.
Project is also backtested against the S&P 500 as a benchmark, to compare effectiveness.

## Table of Contents
- [Topics Covered](#topics-covered-1)
- [Usage](#usage-1)

## Topics Covered 
- Data download and basic metric computations - Prices aligned with dates for assets and benchmark, and computation of CAGR and max drawdown.
- PMPT metrics - Calculated downside deviation and sortino ratio
- Optimised Sortino Ratio - Compute signle-asset metrics, and comparative metrics vs. benchmark, including equal-weight and optimised-weight comparison.
- Visualisations - Monte-Carlo PMPT space, with sortino optimum, as well as single-asset sortino. And efficient frontier using PMPT frontier.

## Usage 
To run the project, use:
python PortfolioOptimisation.py
----------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------
# Risk Simulation
Another quantitative project; this time working with different risk measures as to question the most effective risk management strategy for a portfolio, Value at Risk -VaR-, or Expected Shortfall -ES-?

## Table of Contents
- [Topics Covered](#topics-covered-2)
- [Usage](#usage-2)

## Topics Covered
- Introduce assumptions and data - Data downloaded for assets, and assume different confidence levels and horizons.
- Create helper functions - Functions made for hostorical returns of VaR, ES, and the returns of different horizons. 
- ES and VaR calculated across different horizons, and confidence levels
- Visualisations in monetary terms, and determine how much more conservative ES is - difference in value and ratio. 

## Usage
To run the project, use:
python RiskSimulation.py
----------------------------------------------------------------------------------------------