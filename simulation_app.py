import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import git
import os
# Specify the GitHub repository URL
repo_url = "https://github.com/mohanteja7m/App.git"  # Replace with your GitHub repository URL

# Specify the local directory where you want to clone the repository
repo_dir = "local_repository"  # Replace with your desired directory name

if not os.path.exists(repo_dir):
    st.text(f"Cloning the repository from {repo_url} to {repo_dir}")
    git.Repo.clone_from(repo_url, repo_dir)
else:
    st.text(f"Repository already exists in {repo_dir}")

# Load the dataset from the cloned repository
dataset_path = os.path.join(repo_dir, "dataset.csv")  # Replace with your dataset file name
dataset = pd.read_csv(dataset_path)
st.title("Simulation of Portfoilo Optimization")

# Function to calculate portfolio performance metrics
def portfolio_performance(weight, log_return):
    mean_returns = log_return.mean()
    sigma = log_return.cov()
    return_p = np.sum(mean_returns * weight) * 252
    vol_p = np.sqrt(np.dot(weight.T, np.dot(sigma, weight))) * np.sqrt(252)
    return return_p, vol_p

# Function to optimize for maximum Sharpe Ratio
def max_sharpe_ratio(log_return, rf_rate=0.025):
    def negative_SR(weight):
        return_p, vol_p = portfolio_performance(weight, log_return)
        return -(return_p - rf_rate) / vol_p

    n_assets = log_return.shape[1]
    weight_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    w0 = np.random.dirichlet(np.full(n_assets, 0.05)).tolist()

    return minimize(negative_SR, w0, method='SLSQP',
                    bounds=((0, 1),) * n_assets,
                    constraints=weight_constraints)

# Function to optimize for minimum volatility
def min_volatility(log_return):
    n_assets = log_return.shape[1]
    weight_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    w0 = np.random.dirichlet(np.full(n_assets, 0.05)).tolist()
    bounds = ((0, 1),) * n_assets

    return minimize(portfolio_volatility, w0, method='SLSQP',
                    bounds=bounds,
                    constraints=weight_constraints)


log_return = np.log(df / df.shift(1)).dropna()

# User inputs for risk-free rate and portfolio size
rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.5, 0.1)
portfolio = st.slider("Portfolio Size", 10, 100, 50, 5)

# Optimize for maximum Sharpe Ratio
sharpe_maximum = max_sharpe_ratio(log_return, rf_rate / 100)
return_p, vol_p = portfolio_performance(sharpe_maximum['x'], log_return)

# Optimize for minimum volatility
min_volatility_result = min_volatility(log_return)
return_min, vol_min = portfolio_performance(min_volatility_result['x'], log_return)

# Display portfolio statistics and plot
st.write(f"Maximum Sharpe Ratio Portfolio:")
st.write(f"Expected Annual Return: {return_p:.2%}")
st.write(f"Annual Volatility: {vol_p:.2%}")

st.write(f"Minimum Volatility Portfolio:")
st.write(f"Expected Annual Return: {return_min:.2%}")
st.write(f"Annual Volatility: {vol_min:.2%}")

st.subheader("Efficient Frontier")
target = np.linspace(return_min, 1.02, 100)
efficient_portfolios = [portfolio_performance(eff['x'], log_return) for eff in efficient_frontier(target)]

efficient_returns = [eff[0] for eff in efficient_portfolios]
efficient_volatilities = [eff[1] for eff in efficient_portfolios]

fig, ax = plt.subplots()
ax.scatter(efficient_volatilities, efficient_returns, c=target, cmap='plasma')
ax.set_xlabel('Annualized Volatility')
ax.set_ylabel('Annualized Return')
ax.set_title('Efficient Frontier')
ax.scatter(vol_p, return_p, c='r', marker='*', s=500, label='Maximum Sharpe Ratio')
ax.scatter(vol_min, return_min, c='g', marker='*', s=500, label='Minimum Volatility Portfolio')
ax.legend()

st.pyplot(fig)
