import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

# Clone the GitHub repository
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

# Streamlit app starts here
st.title("Portfolio Management Optimization")

# Display the loaded dataset
st.write("Loaded Portfolio:")
st.dataframe(dataset)
st.sidebar.header('Portfolio Weights')
stocks = ['AMAZON', 'MICROSOFT', 'FDX', 'Netflix']
weights = {}

for stock in stocks:
    weights[stock] = st.sidebar.slider(f"{stock} Weight", 0.0, 1.0, 0.25, 0.05)

# Calculate portfolio statistics
st.sidebar.header('Portfolio Statistics')

# Calculate portfolio returns
portfolio_returns = np.sum(dataset.pct_change().mean() * list(weights.values())) * 252

# Convert the portfolio weights list to a NumPy array
weights_array = np.array(list(weights.values()))

# Calculate portfolio volatility
portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(dataset.pct_change().cov() * 252, weights_array)))

# Calculate Sharpe Ratio
risk_free_rate = st.number_input("Enter the risk-free rate (as a decimal):", min_value=0.0, value=0.03, step=0.01)
sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_volatility

# Display portfolio statistics
st.sidebar.write('**Portfolio Statistics**')
st.sidebar.write(f'Expected Annual Return: {portfolio_returns:.2%}')
st.sidebar.write(f'Annual Volatility: {portfolio_volatility:.2%}')
st.sidebar.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')

# Main content
st.header('Portfolio Visualization')

# Plot boxplot
st.subheader('Boxplot of Stock Prices')
plt.figure(figsize=(10, 5))
boxplot = sns.boxplot(data=[dataset['AMAZON'], dataset['MICROSOFT'], dataset['FDX'], dataset['Netflix']])
boxplot.set(xlabel='Stocks', ylabel='Price')
boxplot.set_xticklabels(['AMAZON', 'MICROSOFT', 'FDX', 'Netflix'])  # Set the x-axis labels

# Add legends
legends = ['AMAZON', 'MICROSOFT', 'FDX', 'Netflix']
for i, label in enumerate(legends):
    boxplot.artists[i].set_facecolor('C' + str(i))  # Set box colors

plt.title("Boxplot of Stock Prices")
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)


# Plot daily close prices
st.subheader('Daily Prices of Portfolios')
plt.figure(figsize=(10, 5))
plt.plot(dataset)
plt.title('Daily Close Prices of Stocks')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(dataset.columns)
st.pyplot()

# Calculate and display correlation heatmap
corr = dataset.corr()
st.subheader('Correlation Heatmap')
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot()

st.subheader("Distribution of the Portfolio")
pd.plotting.scatter_matrix(dataset[['AMAZON','MICROSOFT','FDX', 'Netflix']], figsize=(10,10))
st.pyplot()

# Portfolio Optimization Section
st.header('Markowitz Portfolio Optimization')
def log_returns(prices):
    return np.log(prices / prices.shift(1))

def arithmetic_returns(prices):
    return prices/prices.shift(1) - 1
log_return = log_returns(prices=dataset).dropna()
weights_array /= np.sum(weights_array)

# We generally do log return instead of return
Markowitz_log_ret = np.log(dataset / dataset.shift(1))
# Calculate mean log returns as a NumPy array
mean_log_returns = Markowitz_log_ret.mean().values
# Calculate expected return (weighted sum of mean returns)
Markowitz_exp_ret = mean_log_returns.dot(weights_array) * 252
st.subheader(f'\nExpected return of the portfolio is : {Markowitz_exp_ret}')
# Calculate expected volatility (risk)
Markowitz_exp_vol = np.sqrt(weights_array.T.dot(252 * Markowitz_log_ret.cov().dot(weights_array)))
st.subheader(f'\nExpected Volatility of the portfolio is : {Markowitz_exp_vol}')
# Calculate Sharpe ratio
Markowitz_sr = Markowitz_exp_ret / Markowitz_exp_vol
st.subheader(f'\nSharpe Ratio of the portfolio is : {Markowitz_sr}')

# Efficient Frontier Calculation
num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    # Generate random weights
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    # Save the weights
    all_weights[ind, :] = weights

    # Expected Portfolio Return
    ret_arr[ind] = np.sum((dataset.pct_change().mean() * weights) * 252)

    # Expected Portfolio Volatility
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(dataset.pct_change().cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

# Find portfolio with maximum Sharpe Ratio
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

min_var_ret = ret_arr[vol_arr.argmin()]
min_var_vol = vol_arr.min()

# Plot Efficient Frontier
st.subheader('Efficient Frontier')
plt.figure(figsize=(10, 5))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black', label='Maximum Sharpe Ratio')
plt.scatter(min_var_vol, min_var_ret, c='green', s=50, edgecolors='black', label='Minimum Variance')
plt.legend()
st.pyplot()

# Define portfolio statistics functions
def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(dataset.pct_change().cov() * 252, weights)))

def portfolio_return(weights):
    return np.sum(dataset.pct_change().mean() * weights) * 252
    
def portfolio_performance(weight):
    return_p = portfolio_return(weight)
    vol_p    = portfolio_volatility(weight)
    return return_p, vol_p
    
def negativeSR(weight):
    return_p, vol_p = portfolio_performance(weight)
    rf_rate         = 0.025
    return -(return_p - rf_rate)/vol_p


# Create constraints
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Initial Guess (equal distribution)
initial_guess = [1 / len(stocks)] * len(stocks)

# Optimize portfolio for maximum Sharpe Ratio
optimal_weights = minimize(negativeSR, initial_guess, method='SLSQP', bounds=[(0, 1)] * len(stocks),
                           constraints=constraints)

# Display optimized portfolio statistics
st.subheader('Optimized Portfolio Statistics')

# Calculate portfolio statistics for the optimized portfolio
optimal_portfolio_returns = portfolio_return(optimal_weights.x)
optimal_portfolio_volatility = portfolio_volatility(optimal_weights.x)

# Calculate Sharpe Ratio for the optimized portfolio
optimal_sharpe_ratio = (optimal_portfolio_returns - risk_free_rate) / optimal_portfolio_volatility
st.write(f'Expected Annual Return (Optimized): {optimal_portfolio_returns:.2%}')
st.write(f'Annual Volatility (Optimized): {optimal_portfolio_volatility:.2%}')
st.write(f'Sharpe Ratio (Optimized): {optimal_sharpe_ratio:.2f}')

# Efficient Frontier plot with optimized portfolio
st.subheader('Efficient Frontier with Optimized Portfolio')
plt.figure(figsize=(10, 5))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier with Optimized Portfolio')
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black', label='Maximum Sharpe Ratio')
plt.scatter(optimal_portfolio_volatility, optimal_portfolio_returns, c='green', s=50, edgecolors='black',
            label='Optimized Portfolio')
plt.legend()
st.pyplot()

# Display optimal portfolio weights
st.subheader('Optimized Portfolio Weights')
for i, stock in enumerate(stocks):
    st.write(f"{stock}: {optimal_weights.x[i]:.2%}")
