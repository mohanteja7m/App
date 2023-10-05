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
log_return = np.log(dataset / dataset.shift(1)).dropna()
sigma            = log_return.cov()
n_assets         = log_return.shape[1]
mean_returns     = log_return.mean()
# Function to calculate portfolio performance metrics
 
def portfolio_performance(weight, log_return):
    mean_returns = log_return.mean()
    sigma = log_return.cov()
    return_p = np.sum(mean_returns * weight) * 252
    vol_p = np.sqrt(np.dot(weight.T, np.dot(sigma, weight))) * np.sqrt(252)
    return return_p, vol_p
def portfolio_return(weight):

    return np.sum(mean_returns*weight)*252

def negativeSR(weight):
    return_p, vol_p = portfolio_performance(weight,log_return)
    rf_rate         = 0.025
    return -(return_p - rf_rate)/vol_p

def portfolio_volatility(weight):
    return np.sqrt(np.dot(weight.T, np.dot(sigma,weight)))*np.sqrt(252)


# Function to optimize for minimum volatility
def min_volatility(log_return):
    n_assets = log_return.shape[1]
    weight_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    w0 = np.random.dirichlet(np.full(n_assets, 0.05)).tolist()
    bounds = ((0, 1),) * n_assets

    return minimize(portfolio_volatility, w0, method='SLSQP',
                    bounds=bounds,

                    constraints=weight_constraints)
 
def min_vol():

    n_assets           = log_return.shape[1]
    weight_constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})
    w0                 = np.random.dirichlet(np.full(n_assets,0.05)).tolist()
    bounds             = ((0,1),)*n_assets

    return minimize(portfolio_volatility,w0,method='SLSQP',
                   bounds      = bounds,
                   constraints = weight_constraints)

# User inputs for risk-free rate and portfolio size
rf_rate = st.number_input("Enter the risk-free rate (as a decimal):", min_value=0.0, max_value=1.0, step=0.01)

portfolio = st.slider("Portfolio Size", 10, 100, 50, 5)
num_portfolios = int(st.number_input("Number of Portfolios to Simulate", min_value=1, value=10000, step=1))
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
portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(dataset.pct_change().cov() * 252, weights_array)))

# Calculate Sharpe Ratio
sharpe_ratio = (portfolio_returns - rf_rate) / portfolio_vol
 
# Display portfolio statistics
st.sidebar.write(f'Expected Annual Return: {portfolio_returns:.2%}')
st.sidebar.write(f'Annual Volatility: {portfolio_vol:.2%}')
st.sidebar.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')
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
all_weights = np.zeros((num_portfolios, len(stocks)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)
 
for ind in range(num_portfolios):
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
st.set_option('deprecation.showPyplotGlobalUse', False)
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
optimal_sharpe_ratio = (optimal_portfolio_returns - rf_rate) / optimal_portfolio_volatility
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

min_volatility_result = min_volatility(log_return)
return_min, vol_min = portfolio_performance(min_volatility_result['x'], log_return)

tickers = []
for i in dataset[['AMAZON','MICROSOFT','FDX','Netflix']].columns:
    tickers.append(i)
 
def calc_portfolio_perf(weights, mean_returns, cov, rf):# portfolio performance, calculate the annualised return, sharpe ratio
    portfolio_return = np.sum(mean_returns*weights)*252 #252 working days at the stock exchange
    portfolio_std = np.sqrt(np.dot(weights.T,np.dot(cov,weights)))*np.sqrt(252) # np.dot multiplication of matrices
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio

# ...
 
def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
    num_assets = len(mean_returns)  # Convert the number of assets to an integer
    num_portfolios = int(num_portfolios)  # Convert the number of portfolios to an integer
    results_matrix = np.zeros((num_assets + 3, num_portfolios))  # Use num_assets as an integer
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = sharpe_ratio
        for j in range(num_assets):
            results_matrix[j + 3, i] = weights[j]
    results_df = pd.DataFrame(results_matrix.T, columns=['ret', 'stdev', 'sharpe'] + [ticker for ticker in tickers])
    return results_df

# ...


mean_returns = dataset[['AMAZON','MICROSOFT','FDX','Netflix']].pct_change().mean()
cov = dataset[['AMAZON','MICROSOFT','FDX','Netflix']].pct_change().cov() 
results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf_rate)
font1 = {'family':'serif','color':'darkred','size':20,'weight':'bold'}
font2 = {'family':'serif','color':'darkred','size':20,'weight':'bold'}
#Locate position of portfolio with highest Sharpe Ratio
 
max_sharpe_port=results_frame.iloc[results_frame["sharpe"].idxmax()] # max sharp ratio rouge
#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame["stdev"].idxmin()] # min volatility = min variance portfolio vert
#create scatter plot coloured by Sharpe Ratio
plt.subplots(figsize=(15,10)) # Number of rows/colums of the subplot grid
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='plasma') #Colormaps in Matplotlib
plt.title('Optimization of the portfolio',fontdict=font1)
plt.xlabel('Risk/Annualised Volatility',fontdict=font2)
plt.ylabel('Annualised Returns',fontdict=font2)
#plt.colorbar () # match the colorbar

plt.colorbar().set_label('Sharpe Ratio', size= 20, color = 'g', family='serif',weight='bold')
target               = np.linspace(return_min,1.02,100)
#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500, label = 'Maximum Sharpe Ratio')
#plot green star to highlight position of minimum vartance portfolio

plt.scatter(min_vol_port[1] ,min_vol_port[0],marker=(5,1,0),color='g', s=500, label='Minimum Volatility Portfolio')
plt.legend(labelspacing=0.8)
st.pyplot(plt)
st.subheader("Portfolio with Minimum Variance(Volatility)")

min_vol_port.to_frame().T #portfolio with the Lowest variance portfolio
st.subheader("Portfolio with Maximum Sharpe Ratio(Expected Return)")
max_sharpe_port.to_frame().T
