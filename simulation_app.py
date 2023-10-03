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
    return_p, vol_p = portfolio_performance(weight)
    rf_rate         = 0.025
    return -(return_p - rf_rate)/vol_p

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
def efficient_portfolio_target(target):

    constraints = ({'type':'eq','fun': lambda x: portfolio_return(x)- target},
                  {'type' :'eq','fun': lambda x: np.sum(x)-1})
    w0          = np.random.dirichlet(np.full(n_assets,0.05)).tolist()
    bounds      = ((0,1),)*n_assets

    return minimize(portfolio_volatility,w0, method = 'SLSQP',
                    bounds      = bounds,
                    constraints = constraints)
def efficient_frontier(return_range):
    return [efficient_portfolio_target(ret) for ret in return_range]


# User inputs for risk-free rate and portfolio size
rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.5, 0.1)
portfolio = st.slider("Portfolio Size", 10, 100, 50, 5)
num_portfolios = st.number_input("Number of Portfolios to Simulate", min_value=1, value=10000, step=1)

# Optimize for maximum Sharpe Ratio
sharpe_maximum = max_sharpe_ratio(log_return, rf_rate / 100)
return_p, vol_p = portfolio_performance(sharpe_maximum['x'], log_return)

# Optimize for minimum volatility
min_volatility_result = min_volatility(log_return)
return_min, vol_min = portfolio_performance(min_volatility_result['x'], log_return)

# Display portfolio statistics and plot
st.write(f"Maximum Sharpe Ratio: {sharpe_maximum:.2%}")
st.write(f"Expected Annual Return: {return_p:.2%}")
st.write(f"Annual Volatility: {vol_p:.2%}")

st.write(f"Minimum Volatility Portfolio:{vol_min:.2%}")
st.write(f"Expected Annual Return: {return_min:.2%}")

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
min_vol_port.to_frame().T #portfolio with the Lowest variance portfolio
max_sharpe_port.to_frame().T
