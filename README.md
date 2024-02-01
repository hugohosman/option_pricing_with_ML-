# option_pricing_with_ML-
import numpy as np
import pandas as pd 
import statistics
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Read the csv into a DataFrame with pandas.
dataframe = pd.read_csv('^SPX.csv', parse_dates = True)

#Adjust Date column to datetime format to simplify readability.
dataframe['Date'] = dataframe['Date'].astype('datetime64[ns]')

#Adjust the daily returns to log returns.
dataframe['SPX_log_returns'] = np.log(((dataframe['Adj Close']) / np.roll(dataframe['Adj Close'], 1)))
dataframe['SPX_log_returns'][0] = math.nan
dataframe = dataframe.dropna()

#Add a dataframe with daily realized volatility.
dataframe['Daily_Variance'] = np.power((dataframe['SPX_log_returns'] - np.mean(dataframe['SPX_log_returns'])), 2)
dataframe['Daily_Volatility'] = np.sqrt(dataframe['Daily_Variance'])

y_t = dataframe['SPX_log_returns'].values

#Build the GARCH(1,1)-model.
def garch(omega, alpha, beta, y_t):
    T = len(y_t)
    '''start with an array of zeros for the sigma^2 to fill in with results of the function'''
    sigma_2 = np.zeros(T)   
    for i in range(T):
        if i == 0:
            '''from the theory this formula for the first sigma^2'''
            sigma_2[i] = omega / (1 - alpha - beta)     
        else:
            sigma_2[i] = omega + alpha*y_t[i - 1]**2 + beta*sigma_2[i -1]
    return sigma_2

#Build the negative log-likelihood function.
def NLL(param, y_t):
    omega = param[0]
    alpha = param[1]
    beta = param[2]
    sigma_2 = garch(omega, alpha, beta, y_t)
    nll = np.sum(np.log(sigma_2) + (y_t**2) / sigma_2)
    return nll

#Starting parameters to let the model begin.
start_parameters = (0.00001, 0.01, 0.9)

#Minimize the negative log-likelihood is maximizing the likelihood.
result = minimize(NLL, start_parameters, args=y_t, method='nelder-mead', options={'disp': True})

#Obtain the parameters.
omega = result.x[0]
alpha = result.x[1]
beta = result.x[2]

#Implement the parameters in a new column in the dataframe
#take the squareroot because the garch derives the variance 
#which is the square of the volatility as mentioned in the theory.
dataframe['GARCH'] = np.sqrt(garch(omega, alpha, beta, y_t)) 
fig = plt.figure()
plt.plot(dataframe['Date'], dataframe['GARCH'], label = 'GARCH', zorder = 2)
plt.plot(dataframe['Date'], dataframe['Daily_Volatility'], label = 'Daily Volatility', zorder = 1)
plt.xlabel('Date')
plt.legend()
plt.show()
