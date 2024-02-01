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
