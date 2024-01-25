# option_pricing_with_ML-
import numpy as np
import pandas as pd 
import statistics

#Read the csv into a DataFrame with pandas.
dataframe = pd.read_csv('^SPX.csv', parse_dates = True)

#Adjust the daily returns to log returns.
dataframe['SPX_log_returns'] = np.log(dataframe['Adj Close']).diff()

#Add a dataframe with daily realized volatility.
dataframe['Daily_Volatility'] = np.power((dataframe['SPX_log_returns'] - np.mean(dataframe['SPX_log_returns'])), 2)
print(dataframe)
