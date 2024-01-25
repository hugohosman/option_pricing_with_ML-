# option_pricing_with_ML-
import numpy as np
import pandas as pd 

#Read the csv into a DataFrame with pandas.
dataframe = pd.read_csv('^SPX.csv', parse_dates = True)


#Adjust the daily returns to log returns.
dataframe['SPX_log_returns'] = np.log(dataframe['Adj Close']).diff()
print(dataframe)

#Create a function to calculate the realized volatility
