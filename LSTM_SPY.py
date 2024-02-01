#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:17:15 2024

@author: willemijnvandervaart
"""

import numpy as np 
import pandas as pd 
import math
from sklearn.metrics import mean_squared_error as mse #this we use as our performance metric
#Read the csv into a DataFrame with pandas. 
dataframe = pd.read_csv('/Users/willemijnvandervaart/Downloads/^SPX.csv', parse_dates = True)

#Adjust Date column to datetime format to simplify readability. 
dataframe['Date'] = dataframe['Date'].astype('datetime64[ns]')

#Adjust the daily returns to log returns. 
dataframe['SPX_log_returns'] = np.log(((dataframe['Adj Close'])/np.roll(dataframe['Adj Close'], 1))) 
x = np.roll(dataframe['Adj Close'], 1)

#Add a dataframe with daily realized volatility. 
dataframe['Daily_Volatility'] = np.power((dataframe['SPX_log_returns'] - np.mean(dataframe['SPX_log_returns'])), 2) 


#We no longer need open, High, low, close, adj close, volume and log returns. 
columns_to_drop = ['Open', 'High', 'Low', "Close", "Adj Close", "Volume", 'SPX_log_returns'] 
dataframe.drop(columns=columns_to_drop, inplace=True)

"""
Realized volatility is calculated based on the price difference between the current day  and last day, 
so the first day in the data set has no realized volatility, so we leave it out. 
"""
 
dataframe = dataframe.dropna() #this leaves out all NaNs, in our case (that should be) only the first observation


#importing the libraries needed to make the LSTM: 
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


#put our code into a numpy array, this allows us to perform our algorithms on it.
#It does not really make sense to keep the dates I believe, there is probably not a lot of information in them and it simplifies the code a lot. 
values = dataframe['Daily_Volatility'] 


#this function creates the sequences 
#X and y are common names in ML for "what we use for our prediction" and "what are trying to predict". 
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y) 


#this function, once the model is instantiated, can be used to retrieve predicted values for the whole dataset. 
def LSTM_predict(model, X, y):
    yhat = model.predict(X)
    return pd.DataFrame({'Realized_Volatility': np.array(y), 'LSTM': np.array(yhat)}) 

def model_LSTM(neurons, batch_size, seq_length = 10, epochs=50): 
    """""
    This function allows us to create an LSTM model and choose different hyperparameters: 
    
    neurons: the more neurons the more complex the model, 
    so for small problems (and small datasets) like this we like to keep the nr low
    
    batch_size: number of sequences processed together in a training iteration

    epochs: number of times the entire training set is passed forwards and backwards through the network during the training process. 
    

    """
    
    
    """
    Splitting the dataset: N.B. in financial applications we don't predict using future observations, 
    so we take the first chunk of data as training data and test on the last chunk. 
    values.shape[0] is the number of daily observations/datapoints, 
    so taking 60% and 80% of it gives us a length for how many points we want in our training and validation sets. 
    
    """
    X, y = create_sequences(values, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    train_length = int(round(.6 * values.shape[0]))
    validation_length = int(round(.8 * values.shape[0]))
    
    X_train, y_train = X[:train_length], y[:train_length]
    X_test, y_test = X[train_length: validation_length], y[train_length: validation_length]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    #these lines instantiate a model: 
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse') #adam is an optimizer for gradient descent, mean squared error is the metric we want to minimize.
    
    
    #now we fit the model: 
    model.fit(X_train, y_train, epochs=epochs,
                     batch_size = batch_size, validation_data=(X_test, y_test),
                     verbose=0, shuffle=False)
    #in sample and out sample predicted values: 
    LSTM_train = LSTM_predict(model, X_train, y_train)
    LSTM_test = LSTM_predict(model, X_test, y_test)

    rmse_in = np.sqrt(mse(LSTM_train['Realized_Volatility'], LSTM_train['LSTM']))
    rmse_out = np.sqrt(mse(LSTM_test['Realized_Volatility'], LSTM_test['LSTM']))
model_LSTM(4, 4)





    
    
    
    
    
    
    
    