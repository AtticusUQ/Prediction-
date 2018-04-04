#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:21:14 2018

@author: dan
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math

# Reading the price data from .xlsx file ensure location matches where you save files
xl_aapl = pd.ExcelFile('/users/dan/desktop/Data/AAPL.xlsx')
dfP_aapl = xl_aapl.parse('Sheet1')
P_aapl = dfP_aapl.fillna(method="backfill")
P_aapl = np.array(P_aapl.drop('Date', axis=1))

W_LTMA = 60
W_STMA = 20
T1_aaple = 0.03
T2_aaple = 0.01

[R_aaple, C_aaple] = P_aapl.shape
dm = np.zeros(shape=(R_aaple-W_LTMA,3), dtype=float)
mr = np.zeros(shape=(R_aaple-W_LTMA,1), dtype=float)
pos_aaple = np.zeros(shape=(R_aaple-W_LTMA+1,1), dtype=int)
p_apple_snap = 0.0
PNL_apple = np.zeros(shape=(R_aaple-W_LTMA,1), dtype=float)

idx = 0
for i in range(W_LTMA, R_aaple):
    ps = P_aapl[i-W_LTMA:i]
    pf = P_aapl[i-W_STMA:i]
    ms = np.mean(ps)
    mf = np.mean(pf)
    
    dm[idx,0] = P_aapl[i]
    dm[idx,1] = ms
    dm[idx,2] = mf
    mr[idx] = (mf/ms) - 1
    
    if pos_aaple[idx-1] == 0 and mr[idx] > T1_aaple:
        # If the stock has no position and the spread of the ratio above the threshold T1
        # Short position is opened because we expect the ratio will go down to get back its normal behaviour (mean)
        pos_aaple[idx] = -1
        p_apple_snap = P_aapl[i]
    elif pos_aaple[idx-1] == 0 and mr[idx] < -T1_aaple:
        # If the stock has no position and the spread of the error signal below the threshold -T1
        # Long position is opened because we expect the stock will go up to get back its normal behaviour (mean)
        pos_aaple[idx] = 1
        p_apple_snap = P_aapl[i]
    elif pos_aaple[idx-1] == -1 and mr[idx] < T2_aaple:
        # If the stock is in short position and the spread is below threshold T2, 
        # that means it almost gets back to its normal behaviour and we close the position to generate profit
        pos_aaple[idx] = 0
        PNL_apple[idx] = -(P_aapl[i] - p_apple_snap)
    elif pos_aaple[idx-1] == 1 and mr[idx] > -T2_aaple:
        # If the stock is in long position and the spread is above threshold -T2, 
        # that means it almost gets back to its normal behaviour and we close the position to generate profit
        pos_aaple[idx] = 0
        PNL_apple[idx] = P_aapl[i] - p_apple_snap
    else:
        # Otherwise, no position is opened
        pos_aaple[idx] = pos_aaple[idx-1]
    
    idx += 1
    
plt_p, = plt.plot(dm[:,0], label='Price')
plt_ltma, = plt.plot(dm[:,1], label='LTMA')
plt_stma, = plt.plot(dm[:,2], label='STMA')
plt.legend(handles=[plt_p, plt_ltma, plt_stma])

plt.title('AAPL')
plt.ylabel('$')
plt.xlabel('Days')
plt.show()
    
plt.plot(mr)
plt.title('AAPL')
plt.ylabel('STMA/LTMA')
plt.xlabel('Days')
plt.show()

plt.plot(pos_aaple)
plt.title('AAPL')
plt.ylabel('Position')
plt.xlabel('Days')
plt.show()

plt.plot(np.cumsum(PNL_apple))
plt.title('AAPL')
plt.ylabel('PNL ($)')
plt.xlabel('Days')
plt.show()

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import metrics
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def getReturn(data):
    [R, C] = data.shape
    dataND = data[1:R]   
    dataTD = data[0:R-1]
    rts = np.log(np.divide(dataND, dataTD))
    return rts

def normalizeReturn(data):
    data = np.divide((data - np.mean(data, axis=0)), np.std(data, axis=0)) 
    return data

def getTrainedAutoencoderModel(hParams, trainData):
    # 5 layers of deep autoencoder is defined
    model = Sequential()
    model.add( Dense( hParams['hl1'], input_dim = hParams['inputOutputDimensionality'], activation = 'linear'))
    model.add( Dense( hParams['hl2'], activation = 'sigmoid'))
    model.add( Dense( hParams['hl1'], activation = 'sigmoid'))
    model.add( Dense( hParams['inputOutputDimensionality'], activation = 'linear',))
    
    model.compile(optimizer = 'adam', loss = 'mse')
    early_stopping = EarlyStopping( monitor = 'val_loss', patience = 10)
    checkpointer = ModelCheckpoint( filepath = 'synthetic_weights.hdf5', verbose=0, save_best_only = True)
    
    # Model is trained
    model.fit( trainData,
           trainData,
           batch_size = hParams['batchSize'], 
           epochs = hParams['epochs'],
           shuffle = True,
           callbacks = [early_stopping, checkpointer],
           validation_data = (trainData, trainData),
           verbose = 0 )
    
    return model

def genTradingSignals(pos, spread, Ts, Ss, Ps, N):
    # Based on the spread, signal generation is similar to pairs trading 
    newPos = np.zeros(shape=(1, N), dtype=int) # New position for each stock will be stored in this vector
    signal = np.zeros(shape=(1, N), dtype=int) # Generated signal to get to new position will be stored in this vector
    totalOpenedPos = 0
    
    for stock in range(0, N):
        if pos[stock] == Ps['NO_POSITION'] and spread[stock] > Ts['T1']:
            # If the stock has no position and the spread of the error signal above the threshold T1
            # Short position is opened because we expect the stock will go down to get back its normal behaviour (mean)
            signal[0, stock] = Ss['SHORT_SELL']
            newPos[0, stock] = Ps['SHORT']
        elif pos[stock] == Ps['NO_POSITION'] and spread[stock] < -Ts['T1']:
            # If the stock has no position and the spread of the error signal below the threshold -T1
            # Long position is opened because we expect the stock will go up to get back its normal behaviour (mean)
            signal[0, stock] = Ss['BUY']
            newPos[0, stock] = Ps['LONG']
        elif pos[stock] == Ps['SHORT'] and spread[stock] < Ts['T2']:
            # If the stock is in short position and the spread is below threshold T2, 
            # that means it almost gets back to its normal behaviour and we close the position to generate profit
            signal[0, stock] = Ss['BUY_TO_COVER']
            newPos[0, stock] = Ps['NO_POSITION']
        elif pos[stock] == Ps['LONG'] and spread[stock] > -Ts['T2']:
            # If the stock is in long position and the spread is above threshold -T2, 
            # that means it almost gets back to its normal behaviour and we close the position to generate profit
            signal[0, stock] = Ss['SELL']
            newPos[0, stock] = Ps['NO_POSITION']
        else:
            # Otherwise, no position is opened
            newPos[0, stock] = pos[stock]
          
        if newPos[0, stock] !=  Ps['NO_POSITION']:
            totalOpenedPos += 1
            
    return (newPos, signal, totalOpenedPos)

# This function executes the new orders, calculates the cost and P&L
def executeOrders(signal, numOfShares, currentPrice, lastPriceSnap, tParam, Ss, N):
    #P&L of each stock is stored in this vector
    PNLAsset = np.zeros(shape=(1, N), dtype=float)
    # This variables keeps the P&L of the portfolio
    PNLTotal = 0.0
    
    for stock in range(0, N):
        if signal[0, stock] == Ss['SHORT_SELL'] or signal[0, stock] == Ss['BUY']:
            # New position will be opened
            # Converting the investment amount to number of shares
            nOfS = round(tParam['investPerAsset'] / currentPrice[0, stock])
            # # Calculating the cost
            cost = nOfS * tParam['tradingCostPerShare']
            # Since new position is opening, there is no profit or loss, only the cost
            PNLAsset[0, stock] = -cost 
            PNLTotal += PNLAsset[0, stock]
            # Taking the snapshot of the purchase price
            lastPriceSnap[0, stock] = currentPrice[0, stock]
            # Storing the number of purchased shares
            if signal[0, stock] == Ss['SHORT_SELL']:
                numOfShares[0, stock] = -nOfS
            else:
                numOfShares[0, stock] = nOfS
        elif signal[0, stock] == Ss['SELL'] or signal[0, stock] == Ss['BUY_TO_COVER']:
            # Position will be closed
            # Calculating the profit or loss
            priceDiff = currentPrice[0, stock] - lastPriceSnap[0, stock]
            # Calculating the cost
            cost = numOfShares[0, stock] * tParam['tradingCostPerShare']
            # Updating the P&L for the stock
            PNLAsset[0, stock] = (priceDiff * numOfShares[0, stock]) - cost 
            # Updating the P&L of the portfolio
            PNLTotal += PNLAsset[0, stock]
            # Setting the number of stock to zero
            numOfShares[0, stock] = 0
            
    return (PNLAsset, lastPriceSnap, numOfShares, PNLTotal)

# Reading the price data from .xlsx file. Again need to change files location
xl = pd.ExcelFile('/users/dan/desktop/Data/DJI-AdjPrice-Train.xlsx')
dfP = xl.parse('Sheet1')
P = dfP.fillna(method="backfill")

# The strategy will be run for a given subset of data 
# We will be sliding over the data with a sample after each trading cycle
[R, C] = P.shape
N = C - 1

# Integer values to represent trade signals
Ss = {}
Ss['SELL'] = -1 # Sell the all the shares to close the position
Ss['SHORT_SELL'] = -2 # Enter a short position
Ss['BUY'] = 1 # Enter a long position
Ss['BUY_TO_COVER'] = 2 # Buy shares to close the short position

# Integer values to represent the current position of the assets
Ps = {}
Ps['NO_POSITION'] = 0
Ps['SHORT'] = -1
Ps['LONG'] = 1

# Parameters of deep autoencoder. 
# In this case, we only use 5-layer autoencoder but you can try different models
hParams = {}
hParams['inputOutputDimensionality'] = N
hParams['hl1'] = round(N / 2)
hParams['hl2'] = round(hParams['hl1'] / 2)
hParams['batchSize'] = 256
hParams['epochs'] = 10

# Window size to train the model
TrainingSetSize = 200 # 200
# Window size to calculate the spread of the error signal
ErrorSignalWindow = 50 # 50

# These are the threshold values that will be used in stat-arb strategy.
# The first threshold, T1, is used to open a position.
# The second threshold, T2, is used to close a position
Ts = {}
Ts['T1'] = 1.0 # Threshold for entering position
Ts['T2'] = 0.6 # Threshold for exiting the position

# Matrices are defined to store errors and spreads
errors = np.empty((0,N))
spread = np.empty((0,N))
# Matrices to store current position, num of shares bought, PNL per asset, PNL of portfolio
pos = np.zeros(shape=(1,N), dtype=int)
numOfShares = np.zeros(shape=(1,N), dtype=int)
lastPriceSnap = np.zeros(shape=(1,N), dtype=float)
PNLPerAsset = np.zeros(shape=(1,N), dtype=float)
PNLPortfolio = np.zeros(shape=(1, 1), dtype=float)
PNLPortfolioTotal = 0.0

# Parameters for the algorithmic trading strategy
tParam ={}
# Amount of money will be invested when entered a position.
# If you would like to invest a different amount, please update the number
tParam['investPerAsset'] = 5000 
# Trading cost ($) per share. You can update the number based on your cost 
tParam['tradingCostPerShare'] = 0.005 

idx = 0
# Historical prices will be analyzed in a sliding window fashion
# In each iteration, a subset of the data will be taken and analyzed 
for i in range(TrainingSetSize, R):
    # Takes the subset of the price data for the row indeces between (i-TrainingSetSize) and i
    pts = P[(i-TrainingSetSize):i].drop('Date', axis=1) 
    pts = np.array(pts)
    rts = getReturn(pts) # Log-returns are calculated
    rts = normalizeReturn(rts) # Log-returns are normalized
    lastReturn = rts[rts.shape[0]-1:rts.shape[0]] # Last row of log-returns is stored to be used later
        
    print("Training %d", idx)
    # Deep autoencoder is trained using the log-returns
    model = getTrainedAutoencoderModel(hParams, rts)
    
    # Last row of the log-returns is forward passed to measure its reconstruction error.
    # And, these error values are stored in a matrix to analyze it
    # This error will give us information about how the individual stock returns are deviating from the normal
    # In other words, it is checking if there is anything anomalous in the stock.
    # If there is an anomalous behavior in the stock, we will generate a position based on the expection of 
    # going back to normal
    predReturn = model.predict(lastReturn)
    err = np.subtract(lastReturn, predReturn)  
    errors = np.vstack((errors, err))
       
    idx += 1
    # After a certain number of iteration, the error signal is analzed to generate trade signals
    # Again, sliding window approach is also applied 
    if idx > ErrorSignalWindow:
        ce = np.shape(errors)[0]
        # Get the last rows of the error signal for a given window size in "ErrorSignalWindow" variable
        # Then, calculate the cumulative sum to see its mean reversion behavior
        errsCum = np.cumsum(errors[ce-ErrorSignalWindow:ce], axis=0)
        
        cec = np.shape(errsCum)[0]
        # Cumulative summed error signal is z-scored to understand the spread in normalized form
        errsNorm = np.divide((errsCum[len(errsCum)-1] - np.mean(errsCum)), np.std(errsCum))  
        spread = np.vstack((spread, errsNorm))
        
        # If the spread is below or above a certain threshold, trading signal is generated
        newPos, signal, totalOpenedPos = genTradingSignals(pos[len(pos)-1], spread[len(spread)-1], Ts, Ss, Ps, N) 
        # After the new positions are calculated, they are executed and P&L info is returned
        PNLAsset, lastPrice, nOfShares, PNLTotal = executeOrders(signal, 
                                                       numOfShares[len(numOfShares)-1:len(numOfShares)], 
                                                       pts[len(pts)-1:len(pts)], 
                                                       lastPriceSnap[len(lastPriceSnap)-1:len(lastPriceSnap)], 
                                                       tParam, 
                                                       Ss,
                                                       N)
        
        # State (current position, number of shares, P&L, etc) of the strategy is stored 
        pos = np.vstack((pos, newPos))
        lastPriceSnap = lastPrice
        numOfShares = np.vstack((numOfShares, nOfShares))
        PNLPerAsset = np.vstack((PNLPerAsset, PNLAsset))
        PNLPortfolio = np.vstack((PNLPortfolio, PNLTotal))
        PNLPortfolioTotal += PNLTotal
        TotalInvestedEquity = totalOpenedPos * tParam['investPerAsset']
        print("Invested Equity: %s, PNL Last Trade: %s, PNL Portfolio Total: %s" % (TotalInvestedEquity, PNLTotal, PNLPortfolioTotal))
        
        assetID = 3

plt.plot(spread[:,assetID])
plt.ylabel('Spread')
plt.xlabel('Days')
plt.show()

plt.plot(pos[:,assetID])
plt.ylabel('Positions (Long: 1, Short: -1, No Position: 0)')
plt.xlabel('Days')
plt.show()

# Annual Sharpe ratio is calculated
SR = (np.mean(PNLPortfolio, axis=0) / np.std(PNLPortfolio, axis=0)) * math.sqrt(252)
print("Annual SR: ", SR)

cumPNLPortfolio = np.cumsum(PNLPortfolio, axis=0)
plt.plot(cumPNLPortfolio)
plt.ylabel('PNL ($)')
plt.xlabel('Days')
plt.show()
