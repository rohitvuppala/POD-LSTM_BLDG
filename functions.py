#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:50:21 2021

@author: Rohit K S S Vuppala
@email : rvuppal@okstate.edu
         Graduate Student
         Mechanical and Aerospace Engineering
         Oklahoma State University


"""


from sys import exit
import random
random.seed(99)
import numpy as np
np.random.seed(99)

import time as tm

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mp

#Corner masking for NAN Values in plots
mp.rcParams["contour.corner_mask"] = False

#Import CMOcean colormaps
import cmocean

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import h5py 
from tqdm import tqdm as tqdm

from numpy import linalg as la # For SVD in basis construction

#%%
"""
Define function for square of singular values calc
#Input : A - matrix of data 
#Output: S2  - Square of singular values 
         RIC - Relative Importance Index (%)
"""  
def calc_S2(A):
    nx,ny = A.shape   #For debugging
    
    U,S,Vh= la.svd(A,full_matrices=False)
    S2     = S**2
    
    return S2

"""
Define function for Basis construction
#Input : A - matrix of data 
         nr - number of modes required
#Output: Phi - Basis vectors
         S2  - Square of singular values 
         RIC - Relative Importance Index (%) 
"""    
def con_basis(A,nr):
    nx,ny = A.shape   #For debugging
    
    U,S,Vh= la.svd(A,full_matrices=False)
    Phi   = U[:,:nr] 
    S2     = S**2
    
    #
    #RIC = sum(squares of R modes))/sum(square of all modes) * 100
    #
    print("U",U.shape)
    print("S",S.shape)
    print("Vh",Vh.shape)
    RIC  = sum(S2[:nr])/sum(S2) * 100.0
    return Phi,S2,RIC

"""
Define function for Projection (to get the coeffcients in the basis) 
Note: u = Phi*a.T
#Input : u  - Data Matrix 
         Phi- Basis vectors 
#Output: a  - Coefficient matrix
"""     
def get_coeff(u,Phi):
    a = np.dot(u.T,Phi)  
    return a


"""
Define function for Projection (to get the coeffcients in the basis) 
Note: u = Phi*a.T
#Input : a  - Coeffcient matrix
         Phi- Basis vectors 
#Output: u  - Reconstructed Data Matrix 
"""     
def recon_data(a,Phi):
    u = np.dot(Phi,a.T)  
    return u

"""
Define function to create training data for lstm
#Input : train_set - Training data set
         m         - Size of the data set
         n         - Number of modes
         lookback  - number of time steps used for LSTM
#Output: xtrain    - Current time data
         ytrain    - Corresponding future time
       
"""
def create_training_data_lstm(training_set, m, n, lookback):
    ytrain = [training_set[i+1] for i in range(lookback-1,m-1)]
    ytrain = np.array(ytrain)
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j]))
        xtrain[i] = a

    return xtrain, ytrain

"""
Define function to find a metric(cost function) while determining coefficients for LSTM training
#Input : y_true - True coefficients 
         y_pred - Predicted coeffcients
#Output: val    - the value of the metric
"""
def coeff_metric(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    val    = 1 - SS_res/(SS_tot + K.epsilon())
    return val
#%%
"""
Define LSTM model network
"""

def model_lstm(nl,nn,lookback,nr,drp,rdrp):
    input = Input(shape=(lookback,nr))
    
    for i in range(nl):
        if (i==0):
            x1 = LSTM(nn, return_sequences=True,activation='tanh', kernel_initializer='glorot_normal',dropout=0.0,recurrent_dropout=rdrp)(input)
        elif (i<nl-1):
            x1 = LSTM(nn, return_sequences=True,activation='tanh', kernel_initializer='glorot_normal',dropout=0.0,recurrent_dropout=rdrp)(x1)      
        else:
            x1 = LSTM(nn, return_sequences=False,activation='tanh', kernel_initializer='glorot_normal',dropout=0.0,recurrent_dropout=rdrp)(x1)      

            
    x1 = Dense(nr, activation='linear')(x1)
    return Model(input, x1)

#%%
"""
Detrend the training data for specified order and return (data,coefficients)
#Input : a   - Data of size(nl,nm)    
         nl  - Number of snapshots
         nm  - Number of modes
         nord- Order for detrending
#Output: a   - Detrended data
         p   - polynomial coefficients
"""

def detrend_train(a,nm,nl,nord):
    p = np.zeros((nm,nord+1))
    pspace = np.linspace(1,nl,nl)
    
    b = np.copy(a)
    for i in range(nm):
        p[i,:] = np.polyfit(pspace,b[:,i],nord)
    
    
        for j in range(nord):
            b[:,i] = b[:,i] -  p[i,j]*(pspace**(nord - j)) 
    
        b[:,i] = b[:,i] - p[i,nord]

    return (b,p)
#%%
"""
Detrend the prediction data for specified order and return (data)
#Input : a   - Data of size(nl,nm)    
         nb  - Starting number of snapshots
         ne  - Ending number of snapshots 
         nm  - Number of modes
         nord- Order for detrending
         p   - polynomial coefficients 
#Output: a   - Detrended data
"""

def detrend_pred(a,p,nm,nb,ne,nord):
    pspace = np.linspace(nb,ne,ne-nb+1)
    
    b = np.copy(a)
    for i in range(nm): 
        for j in range(nord):
            b[:,i] = b[:,i] -  p[i,j]*(pspace**(nord - j)) 
    
        b[:,i] = b[:,i] - p[i,nord]
    
    
    return (b)
#%%
"""
Retrend the data for specified order and return (data)
#Input : a   - Data of size(nl,nm)    
         nb  - Starting number of snapshots
         ne  - Ending number of snapshots 
         nm  - Number of modes
         nord- Order for detrending
         p   - polynomial coefficients 
#Output: a   - Retrended data
"""

def retrend(a,p,nm,nb,ne,nord):
    pspace = np.linspace(nb,ne,ne-nb+1)
    
    b = np.copy(a)
    for i in range(nm): 
        for j in range(nord):
            b[:,i] = b[:,i] +  p[i,j]*(pspace**(nord - j)) 
    
        b[:,i] = b[:,i] + p[i,nord]
    
    return (b)

