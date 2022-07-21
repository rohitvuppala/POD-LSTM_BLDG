#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:26:24 2021

author: Rohit K S S Vuppala
         Graduate Student, 
         Mechanical and Aerospace Engineering,
         Oklahoma State University.

@email: rvuppal@okstate.edu

"""

# import necessary libraries
"Generate seed for random number generator to behave same"
from sys import exit
import os
import random
#random.seed(99)
import numpy as np
#np.random.seed(99)
import datetime
#%load_ext tensorboard

#mp.rcParams["contour.corner_mask"] = False

import time as tm

import matplotlib as mp
#%matplotlib inline
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mp
import pylab


import tensorflow as tf
tf.float16

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import h5py 
from tqdm import tqdm as tqdm

from numpy import linalg as la # For SVD in basis construction
from netCDF4 import Dataset

from scipy.io import loadmat,savemat
from hurst import compute_Hc