# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 21:17:19 2020

@author: Lenovo
"""

import os
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.layers import Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#输入数据文件地址
inputfile = 'C:\\Users\\Lenovo\\Desktop\\data3.xlsx'  

trainX=0.8
b_size = 20
max_epochs = 400

data = pd.read_excel(inputfile,index='Date') 
datannum=data.shape[0]
#因子所在列
label = ['Y0','Y1','Y2','Y3','Y4','Y5','Y6'] 	
factor = ['T1','T2','T3','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17']	

data = data.reindex(np.random.permutation(data.index))
trainnum =int(len(data)*trainX)
data_std = data.max() - data.min()
print(data.max())
print(data.min())