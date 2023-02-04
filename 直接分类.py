# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:51:39 2020

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
inputfile = 'C:\\Users\\Lenovo\\Desktop\\test.xlsx' 	 
predictfile='C:\\Users\\Lenovo\\Desktop\\predict.xlsx' 	
trainX=0.8
b_size = 10
max_epochs = 400

data = pd.read_excel(inputfile,index='Date') 
datannum=data.shape[0]
#因子所在列
label = ['T4'] 	
factor = ['T1','T2','T3','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']	
fact=['T1','T2','T3','T4']
data = data.reindex(np.random.permutation(data.index))
trainnum =int(len(data)*trainX)
data_train = data[0:trainnum].copy() 
data_test = data[trainnum:datannum].copy()
data_std = data.max() - data.min()
data_min= data.min()
data_min[fact]=0
data_std[fact]=1
data_train = (data_train- data_min) / data_std
data_test = (data_test - data_min) / data_std
x_train = data_train[factor].values
y_train = data_train[label].values
  # 预测数据
x_test = data_test[factor].values
y_test = data_test[label].values

init = K.initializers.glorot_uniform(seed=1)
model = K.models.Sequential()
model.add(K.layers.Dense(units=50, input_dim=13, kernel_initializer='normal'))
model.add(K.layers.Dense(units=50, kernel_initializer='normal'))
model.add(K.layers.Dense(units=1, kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
h = model.fit(x_train, y_train, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1,
                  validation_split=0.4,callbacks=[reduce_lr])

eval = model.evaluate(x_test, y_test, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
      % (eval[0], eval[1] * 100) )