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
inputfile = 'C:\\Users\\Lenovo\\Desktop\\data1.xlsx'  
predictfile='C:\\Users\\Lenovo\\Desktop\\predict.xlsx' 	
trainX=0.8
b_size = 10
max_epochs = 400

data = pd.read_excel(inputfile,index='Date') 
features = list(data.columns)
datannum=data.shape[0]
#因子所在列
label = ['Y0','Y1','Y2','Y3','Y4','Y5','Y6'] 	
factor = ['T1','T2','T3','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17']	
data = data.reindex(np.random.permutation(data.index))
trainnum =int(len(data)*trainX)
data_train = data[0:trainnum].copy() 
data_test = data[trainnum:datannum].copy()
data_std = data.max() - data.min()
data_train = (data_train - data.min()) / data_std
data_test = (data_test - data.min()) / data_std
x_train = data_train[factor].values
y_train = data_train[label].values
  # 预测数据
x_test = data_test[factor].values
y_test = data_test[label].values

init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam()
model = K.models.Sequential()
model.add(K.layers.Dense(units=40, input_dim=20, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(K.layers.Dense(units=40, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(K.layers.Dense(units=40, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(K.layers.Dense(units=7, kernel_initializer='normal', activation='softmax'))
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10)
h = model.fit(x_train, y_train, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1,
                  validation_split=0.3)
print("Training finished \n")

eval = model.evaluate(x_test, y_test, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
      % (eval[0], eval[1] * 100) )
    
proba=model.predict_proba(x_test)
data_test[u'L1_pred'] = proba[:, 0]
data_test[u'L2_pred'] = proba[:, 1]
data_test[u'L3_pred'] = proba[:, 2] 
data_test[u'L4_pred'] = proba[:, 3]
data_test[u'L5_pred'] = proba[:, 4]
data_test[u'L6_pred'] = proba[:, 5]
data_test[u'L7_pred'] = proba[:, 6]
data_test.to_excel(predictfile) 