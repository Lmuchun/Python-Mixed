# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:59:36 2020

@author: Lenovo
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ReduceLROnPlateau


#####参数设置
epoch = 5000	#迭代次数
inputnum = 16		#输入节点数s
midnum = 20		#隐层节点数
outputnum = 1		#输出节点数
learnrate =0.02	#学习率
Terror = 1e-3		#迭代中止条件
datannum = 79370	#样本总数
trainX = 0.8		#训练数据比例

#输入数据文件地址
inputfile = 'C:\\Users\\Lenovo\\Desktop\\sub_all.xlsx'   		 	
#输出预测的文件地址
outputfile = 'C:\\Users\\Lenovo\\Desktop\\output.xlsx' 		
#模型保存地址
modelfile = 'C:\\Users\\Lenovo\\Desktop\\modelweight.model' 	
#因子所在列
factor = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15']														
#目标所在列
label = ['L1','L2','L3','L4','L5','L6','L7','L8'] 	

#初始处理
data = pd.read_excel(inputfile,index='Date') 
data = data.reindex(np.random.permutation(data.index))
trainnum =int(len(data)*trainX)
data_train = data[0:trainnum].copy() 
data_test = data[trainnum:datannum].copy()
x_train = data_train[factor].values
y_train = data_train[label].values
# 预测数据
x_test = data_test[factor].values
y_test = data_test[label].values
#构建模型
model = Sequential()
#初始化权值阈值
model.add(Dense(units=30,
                input_dim=15,
                activation='relu'))
model.add(Dense(units=8,
                activation='softmax'))

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='adam',
              metrics=['categorical_accuracy'])
train_history=model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        epochs=300,
                        batch_size=60)
model.save_weights(modelfile)
#评估模型
print("模型评估")
loss,accuracy=model.evaluate(x_test,y_test)
print('loss:',loss)
print('accuracy:',accuracy)

#写入训练结果
proba=model.predict(x_test)
data_test[u'L0_pred'] = proba[:, 0]
data_test[u'L1_pred'] = proba[:, 1]
data_test[u'L2_pred'] = proba[:, 2]
data_test[u'L3_pred'] = proba[:, 3]
data_test[u'L4_pred'] = proba[:, 4]
data_test[u'L5_pred'] = proba[:, 5]
data_test[u'L6_pred'] = proba[:, 6]
data_test[u'L7_pred'] = proba[:, 7]
data_test.to_excel(outputfile) 






# summarize history for loss

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

    
