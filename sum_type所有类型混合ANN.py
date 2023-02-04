# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:59:36 2020

@author: Lenovo
"""
import numpy as np
from numpy import mat
import pandas as pd
import xlrd
import matplotlib.pyplot as plt 

from keras.optimizers import SGD,Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
import keras.layers as layers
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
from sklearn import preprocessing


#####参数设置
epoch = 5	#迭代次数
inputnum = 8		#输入节点数s
midnum = 12		#隐层节点数
outputnum = 1		#输出节点数
trainX = 0.03		#训练数据比例
datannum = 8000	#样本总数
#输入数据文件地址
inputfile = 'C:\\Users\\Lenovo\\Desktop\\1.xls'   		 	
#输出预测的文件地址
outputfile = 'C:\\Users\\Lenovo\\Desktop\\alldata.xlsx' 	

#输出预测的文件地址
output = 'C:\\Users\\Lenovo\\Desktop\\output1.xlsx' 		
#模型保存地址
modelfile = 'E:\\log_data\\modelweight.model' 	

#因子所在列
factor = [2,3,4,5,6,7,8,9]														
#目标所在列
label = [10,11,12,13,14] 


in_path = "C:\\Users\\Lenovo\\Desktop\\dg2001coor.tif"
out_path= "C:\\Users\\Lenovo\\Desktop\\dg2006true.tif"
raw_image = gdal.Open(in_path,GA_ReadOnly)

try:
    cols = raw_image.RasterXSize
    rows = raw_image.RasterYSize
    bands = raw_image.RasterCount
except:
    print ("Error: It is not an image")
# Get the spatial reference of the input image
projInfo = raw_image.GetProjection()
transInfo = raw_image.GetGeoTransform()
train=raw_image.ReadAsArray(0, 0, cols, rows)
result=gdal.Open(out_path,GA_ReadOnly).ReadAsArray(0, 0, cols, rows)

factor_path1="C:\\Users\\Lenovo\\Desktop\\Aspect.tif"
factor_path2="C:\\Users\\Lenovo\\Desktop\\dem_dg.tif"
factor_path3="C:\\Users\\Lenovo\\Desktop\\distohighway.tif"
factor_path4="C:\\Users\\Lenovo\\Desktop\\distorailway.tif"
factor_path5="C:\\Users\\Lenovo\\Desktop\\distoroad.tif"
factor_path6="C:\\Users\\Lenovo\\Desktop\\distotown.tif"
factor_path7="C:\\Users\\Lenovo\\Desktop\\slope.tif"
factor_path8="C:\\Users\\Lenovo\\Desktop\\tocity_dg.tif"
factor_list=[factor_path1,factor_path2,factor_path3,factor_path4,factor_path5,factor_path6,factor_path7,factor_path8]

for i in range(8):
    factor_image=gdal.Open(factor_list[i],GA_ReadOnly)
    factor_list[i]=factor_image.ReadAsArray(0, 0, cols, rows)
    
all_factor=np.dstack((train,result,factor_list[0],factor_list[1],factor_list[2],factor_list[3],factor_list[4],factor_list[5],factor_list[6],factor_list[7]))
all_factor=np.reshape(all_factor,(-1,10))
all_factor[all_factor<0] = 0
all_factor[all_factor>10000] = 0
idx1 = np.argwhere(np.all(all_factor[..., :1] ==0, axis=1))
all_factor=np.delete(all_factor, idx1, 0)
length=np.shape(all_factor)[0] 
z = np.zeros((length, 5)) 
data=np.append(all_factor, z,axis=1)
# index=idx1[0]
# type1=all_factor[index]
# for i in idx1:
#     type1=np.vstack([type1, all_factor[i]])
# all_factor = pd.DataFrame(all_factor)
# all_factor.to_excel(outputfile) 
for i in data:
    if i[1]==1:
        i[10]=1
    if i[1]==2:
        i[11]=1
    if i[1]==3:
        i[12]=1
    if i[1]==4:
        i[13]=1
    if i[1]==5:
        i[14]=1
data=pd.DataFrame(data)
data = data.reindex(np.random.permutation(data.index))
trainnum =int(len(data)*trainX)
testnum = datannum - trainnum
data_train = data[0:trainnum].copy() 
data_test = data[trainnum:datannum].copy()
data_std = data.max() - data.min()
data_train = (data_train - data.min()) / data_std
data_test = (data_test - data.min()) / data_std
x_train = data_train[factor].values
y_train = data_train[label].values



#构建模型
model = Sequential()
#初始化权值阈值
model.add(Dense(units=12,
                input_dim=8,
                kernel_initializer='normal'))
model.add(Dense(units=5,
                kernel_initializer='normal',
                activation='sigmoid'))
sgd=SGD(lr=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                              patience=10, min_lr=0.005)
train_history=model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        callbacks=[reduce_lr],
                        epochs=500,
                        batch_size=10)

model.save_weights(modelfile)
# # 预测数据
x_test = data_test[factor]
y_test = data_test[label]
#评估模型
loss,accuracy=model.evaluate(x_test,y_test)
print('loss:',loss)
print('accuracy:',accuracy)
# #写入训练结果
# proba=model.predict_proba(x_test)
# data_test[u'L1_pred'] = proba[:, 0]
# data_test[u'L2_pred'] = proba[:, 1]
# data_test[u'L3_pred'] = proba[:, 2]
# data_test[u'L4_pred'] = proba[:, 3]
# data_test[u'L5_pred'] = proba[:, 4]
# data_test.to_excel(output) 
