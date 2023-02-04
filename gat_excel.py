# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:30:23 2020

@author: Lenovo
"""

1# -*- coding: utf-8 -*-
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
outputfile = 'C:\\Users\\Lenovo\\Desktop\\new_type5.xlsx' 	

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
idx2 = np.argwhere(np.all(data[..., :1] ==5, axis=1))
ind=idx2[0]
type4=data[ind]
for i in idx2:
    type4=np.vstack([type4, data[i]])
type4 = pd.DataFrame(type4)
type4.to_excel(outputfile) 