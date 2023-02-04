# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:35:45 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt 
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
import numpy.matlib 
#模型保存地址
model1 = 'C:\\Users\\Lenovo\\Desktop\\type1_modelweight.h5' 	
model2 = 'C:\\Users\\Lenovo\\Desktop\\type2_modelweight.h5' 
model3 = 'C:\\Users\\Lenovo\\Desktop\\type3_modelweight.h5' 
model4 = 'C:\\Users\\Lenovo\\Desktop\\type4_modelweight.h5' 
model5 = 'C:\\Users\\Lenovo\\Desktop\\type5_modelweight.h5' 
in_path = "C:\\Users\\Lenovo\\Desktop\\dg2001coor.tif"
prob_path="C:\\Users\\Lenovo\\Desktop\\Probability.tif"

load_model1 = load_model(model1)
load_model2 = load_model(model2)
load_model3 = load_model(model3)
load_model4 = load_model(model4)
load_model5 = load_model(model5)
np.set_printoptions(precision=4)

gdal.AllRegister()
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
# Read the image as an array
pre_classified = raw_image.ReadAsArray(0, 0, cols, rows)

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
all_factor=np.dstack((factor_list[0],factor_list[1],factor_list[2],factor_list[3],factor_list[4],factor_list[5],factor_list[6],factor_list[7]))
all_factor=np.reshape(all_factor,(-1,8))
all_factor[all_factor<0] = 0
all_factor = pd.DataFrame(all_factor)
data_std=all_factor.max()-all_factor.min()

def getInput(m,n):
    input_fac= np.empty([8,1], dtype = float) 
    index=0
    for row in input_fac:
        temp=factor_list[index]   
        input_fac[index,0]=temp[m,n]/data_std[index]
        index+=1
    input_fac=input_fac.reshape(1,8)
    return input_fac    

pp=np.zeros((5,rows, cols))

# factor=getInput(4, 204)
# predicted = load_model1.predict(factor)
# print("Using model to predict species for features: ")
# print(factor)
# print("\nPredicted softmax vector is: ")
# print(predicted)

i=0
j=0
while i < rows - 1:    
    print(i)
    while j < cols - 1:  
        if pre_classified[i,j]>5:
            j=j+1
            continue
        if pre_classified[i,j]==1:
            factor=getInput(i, j)
            predicted = load_model1.predict(factor)
            pp[0][i][j]=predicted[0,0]
            pp[1][i][j]=predicted[0,1]
            pp[2][i][j]=predicted[0,2]
            pp[3][i][j]=predicted[0,3]
            pp[4][i][j]=predicted[0,4]
        if pre_classified[i,j]==2:
            factor=getInput(i, j)
            predicted = load_model2.predict(factor)
            pp[0][i][j]=predicted[0,0]
            pp[1][i][j]=predicted[0,1]
            pp[2][i][j]=predicted[0,2]
            pp[3][i][j]=predicted[0,3]
            pp[4][i][j]=predicted[0,4]
        if pre_classified[i,j]==3:
            factor=getInput(i, j)
            predicted = load_model3.predict(factor)
            pp[0][i][j]=predicted[0,0]
            pp[1][i][j]=predicted[0,1]
            pp[2][i][j]=predicted[0,2]
            pp[3][i][j]=predicted[0,3]
            pp[4][i][j]=predicted[0,4]
        if pre_classified[i,j]==4:
            factor=getInput(i, j)
            predicted = load_model4.predict(factor)
            pp[0][i][j]=predicted[0,0]
            pp[1][i][j]=predicted[0,1]
            pp[2][i][j]=predicted[0,2]
            pp[3][i][j]=predicted[0,3]
            pp[4][i][j]=predicted[0,4]
        if pre_classified[i,j]==5:
            factor=getInput(i, j)
            predicted = load_model5.predict(factor)
            pp[0][i][j]=predicted[0,0]
            pp[1][i][j]=predicted[0,1]
            pp[2][i][j]=predicted[0,2]
            pp[3][i][j]=predicted[0,3]
            pp[4][i][j]=predicted[0,4]
        j=j+1
    j = 1
    i = i + 1
    
    
driver = gdal.GetDriverByName("GTiff")
outDataset = driver.Create("E:/proba.tif",
                    cols,rows,5,GDT_Float32)
outDataset.SetProjection(projInfo)
outDataset.SetGeoTransform(transInfo)
#逐波段写入栅格
for i in range(5):
    outBand = outDataset.GetRasterBand(i + 1)
    outBand.WriteArray(pp[i])
outDataset = None
del outDataset,outBand