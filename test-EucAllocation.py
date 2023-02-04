# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:06:00 2020

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 22:12:00 2020

@author: Lenovo
"""

import numpy as np 
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
import matplotlib.pyplot as plt
from pylab import *
import gc
import os
import random
import queue
import copy
# Open a pre-classified image
in_path = "C:\\Users\\Lenovo\\Desktop\\dg2001coor.tif"
prob_path="C:\\Users\\Lenovo\\Desktop\\Probability-of-occurrence.tif"
restric_path="C:\\Users\\Lenovo\\Desktop\\restrictedarea.tif"

f = open("C:\\Users\\Lenovo\\Desktop\\out.txt", "w")  
f.truncate()
gdal.AllRegister()
raw_image = gdal.Open(in_path,GA_ReadOnly)
prob_image=gdal.Open(prob_path,GA_ReadOnly)
restric_image=gdal.Open(restric_path,GA_ReadOnly)
try:
    cols = raw_image.RasterXSize
    rows = raw_image.RasterYSize
    bands = raw_image.RasterCount
except:
    print ("Error: It is not an image")
try:
    cols1 = prob_image.RasterXSize
    rows1 = prob_image.RasterYSize
    bands1 = prob_image.RasterCount
except:
    print ("Error: It is not an image")
try:
    cols2 = restric_image.RasterXSize
    rows2 = restric_image.RasterYSize
    bands2 = restric_image.RasterCount
except:
    print ("Error: It is not an image")

# Get the spatial reference of the input image
projInfo = raw_image.GetProjection()
transInfo = raw_image.GetGeoTransform()


# Read the image as an array
pre_classified = raw_image.ReadAsArray(0, 0, cols, rows)
prob_value=prob_image.ReadAsArray(0, 0, cols1, rows1)
restric_value=restric_image.ReadAsArray(0, 0, cols2, rows2) #约束条件

index=np.where(pre_classified==1)
c1=np.vstack((index[0],index[1]))

change_classified=np.full((rows, cols), 2147483647)
distance=np.full((rows, cols), 2147483647)
col1=c1.shape[1]
i=0
j=0
sum=[0,0,0,0,0]
s=0
while i<col1:
    row0=c1[0,i]
    col0=c1[1,i]
    ll = pre_classified[row0-1:row0+2,col0-1:col0+2]
    if 2 in ll:
        distance[row0,col0]=1
        if prob_value[1][row0,col0]>0.8 and prob_value[1][row0,col0]>prob_value[0][row0,col0]:
            change_classified[row0,col0]=2
            s+=1        
    i+=1
        
   

#   write result to disk
driver = gdal.GetDriverByName("GTiff")
outDataset = driver.Create("E:/out.tif",
                    cols,rows,bands,GDT_Float32)
outDataset.SetProjection(projInfo)
outDataset.SetGeoTransform(transInfo)
CA = outDataset.GetRasterBand(1)
CA.SetNoDataValue(2147483647)
CA.WriteArray(change_classified[:,:])
CA.FlushCache()

CA = None
outDataset = None
