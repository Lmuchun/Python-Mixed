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
in_path = "C:\\Users\\Lenovo\\Desktop\\wh2010_refy.tif"
gdal.AllRegister()
raw_image = gdal.Open(in_path,GA_ReadOnly)
try:
    cols = raw_image.RasterXSize
    rows = raw_image.RasterYSize
    bands = raw_image.RasterCount
except:
    print ("Error: It is not an image")
# Read the image as an array
pre_classified = raw_image.ReadAsArray(0, 0, cols, rows)
# Get the spatial reference of the input image
projInfo = raw_image.GetProjection()
transInfo = raw_image.GetGeoTransform()
SUM=np.sum(pre_classified>0) 
T1=np.sum(pre_classified==1) 
T2=np.sum(pre_classified==2) 
T3=np.sum(pre_classified==3) 
T4=np.sum(pre_classified==4) 
T5=np.sum(pre_classified==5) 
T6=np.sum(pre_classified==6) 
T7=np.sum(pre_classified==7) 
print(SUM)
i=0
j=0
f=np.zeros((rows,cols))
while i<rows-2:
    print(i)
    while j<cols-2:

        Y1=0
        Y2=0
        Y3=0
        Y4=0
        Y5=0
        Y6=0
        Y7=0
        for n in range(-2,3) :
            for m in range(-2,3) :
                # if pre_classified[i+n,j+m] == 1:
                    # Y1 = Y1 + 1
                # if pre_classified[i+n,j+m] == 2:
                    # Y2 = Y2 + 1
                # if pre_classified[i+n,j+m] == 3:
                    # Y3 = Y3 + 1
                # if pre_classified[i+n,j+m] == 4:
                    # Y4 = Y4 + 1
                # if pre_classified[i+n,j+m] == 5:
                    # Y5 = Y5 + 1
                if pre_classified[i+n,j+m] == 6:
                    Y6 = Y6 + 1
                # if pre_classified[i+n,j+m] == 7:
                    # Y7 = Y7 + 1
        f[i,j]=(Y6/25)/(T6/SUM)
        # f[1][i,j]=(Y2/9)/(T2/SUM)
        # f[2][i,j]=(Y3/9)/(T3/SUM)
        # f[3][i,j]=(Y4/9)/(T4/SUM)
        # f[4][i,j]=(Y5/9)/(T5/SUM)
        # f[5][i,j]=(Y6/9)/(T6/SUM)
        # f[6][i,j]=(Y7/9)/(T7/SUM)
        j=j+1
    j = 1
    i = i + 1
        
driver = gdal.GetDriverByName("GTiff")
outDataset = driver.Create("E:/25band6qq.tif",
                    cols,rows,1,GDT_Float32)
outDataset.SetProjection(projInfo)
outDataset.SetGeoTransform(transInfo)
outBand = outDataset.GetRasterBand(1)
outBand.WriteArray(f)
outBand.SetNoDataValue(0)

#逐波段写入栅格
# for i in range(7):
#     outBand = outDataset.GetRasterBand(i + 1)
#     outBand.WriteArray(f[i])
outDataset = None
del outDataset,outBand
