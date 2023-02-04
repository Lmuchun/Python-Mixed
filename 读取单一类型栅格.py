# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:16:44 2020

@author: Lenovo
"""

import numpy as np
from numpy import mat
import pandas as pd
import xlrd
import matplotlib.pyplot as plt 
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
import keras.layers as layers
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
from sklearn import preprocessing

in_path = "C:\\Users\\Lenovo\\Desktop\\dg2006true.tif"
out_path="C:\\Users\\Lenovo\\Desktop\\2006_raster5.tif"
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
index=np.where(train==5)
c1=np.vstack((index[0],index[1]))
classified=np.full((rows, cols), 2147483647)
col1=c1.shape[1]
i=0
j=0
while i<col1:
    row0=c1[0,i]
    col0=c1[1,i]
    classified[row0,col0]=5
    i+=1
#   write result to disk
driver = gdal.GetDriverByName("GTiff")
outDataset = driver.Create(out_path,
                    cols,rows,bands,GDT_Float32)
outDataset.SetProjection(projInfo)
outDataset.SetGeoTransform(transInfo)
CA = outDataset.GetRasterBand(1)
CA.SetNoDataValue(2147483647)
CA.WriteArray(classified[:,:])
CA.FlushCache()

CA = None
outDataset = None