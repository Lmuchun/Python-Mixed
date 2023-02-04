# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 21:52:38 2020

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

prob_path="E:\\25band2.tif"
prob_image=gdal.Open(prob_path,GA_ReadOnly)
# Get the spatial reference of the input image
projInfo = prob_image.GetProjection()
transInfo = prob_image.GetGeoTransform()
try:
    cols = prob_image.RasterXSize
    rows = prob_image.RasterYSize
    bands = prob_image.RasterCount
except:
    print ("Error: It is not an image")

prob_value=prob_image.ReadAsArray(0, 0, cols, rows)
driver = gdal.GetDriverByName("GTiff")
outDataset = driver.Create("E:/new_proba.tif",
                    cols,rows,5,GDT_Float32)
outDataset.SetProjection(projInfo)
outDataset.SetGeoTransform(transInfo)
#逐波段写入栅格
for i in range(5):
    outBand = outDataset.GetRasterBand(i + 1)
    outBand.WriteArray(prob_value[i])
    outBand.SetNoDataValue(0)
outDataset = None
del outDataset,outBand