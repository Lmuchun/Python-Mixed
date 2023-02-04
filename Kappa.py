# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:56:48 2020

@author: Lenovo
"""
import numpy as np 
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
from sklearn.metrics import cohen_kappa_score



path1 = "C:\\Users\\Lenovo\\Desktop\\dg2006true.tif"
path2 = "C:\\Users\\Lenovo\\Desktop\\CA.tif"
real_image = gdal.Open(path1,GA_ReadOnly)
pre_image = gdal.Open(path2,GA_ReadOnly)
try:
    cols = real_image.RasterXSize
    rows = real_image.RasterYSize
except:
    print ("Error: It is not an image")
real_classified = real_image.ReadAsArray(0, 0, cols, rows)
pre_classified = pre_image.ReadAsArray(0, 0, cols, rows)
kappa = cohen_kappa_score(np.array(real_classified), np.array(pre_classified))
print(kappa)