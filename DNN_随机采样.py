# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:22:19 2020

@author: Lenovo
"""
import os
import numpy as np
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
import random
import pandas as pd
import xlrd

root_path="C:\\Users\\Lenovo\\Desktop\\wh_test_data\\year"
factor_path="C:\\Users\\Lenovo\\Desktop\\wh_test_data\\factor"
outputfile = 'C:\\Users\\Lenovo\\Desktop\\wh_test_data\\test11111.xlsx' 	
year_list = []
factor_list=[]
files = os.listdir(root_path)

for filename in files:
    path=root_path+"\\"+filename
    raw_image = gdal.Open(path,GA_ReadOnly)
    try:
        cols = raw_image.RasterXSize
        rows = raw_image.RasterYSize
        temp_array=raw_image.ReadAsArray(0, 0, cols, rows)
        year_list.append(temp_array)
    except:
        print ("Error: It is not an image")

fac_files = os.listdir(factor_path)
for filename in fac_files:
    path=factor_path+"\\"+filename
    raw_image = gdal.Open(path,GA_ReadOnly)
    try:
        cols = raw_image.RasterXSize
        rows = raw_image.RasterYSize
        temp_array=raw_image.ReadAsArray(0, 0, cols, rows)
        year_list.append(temp_array)
    except:
        print ("Error: It is not an image")
#随机采样       
def get_random(row,col,num):
    output=[]
    all_num=[0,0,0,0,0,0,0]
    while min(all_num)!=num:
        temp=[]
        a=random.randint(0,row)
        b=random.randint(0,col)
        for rast in year_list:
            t1=rast[a,b]
            temp.append(t1)
        if temp[3]<0:
            continue
        if temp[3]==1 and all_num[0]<num:
            all_num[0]+=1
            output.extend(temp)
        if temp[3]==2 and all_num[1]<num:
            all_num[1]+=1   
            output.extend(temp)
        if temp[3]==3 and all_num[2]<num:
            all_num[2]+=1
            output.extend(temp)
        if temp[3]==4 and all_num[3]<num:
            all_num[3]+=1
            output.extend(temp)
        if temp[3]==5 and all_num[4]<num:
            all_num[4]+=1
            output.extend(temp)
        if temp[3]==6 and all_num[5]<num:
            all_num[5]+=1
            output.extend(temp)
        if temp[3]==7 and all_num[6]<num:
            all_num[6]+=1
            output.extend(temp)
    return output
out=get_random(2930, 2930, 2000)           
final=np.array(out).reshape(14000,21)
final = pd.DataFrame(final)
final.to_excel(outputfile) 