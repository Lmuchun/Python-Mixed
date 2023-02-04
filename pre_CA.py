# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:52:04 2020

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:56:08 2020

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:34:19 2020

@author: Lenovo
"""

import numpy as np 
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
import random

#参数
N=3 #摩尔邻域
td1=np.matrix([[46989,54427,59899,49516,38090],[80016,54427,43599,42433,28653]], dtype=int)
td2=np.matrix([[1,0,0,0,0],[0,1,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,0,1,1,1]], dtype=int)
td3=np.matrix([1,0.9,0.5,1,0.1])
pre_Intertia=[1,1,1,1,1]
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
now_classified=np.zeros((rows,cols))
q =[pre_classified,pre_classified]

i = 0
j = 0
M = 1
 #类型1：城市
 #类型2：水体
 #类型3：耕地
 #类型4：林地
 #类型5：果园
def wheel(p,self_type):
    sum_p=np.sum(p)
    p1=p[0,0]/sum_p
    p2=p1+p[0,1]/sum_p
    p3=p2+p[0,2]/sum_p
    p4=p3+p[0,3]/sum_p
    p5=p4+p[0,4]/sum_p
    change_type=0
    temp_prob = random.uniform(0, 1) # 均匀分布下，随机生成一个0,sum_p之间的概率
    if temp_prob<=p1:
        change_type= 1
    if temp_prob>p1 and temp_prob<=p2:
        change_type= 2
    if temp_prob>p2 and temp_prob<=p3:
        change_type= 3
    if temp_prob>p3 and temp_prob<=p4:
        change_type= 4
    if temp_prob>p4 and temp_prob<=p5:
        change_type= 5
    now_demand=td1[1, (change_type-1)] - sum(next_classified==change_type)
    if now_demand<=0:
        return self_type
    if now_demand>0:
        return change_type
def getchange(land_type):
     k=land_type-1
     U1 = 0  # urban
     W1 = 0  # water
     C1 = 0  # cropland
     T1 = 0  # tree
     O1 = 0  # orchard                
     for n in range(-1,2) :
         for m in range(-1,2) :
             if q[1][i+n,j+m] == 1:
                 U1 = U1 + 1
             if q[1][i+n,j+m] == 2:
                 W1 = W1 + 1
             if q[1][i+n,j+m] == 3:
                 C1 = C1 + 1
             if q[1][i+n,j+m] == 4:
                 T1 = T1 + 1
             if q[1][i+n,j+m] == 5:
                 O1 = O1 + 1

            #邻域密度
             U1 = U1*td3[0,0]/(N*N-1)
             W1 = W1*td3[0,1]/(N*N-1)
             C1 = C1*td3[0,2]/(N*N-1)
             T1 = T1*td3[0,3]/(N*N-1)
             O1 = O1*td3[0,4]/(N*N-1)        
             #转化成本
             U1 = U1*td2[k,0]
             W1 = W1*td2[k,1]
             C1 = C1*td2[k,2]
             T1 = T1*td2[k,3]
             O1 = O1*td2[k,4]
             #适宜性概率
             U1 = U1*prob_value[0][i][j]
             W1 = W1*prob_value[1][i][j]
             C1 = C1*prob_value[2][i][j]
             T1 = T1*prob_value[3][i][j]
             O1 = O1*prob_value[4][i][j]    
             #惯性系数            
             U1 = U1 *pre_Intertia[k]
             #赌盘选择                                  
             temp_p=np.matrix([U1,W1,C1,T1,O1])
             next_classified[i,j] = wheel(temp_p,q[1][i,j])
    
 # run 10 iterations
while M <= 10:   # run 10 iterations
    next_classified=np.array(q[1], copy=True)
    stop = 0
    for j in range(5):
        d1 = td1[1, j] - sum(q[1] == (j+1))  # t-1
        d2 = td1[1, j] - sum(q[0] == (j+1))  # t-2        
        print("需求=",td1[1, j],"t-1分配=",sum(q[1] == (j+1)),"t-2分配=",sum(q[0] == (j+1)),"d1=",d1,"d2=",d2)
        if d1 == 0:
            stop += 1
        if abs(d1) <= abs(d2):
            continue
        if d1 < d2 and d2 < 0:
            pre_Intertia[j] *= (d2 / d1)
        if 0 < d2 and d2 < d1:
            pre_Intertia[j] *= (d1 / d2)
    if stop == 5:
        break  
    # while i < 11:      
    #     while j < cols - 1:   
    while i < rows - 1:      
        print(i)
        while j < cols - 1:   
            if restric_value[i][j]==0 or q[1][i,j] >5: 
                j=j+1
                continue;
            if q[1][i,j] == 1 and restric_value[i][j]==1:
                getchange(1)                       
            # If the center pixel is water
            if q[1][i,j] == 2 and restric_value[i][j]==1:
                getchange(2)                
            # If the center pixel is crop
            if q[1][i,j] == 3 and restric_value[i][j]==1:
                getchange(3)  
            # If the center pixel is tree
            if q[1][i,j] == 4 and restric_value[i][j]==1:
                getchange(4)  
            # If the center pixel is fruit
            if q[1][i,j] == 5 and restric_value[i][j]==1:                
                getchange(5)  
            j = j + 1
        j = 1
        i = i + 1
    print (M, "iteration(s) finished")
    print("  ")
    del q[0]
    q.append(next_classified)     
    i = 1
    j = 1
    M = M + 1
#   write result to disk
driver = gdal.GetDriverByName("GTiff")
outDataset = driver.Create("E:/CA.tif",
                    cols,rows,bands,GDT_Float32)
outDataset.SetProjection(projInfo)
outDataset.SetGeoTransform(transInfo)
CA = outDataset.GetRasterBand(1)
CA.SetNoDataValue(2147483647)
result_classified=q[1]
CA.WriteArray(result_classified[:,:])
CA.FlushCache()
CA = None
outDataset = None
