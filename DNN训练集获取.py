
import os
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ReduceLROnPlateau
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#输入数据文件地址
inputfile = 'C:\\Users\\Lenovo\\Desktop\\wh_test_data\\test11111.xlsx'  
outputfile='C:\\Users\\Lenovo\\Desktop\\wh_test_data\\data.xlsx'
IRIS =pd.read_excel(inputfile,index='Date') 
target_var = 'T4'  # 目标变量
# 数据集的特征
features = list(IRIS.columns)
features.remove(target_var)
# 目标变量的类别
Class = [1,2,3,4,5,6,7]
# 目标变量的类别字典
Class_dict = dict(zip(Class, range(len(Class))))
# 增加一列target, 将目标变量进行编码
IRIS['target'] = IRIS[target_var].apply(lambda x: Class_dict[x])
features1 = list(IRIS.columns)
# 对目标变量进行0-1编码(One-hot Encoding)
lb = LabelBinarizer()
lb.fit(list(Class_dict.values()))
transformed_labels = lb.transform(IRIS['target'])
y_bin_labels = []  # 对多分类进行0-1编码的变量
for i in range(transformed_labels.shape[1]):
    y_bin_labels.append('Y' + str(i))
    IRIS['Y' + str(i)] = transformed_labels[:, i]
features = list(IRIS.columns)
IRIS.to_excel(outputfile) 

