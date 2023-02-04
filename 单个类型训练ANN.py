
import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt 
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ReduceLROnPlateau


#####参数设置
epoch = 5000	#迭代次数
inputnum = 8		#输入节点数s
midnum = 12		#隐层节点数
outputnum = 4		#输出节点数
learnrate =0.02	#学习率
Terror = 1e-7		#迭代中止条件
datannum = 8482	#样本总数
trainX = 0.1		#训练数据比例

#输入数据文件地址
inputfile = 'C:\\Users\\Lenovo\\Desktop\\type2.xlsx'   		 	
#输出预测的文件地址
outputfile = 'C:\\Users\\Lenovo\\Desktop\\output.xlsx' 	
#模型保存地址
modelfile = 'C:\\Users\\Lenovo\\Desktop\\type1_modelweight.model' 	


#因子所在列
factor = ['F1','F2','F3','F4','F5','F6','F7','F8']														
#目标所在列
label = ['L1','L2','L3','L4','L5'] 	


#初始处理


data = pd.read_excel(inputfile,index='Date') 
data = data.reindex(np.random.permutation(data.index))
trainnum =int(len(data)*trainX)
print(trainnum)
data_train = data[0:trainnum].copy() 
data_test = data[trainnum:datannum].copy()
data_std = data.max() - data.min()
data_train = (data_train - data.min()) / data_std
data_test = (data_test - data.min()) / data_std
x_train = data_train[factor].values
y_train = data_train[label].values


#构建模型
model = Sequential()
#初始化权值阈值
model.add(Dense(units=12,
                input_dim=8,
                kernel_initializer='normal'))
model.add(Dense(units=5,
                kernel_initializer='normal',
                activation='sigmoid'))
sgd=SGD(lr=0.2)
model.compile(loss='mse',
              optimizer='adam',metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.001)
train_history=model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.3,
                        callbacks=[reduce_lr],
                        epochs=1000,
                        batch_size=10)

model.save_weights(modelfile)
# 预测数据
x_test = data_test[factor]
y_test = data_test[label]

# # #写入训练结果
# proba=model.predict_proba(x_test)
# data_test[u'L1_pred'] = proba[:, 0]
# data_test[u'L2_pred'] = proba[:, 1]
# data_test[u'L3_pred'] = proba[:, 2]
# data_test[u'L4_pred'] = proba[:, 3]
# data_test[u'L5_pred'] = proba[:, 4]
# data_test.to_excel(outputfile) 
# 评估模型
loss,accuracy=model.evaluate(x_test,y_test)

print('loss:',loss)
print('accuracy:',accuracy)