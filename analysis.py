#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:47:21 2018

@author: flyaway
"""

import sys

reload(sys)

sys.setdefaultencoding('utf-8')
from keras.models import Model  
from keras.models import load_model
from attention_LSTM import Attention_layer
#from feature import get_feature_from_librosa
#from keras.preprocessing import sequence
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cross_validate import cross_validate,no_cross_validate
from keras.utils import np_utils
import gc
from sklearn import manifold
#import time

def plot_embedding_2d(X, Y,pic_name,title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],str(Y[i]),
                 color=plt.cm.Set1((Y[i])),
                
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    plt.savefig("./"+ pic_name +".tNSE-2d.png")
#将降维后的数据可视化,3维
def plot_embedding_3d(X,Y,pic_name, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i,2],str(Y[i]),
                 color=plt.cm.Set1((Y[i]) ),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    plt.savefig("./"+ pic_name +".tNSE-3d.png")
   #plt.savefig("./attention.tNSE-3d.jpg")



model_1 = load_model('./model/attention_lstm_model1.h5',custom_objects = {'Attention_layer':Attention_layer})
attention_layer_output = Model(inputs=model_1.input,  
                                     outputs=model_1.get_layer('Attention_layer',3).output)  
model_2 = load_model('./model/Bilstm_model.h5')
lstm_layer_output =  Model(inputs=model_2.input,  
                                     outputs=model_2.get_layer('LSTM',3).output)  
"""
#以这个model的预测值作为输出  
feat = get_feature_from_librosa("./casia_all/fear/Chang (1).wav",512)
feat = (sequence.pad_sequences(feat,maxlen = 250,dtype = 'float',padding = 'post',truncating = 'pre',value = 0.0)).T
feat = feat.reshape((1,250,62))
attention_output = attention_layer_output.predict(feat)  
plt.plot(attention_output.T)
"""
datasets = no_cross_validate(data_base = "casia",window=512,data_path = "./pkl/data.npz")

for item in datasets:
    train_set_x, train_set_y,test_set_x, test_set_y = item
    test_set_y_org = test_set_y
    del item 
    gc.collect()
    print train_set_x.shape
    #train_set_y = np_utils.to_categorical(train_set_y, 6)
    #test_set_y = np_utils.to_categorical(test_set_y, 6) 
    data_shape = train_set_x.shape
    attention_output = attention_layer_output.predict(train_set_x)
    tsne = manifold.TSNE(n_components = 3)
    X_tsne_1 = tsne.fit_transform(attention_output)
    plot_embedding_2d(X_tsne_1[:,0:2],train_set_y,'attention',"t-SNE 2D")
    plot_embedding_3d(X_tsne_1,train_set_y,'attention',"t-SNE 3D ")
    
    lstm_output = lstm_layer_output.predict(train_set_x)
    tsne = manifold.TSNE(n_components = 3)
    X_tsne_2 = tsne.fit_transform(lstm_output)

