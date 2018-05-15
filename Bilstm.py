#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:37:19 2017

@author: flyaway
"""
import sys

reload(sys)

sys.setdefaultencoding('utf-8')

from numpy.random import seed
seed(1024)
from tensorflow import set_random_seed
set_random_seed(2048)

from get_feature import read_wav
from get_feature import read_all_feature
from cross_validate import cross_validate,no_cross_validate
from numpy import mean
from keras.layers import Masking,Dropout,Dense,Activation,Embedding,Reshape,Flatten,advanced_activations,BatchNormalization,Bidirectional
from keras.layers.recurrent import LSTM
#from keras.utils.visualize_util import plot
from keras.utils import plot_model
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import SGD,Adam
from keras.regularizers import l1 #activity_l1
from keras import regularizers
from keras.callbacks import EarlyStopping,History,ModelCheckpoint
import gc
import keras
from sklearn.metrics import confusion_matrix
from Predict_epoch import PredictEpoch
####load data################
#feats,labels = read_all_feature()
#datasets = cross_validate(n = 5)
#datasets = no_cross_validate()
datasets = no_cross_validate(data_base = "casia",window=512,data_path = "./pkl/data.npz")
train_set_x = []
train_set_y = []
test_set_x = []
test_set_y = []
rate = []
best_rate = []
index = 1
confuse_matrix = {}
for item in datasets:
    train_set_x, train_set_y,test_set_x, test_set_y = item
    test_set_y_org = test_set_y
    del item 
    gc.collect()
    print train_set_x.shape
    train_set_y = np_utils.to_categorical(train_set_y, 6)
    test_set_y = np_utils.to_categorical(test_set_y, 6) 
######load data###############
    ###crate model 线性叠加模式 Sequential######    
    model = Sequential() 
    shape = train_set_x.shape
    model.add(Masking(mask_value = 0.0,input_shape = (shape[1],shape[2])))
    print model.output_shape
    model.add(Bidirectional(LSTM(100, init='glorot_uniform', inner_init='orthogonal', 
                   forget_bias_init='one', activation='tanh', 
                   inner_activation='hard_sigmoid', W_regularizer=regularizers.l2(0.0005), 
                   U_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005),
                   dropout_W=0., dropout_U=0.,return_sequences=True))) # returns a sequence of vectors of dimension 32
    print model.output_shape

    model.add((LSTM(200, init='glorot_uniform', inner_init='orthogonal', 
                   forget_bias_init='one', activation='tanh', 
                   inner_activation='hard_sigmoid', W_regularizer=regularizers.l2(0.0005), 
                   U_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005),
                   dropout_W=0., dropout_U=0.,return_sequences=False)))# returns a sequence of vectors of dimension 32
    print model.output_shape
    #如果return_sequences=True，那么输出3维 tensor(nb_samples, timesteps, output_dim) .否则输出2维tensor(nb_samples,output_dim)。
    #return_sequence: Boolean.False返回在输出序列中的最后一个输出；True返回整个序列。
    #就是说如果要叠加使用LSTM则必须设置return_sequences=True
    #froget_bias_init,遗忘偏置初始化
    model.add(Dense(output_dim = 400,init  = 'glorot_uniform',
            W_regularizer = regularizers.l2(0.0005),b_regularizer=regularizers.l2(0.005)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(output_dim = 560,init  = 'glorot_uniform',
    #        W_regularizer = regularizers.l1(0.000),b_regularizer=regularizers.l1(0.00)))
    #model.add(Activation('relu'))
    
    #model.add(advanced_activations.LeakyReLU(alpha=0.3))
    print model.output_shape

    
    model.add(Dense(6))
    model.add(Activation('softmax'))
    ###crate model 线性叠加模式 Sequential######
    
    #######train model###############   
    opt = SGD(lr = 0.01,decay = 1e-6 ,momentum= 0.9,nesterov = True)
    model.compile(loss = 'categorical_crossentropy',optimizer = 'Adam',metrics = ['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience = 100,mode = 'auto')
    history = History()
    #checkpointer = ModelCheckpoint(filepath="./tmp/weights{0}.hdf5".format(index), verbose=2, save_best_only=True)    
    tb_cb = keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=1,write_graph=True,write_images=False,
                                        embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    PE = PredictEpoch(validation=test_set_x)
    checkpoint = ModelCheckpoint('./model/Bilstm_model.h5', monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
        
    model.fit(train_set_x, train_set_y,nb_epoch = 250,batch_size = 64,#
              verbose=2,shuffle=True,
              validation_data = (test_set_x, test_set_y),
              callbacks=[early_stopping,history,tb_cb,PE,checkpoint]) #[early_stopping,history,checkpointer]
    
    best_rate.append(max(history.history['val_acc']))
    #######train model###############
              
    #######evaluate model############
    loss_and_metrics = model.evaluate(test_set_x, test_set_y , batch_size=64, verbose=2)
    print loss_and_metrics
    
    y_test_pred = model.predict_classes(test_set_x, verbose=2)  
    print y_test_pred
    rate.append(loss_and_metrics[1])
    #model.save('./Bilstm_model{0}.h5'.format(index))
    #plot_model(model,to_file = 'model{0}.png'.format(index),show_shapes = True, show_layer_names= False)
    confuse_matrix[str(index)] = confusion_matrix(test_set_y_org,y_test_pred)
    index += 1
    #######evaluate model############
print('acc list',rate)
print('avg acc',mean(rate))
print('best acc',best_rate,PE.best) #相同代表函数写对了
print('best avg acc',mean(best_rate))
print(confuse_matrix) #"最后一次模型的预测的混淆矩阵"
print(confusion_matrix(test_set_y_org,PE.record)) #最好识别率时混淆矩阵",
print "bilstm.py"
del datasets
gc.collect()
# =============================================================================
# ('acc list', [0.82291666666666663])
# ('avg acc', 0.82291666666666663)
# ('best acc', [0.84583333333333333])
# ('best avg acc', 0.84583333333333333)
# =============================================================================
