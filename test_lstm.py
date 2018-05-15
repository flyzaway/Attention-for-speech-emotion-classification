#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:37:19 2017

@author: flyaway
"""

#设置随机初始种子

# =============================================================================
# from numpy.random import seed
# seed(1024)
# from tensorflow import set_random_seed
# set_random_seed(2048)
# =============================================================================

#设置随机初始种子

from get_feature import read_wav
from get_feature import read_all_feature
from get_enterface_feature import read_enterface_feature
from cross_validate import cross_validate,no_cross_validate
from numpy import mean
from keras.layers import Masking,Dropout,Dense,Activation,Embedding,Input
from keras.layers import Reshape,Flatten,advanced_activations,BatchNormalization,Bidirectional
from keras.layers.recurrent import LSTM,GRU
#from keras.utils.visualize_util import plot
from keras.utils import plot_model
from keras.models import Sequential,Model
from keras.utils import np_utils
from keras.optimizers import SGD,Adam
from keras.regularizers import l1 #activity_l1
from keras import regularizers
from keras.callbacks import EarlyStopping,History,ModelCheckpoint
import gc
from attention_LSTM import Attention_layer,Mult_Attention_layer
import numpy as np
from sklearn.metrics import confusion_matrix
import keras
from Predict_epoch import PredictEpoch
import sys

reload(sys)

sys.setdefaultencoding('utf-8')
####load data################
#feats,labels = read_all_feature()
#datasets = cross_validate(n = 5)

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
    data_shape = train_set_x.shape
    inputs = Input(shape = (data_shape[1],data_shape[2],))
    
    
    
    #mask = Masking(mask_value=0.0)(inputs)#加不加影响不大
    bilstm = Bidirectional(LSTM(100,init='glorot_uniform', inner_init='orthogonal', 
                    forget_bias_init='one', activation='tanh', 
                    inner_activation='hard_sigmoid', W_regularizer=regularizers.l2(0.0005), 
                    U_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005),
                    dropout_W=0., dropout_U=0.,return_sequences=True))(inputs)
    
    #bilstm = BatchNormalization()(bilstm)
    lstm = LSTM(200,init='glorot_uniform', inner_init='orthogonal', 
                    forget_bias_init='one', activation='tanh', 
                    inner_activation='hard_sigmoid', W_regularizer=regularizers.l2(0.0005), 
                    U_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005),
                    dropout_W=0.1, dropout_U=0.1,return_sequences=True)(bilstm)

    #print lstm
    #lstm = BatchNormalization()(lstm)
    
    attention = Attention_layer(W_regularizer = regularizers.l2(0.0005),
                                U_regularizer= regularizers.l2(0.0005),
                                b_regularizer=regularizers.l2(0.005))(lstm)   
    
    #attention = BatchNormalization()(attention)
    dense1 = Dense(output_dim = 400,
                   init  = 'glorot_uniform',
                   W_regularizer = regularizers.l2(0.0005),
                   b_regularizer=regularizers.l2(0.005),
                   activation='relu')(attention) 
    #dense1 = BatchNormalization()(dense1)
    
    drop = Dropout(0.5)(dense1)
    
    predictions = Dense(6,activation='softmax')(drop)
    
    model = Model(inputs = inputs,outputs=predictions)
 
    #######train model###############   
    #opt = SGD(lr = 0.01,decay = 1e-6 ,momentum= 0.9,nesterov = True)
    model.compile(loss = 'categorical_crossentropy',optimizer = 'Adam',metrics = ['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience = 100,mode = 'auto')
    history = History()
    #checkpointer = ModelCheckpoint(filepath="./tmp/weights{0}.hdf5".format(index), verbose=2, save_best_only=True)    
    tb_cb = keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=1,write_graph=True,write_images=False,
                                        embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    PE = PredictEpoch(validation=test_set_x)
    checkpoint = ModelCheckpoint('./model/attention_lstm_model.h5', monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
    
    plot_model(model,to_file = 'model{0}.png'.format(index),show_shapes = True, show_layer_names= False)

    model.fit(train_set_x, train_set_y, epochs = 250, batch_size = 64,#
              verbose=2,shuffle=True,
              validation_data = (test_set_x, test_set_y),
              callbacks=[early_stopping,history,tb_cb,PE,checkpoint]) #history,checkpointer
    
    best_rate.append(max(history.history['val_acc']))
    loss_and_metrics = model.evaluate(test_set_x, test_set_y , batch_size=64, verbose=2)
    #######train model###############
              
    #######evaluate model############
    print loss_and_metrics
    
    y_test_pred =np.argmax( model.predict(test_set_x, verbose=2),axis = 1)  
    print y_test_pred
    rate.append(loss_and_metrics[1])
    #model.save('./best_model{0}.h5'.format(index))
    #p
    confuse_matrix[str(index)] = confusion_matrix(test_set_y_org,y_test_pred)
    index += 1
    
    #######evaluate model############
print('acc list',rate)
print('avg acc',mean(rate))
print('best acc',best_rate,PE.best) #相同代表函数写对了
print('best avg acc',mean(best_rate))
print(confuse_matrix) #"最后一次模型的预测的混淆矩阵"
print(confusion_matrix(test_set_y_org,PE.record)) #最好识别率时混淆矩阵",
print "test_lstm.py"


# =============================================================================
# ('acc list', [0.87986])
# ('avg acc', 87986)
# ('best acc', [0.8806])
# ('best avg acc', 0.8806)
#203,3,6,8,4
#0,191,16,3,4,19
#21,2,9,193,2,1
#2,2,1,3,222,5
#0,17,4,0,20,215
# =============================================================================
# =============================================================================
# ('acc list', [0.8208333333333333])
# ('avg acc', 0.8208333333333333)
# ('best acc', [0.88194444444444442])
# ('best avg acc', 0.88194444444444442)
# {'1': array([[202,   3,   7,  25,   7,   1],
#        [  8, 157,  41,  10,   3,   4],
#        [  2,   2, 226,  14,   1,   0],
#        [  3,   4,   1, 217,   4,   0],
#        [  3,   5,   7,  10, 224,   4],
#        [  0,  39,  27,   3,  20, 156]])}
# test_bilstm.py
# =============================================================================
# =============================================================================
#加入dropout
# ('acc list', [0.85069444444444442])
# ('avg acc', 0.85069444444444442)
# ('best acc', [0.88541666666666663])
# ('best avg acc', 0.88541666666666663)
# {'1': array([[186,   1,   7,  36,   2,   1],
#        [  0, 214,  19,  11,   5,  10],
#        [  1,   8, 228,   2,   0,   1],
#        [  3,   3,   3, 234,   1,   0],
#        [  2,   8,  15,   9, 190,   7],
#        [  1,  40,  10,   1,   8, 173]])}
# test_bilstm.py
# =============================================================================
