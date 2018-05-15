# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:48:24 2017

@author: flyaway
"""
import feature
from sklearn.utils import shuffle
from feature import read_wav
from keras.preprocessing import sequence
import numpy as np
import gc
import h5py
#anger_feat = feature.get_feature_from_python_speech_features("E:/casia_all/anger/anger001.wav")
#实现了两种特征提取的方式，使用的是read_all_feature()
def read_all_feature(window): 
    anger_feat,max_len1= read_wav("./casia/anger/",window)       
    fear_feat,max_len2= read_wav("./casia/fear/",window)
    happy_feat,max_len3 = read_wav("./casia/happy/",window)
    neutral_feat,max_len4 = read_wav("./casia/neutral/",window)
    sad_feat,max_len5 = read_wav("./casia/sad/",window)
    surprise_feat,max_len6 = read_wav("./casia/surprise/",window)
    maxlen = max([max_len1,max_len2,max_len3,max_len4,max_len5,max_len6])
    print maxlen
    anger = []
    fear = []
    happy =[]
    neutral = []
    sad = []
    surprise = []
    anger = []
    maxlen = min([250,maxlen])#太长内存不够
    for i in range(len(anger_feat)):
         anger.append((sequence.pad_sequences(anger_feat[i],maxlen = maxlen,dtype = 'float',
                                   padding = 'post',truncating = 'pre',value = 0.0)).T)#-100000
    for i in range(len(fear_feat)):
        fear.append((sequence.pad_sequences(fear_feat[i],maxlen = maxlen,dtype = 'float',
                                  padding = 'post',truncating = 'pre',value = 0.0)).T)
    for i in range(len(happy_feat)):
        happy.append((sequence.pad_sequences(happy_feat[i],maxlen = maxlen,dtype = 'float',
                                  padding = 'post',truncating = 'pre',value = 0.0)).T)
    for i in range(len(neutral_feat)):
        neutral.append((sequence.pad_sequences(neutral_feat[i],maxlen = maxlen,dtype = 'float',
                                  padding = 'post',truncating = 'pre',value = 0.0)).T)
    for i in range(len(sad_feat)):
        sad.append((sequence.pad_sequences(sad_feat[i],maxlen = maxlen,dtype = 'float',
                                  padding = 'post',truncating = 'pre',value = 0.0)).T)
    for i in range(len(surprise_feat)):
        surprise.append((sequence.pad_sequences(surprise_feat[i],maxlen = maxlen,dtype = 'float',
                                  padding = 'post',truncating = 'pre',value = 0.0)).T)
                                  
    del anger_feat,fear_feat,happy_feat,neutral_feat,sad_feat,surprise_feat
    gc.collect()
    
    anger = shuffle(np.asarray(anger),random_state = 20)
    fear = shuffle(np.asarray(fear),random_state = 20)
    happy = shuffle(np.asarray(happy),random_state = 20)
    neutral = shuffle(np.asarray(neutral),random_state = 20)
    sad =  shuffle(np.asarray(sad),random_state = 20)
    surprise = shuffle(np.asarray(surprise),random_state = 20)
    """
    data =  np.zeros((7198,maxlen,anger.shape[2]),dtype = 'float')
    data[0:1200] = fear
    data[1200:2400] = happy
    data[2400:3600] = neutral
    data[3600:4800] = sad
    data[4800:6000] = surprise
    data[6000:7198] = anger
    
    del anger,fear,happy,neutral,sad,surprise
    gc.collect()
    data_label =  np.zeros((7198,1),dtype = 'int')
    data_label[0:1200] = 0
    data_label[1200:2400] = 1
    data_label[2400:3600] = 2
    data_label[3600:4800] = 3
    data_label[4800:6000] = 4
    data_label[6000:7198] = 5
    """

    data = np.r_[fear,happy,neutral,sad,surprise,anger]
    fear_label = np.zeros((fear.shape[0],1),dtype = 'int')
    happy_label = np.ones((happy.shape[0],1),dtype = 'int')
    neutral_label = 2 * np.ones((neutral.shape[0],1),dtype = 'int')
    sad_label = 3 * np.ones((sad.shape[0],1),dtype = 'int')
    surprise_label = 4 * np.ones((surprise.shape[0],1),dtype = 'int')
    anger_label = 5 * np.ones((anger.shape[0],1),dtype = 'int')
    
    data_label = np.r_[fear_label,happy_label,neutral_label,sad_label,surprise_label,anger_label]
 

    del anger,fear,happy,neutral,sad,surprise
    gc.collect()
    
    #save as npy
    np.savez("./pkl/data.npz",data,data_label)
    
    #save as h5py
    #f = h5py.File('data.h5','w')
    #f.create_dataset('data',data = data)
    #f.create_dataset('label',data = data_label)
    #f.close()
    
    # save as pkl
    #wf = open('./pkl/data250_0_0.pkl','wb')
    #cPickle.dump([data,data_label],wf)
    #wf.close()
    #return data,data_label

def read_mult_feature(window,mult):
    anger,max_len1= read_wav("./casia/anger/",window,mult)       
    fear,max_len2= read_wav("./casia/fear/",window,mult)
    happy,max_len3 = read_wav("./casia/happy/",window,mult)
    neutral,max_len4 = read_wav("./casia/neutral/",window,mult)
    sad,max_len5 = read_wav("./casia/sad/",window,mult)
    surprise,max_len6 = read_wav("./casia/surprise/",window,mult)
    
    anger = shuffle(np.asarray(anger),random_state = 20)
    fear = shuffle(np.asarray(fear),random_state = 20)
    happy = shuffle(np.asarray(happy),random_state = 20)
    neutral = shuffle(np.asarray(neutral),random_state = 20)
    sad =  shuffle(np.asarray(sad),random_state = 20)
    surprise = shuffle(np.asarray(surprise),random_state = 20)
    
    data = np.r_[fear,happy,neutral,sad,surprise,anger]
    fear_label = np.zeros((fear.shape[0],1),dtype = 'int')
    happy_label = np.ones((happy.shape[0],1),dtype = 'int')
    neutral_label = 2 * np.ones((neutral.shape[0],1),dtype = 'int')
    sad_label = 3 * np.ones((sad.shape[0],1),dtype = 'int')
    surprise_label = 4 * np.ones((surprise.shape[0],1),dtype = 'int')
    anger_label = 5 * np.ones((anger.shape[0],1),dtype = 'int')
    
    data_label = np.r_[fear_label,happy_label,neutral_label,sad_label,surprise_label,anger_label]
 

    del anger,fear,happy,neutral,sad,surprise
    gc.collect()
    f = h5py.File('data.h5','w')
    f.create_dataset('data',data = data)
    f.create_dataset('label',data = data_label)
    f.close()
    
if __name__ == '__main__':
    read_all_feature(window=512)
    #read_mult_feature(window=512,mult=True)

