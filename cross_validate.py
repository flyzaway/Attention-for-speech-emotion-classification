# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:19:50 2017

@author: flyaway
"""
from sklearn.model_selection import StratifiedKFold
from get_feature import read_all_feature
from get_enterface_feature import read_enterface_feature
from sklearn.cross_validation import train_test_split
import numpy as np
import os

import gc
import h5py
def load_data(data_base,window,data_path):
    """
    if not os.path.exists("./pkl/data250_0_0.pkl"): 
        if data_base == "enterface":
            read_enterface_feature(window)
        else:
            read_all_feature(window)
    rf = open('./pkl/data250_0_0.pkl','rb')
    data,data_label = cPickle.load(rf)
    rf.close()
    """
    if data_path[-3:-1] == "h5":
        if not os.path.exists("./pkl/data.h5"):
            if data_base == "enterface":
                read_enterface_feature(window)
            else:
                read_all_feature(window)
        f = h5py.File('./pkl/data.h5','r')
        data = f['data']
        data_label = f['data_label']
    
    else:
        if not os.path.exists(data_path): 
            if data_base == "enterface":
                read_enterface_feature(window)
            else:
                read_all_feature(window)
        r = np.load("./pkl/data.npz")
        data = r["arr_0"] 
        data_label = r["arr_1"]
        del r
        gc.collect()
    
    return data,data_label
def cross_validate(n = 5):
    data,data_label = load_data()
    print data.shape
    data_label = data_label.reshape(data_label.shape[0])
    print data_label.shape
    skf = StratifiedKFold(n_splits = n)
    skf.get_n_splits(data,data_label)
    
    rval = []
    for train_index, test_index in skf.split(data, data_label):    
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = data_label[train_index], data_label[test_index]
        rval.append((X_train,  y_train, X_test, y_test))
        print np.sum(y_train == 0),np.sum(y_train == 1),np.sum(y_train == 2),np.sum(y_train == 3), np.sum(y_train == 4), np.sum(y_train == 5)
        print np.sum(y_test == 0), np.sum(y_test == 1), np.sum(y_test == 2),np.sum(y_test == 3), np.sum(y_test == 4),np.sum(y_test == 5)
    return rval
def no_cross_validate(data_base,window,data_path):
    data,data_label = load_data(data_base,window,data_path)
    print data.shape
    data_label = data_label.reshape(data_label.shape[0])
    print data_label.shape
    X_train, X_test, y_train, y_test = train_test_split(data, data_label, test_size=0.2,random_state=1024)
    rval = []
    rval.append((X_train,y_train,X_test,y_test))
    return rval

if __name__ == '__main__':
    rval = no_cross_validate('casia',512,"./pkl/data.h5")