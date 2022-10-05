# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 02:35:06 2022

@author: Eng.Mohammad Sakka
"""
import numpy as np
import pandas as pd

def data_balancer(train_data,train_targets):
    minor_class_recs = train_targets==1
    train_minor =  train_data[minor_class_recs]
    mul_times = 2 # how much will the class be duplicated
    records = train_minor.to_numpy()
    new_recs = []
    for i in range(int(mul_times*records.shape[0])):
        nnn = int(np.random.uniform(low=0,high=records.shape[0]))
        tmp_rec = records[nnn,:] 
        new_recs.append(tmp_rec)
    new_recs = np.array(new_recs)
    old_recs = records
    new_train_minor_data = np.concatenate((old_recs,new_recs),axis=0)
    major_class_recs = train_targets==0
    train_major =  train_data[major_class_recs]
    train_major_np = train_major.to_numpy()
    new_train_data_np = np.concatenate((train_major_np,new_train_minor_data),axis=0)
    new_train_data = pd.DataFrame(new_train_data_np, columns=train_data.columns)
    new_targets_np = np.concatenate((np.zeros((train_major_np.shape[0],1)),np.ones((new_train_minor_data.shape[0],1))),axis = 0)
    new_train_data["targets"] = (new_targets_np. astype(int))
    from sklearn.utils import shuffle
    new_data = shuffle(new_train_data)
    new_train_targets = new_data["targets"]
    new_data.drop('targets', axis=1, inplace=True)
    return new_data,new_train_targets