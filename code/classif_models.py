# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 01:52:28 2022

@author: Eng.Mohammad Sakka
"""

def Log_reg_L2_fit(train_data,test_data):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(train_data, test_data)
    return clf

def Log_reg_fit(train_data,test_data):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0,penalty='none').fit(train_data, test_data)
    return clf
