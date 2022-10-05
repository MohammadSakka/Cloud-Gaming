# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 00:18:10 2022

@author: Eng.Mohammad Sakka
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
def feature_selector(data,targets,type_,num_selected):
    if type_==0:
        bestfeatures = SelectKBest(score_func=f_regression, k=num_selected)
    else:
        bestfeatures = SelectKBest(score_func=mutual_info_classif, k=num_selected)
    bestfeatures.fit(data.astype('int'),targets.astype('int'))
    selected_feats = bestfeatures.get_feature_names_out(input_features=None)
    return selected_feats