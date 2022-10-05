# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 02:09:11 2022

@author: Eng.Mohammad Sakka
"""
def remove_outliers(train_data,train_targets):
    from scipy import stats
    import numpy as np

    # Calculate the z-scores
    z_scores = stats.zscore(train_data)
    abs_z_scores = np.abs(z_scores)
    # Select data points with a z-scores above or below 3
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    train_wo_outliers = train_data[filtered_entries]
    train_targets = train_targets[filtered_entries]
    return train_wo_outliers,train_targets