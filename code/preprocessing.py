# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 23:57:32 2022

@author: Eng.Mohammad Sakka
"""

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scaler_robust(data):
    scaler = RobustScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data


def scaler_standard(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data
    
    
def preprocesser(data,type_):
    if type_==0:
        preprocessed_data = scaler_robust(data)
    else:
        cleanup_nums = {"auto_bitrate_state":     {"off": 0, "full": 1,"partial":0.5},
                "auto_fec_state": {"off": 0, "partial": 0.5}}
        preprocessed_data = data.replace(cleanup_nums)
        preprocessed_data = scaler_standard(preprocessed_data)
    preprocessed_data = pd.DataFrame(preprocessed_data, index=data.index, columns=data.columns)
    return preprocessed_data