# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 00:29:59 2022

@author: Eng.Mohammad Sakka
"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def projector(data):
    dim_reducer = PCA(n_components=2)
    data_reduced = dim_reducer.fit_transform(data)
    return data_reduced
    
def visualizer(data):
    data_reduced = projector(data)
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1], marker='.')
    plt.xlabel('Prinicipal Comp1')
    plt.ylabel('Prinicipal Comp2')
    plt.show()
    
