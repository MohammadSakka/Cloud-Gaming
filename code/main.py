# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:42:22 2022

@author: Eng. Mohammad Sakka

Main File
"""
#%% reading the data
# select the data number
# 0- bitrate
# 1 - stream quality
SDn = 1
num_selected = 3
#
DSs = ['bitrate_prediction','stream_quality_data']
targets_names = ['target','stream_quality']
target_name = targets_names[SDn]

# loading training and testing
import os
init_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\data\\"
import pandas as pd
train_data = pd.read_csv(init_dir + DSs[SDn] + '\\train_data.csv', low_memory=False)
test_data = pd.read_csv(init_dir + DSs[SDn] + '\\test_data.csv', low_memory=False)

# split targets and features

train_targets = train_data[target_name]
test_targets = test_data[target_name]
train_data.drop(target_name, axis=1, inplace=True)
test_data.drop(target_name, axis=1, inplace=True)
# generating validation data
from sklearn.model_selection import train_test_split
test_data, val_data, test_targets, val_targets = train_test_split(test_data, test_targets,test_size=0.5)
#%% preprocessing
import preprocessing
train_data = preprocessing.preprocesser(train_data,SDn)
test_data = preprocessing.preprocesser(test_data,SDn)
val_data = preprocessing.preprocesser(val_data,SDn)
#%% Feature Selection
import feature_selection
tmp = train_data.copy()
tmp["target"] = train_targets
sample = tmp.sample(n = int(0.001*len(tmp)))
sample_targets = sample["target"]
sample.drop("target", axis=1, inplace=True)
best_feats = feature_selection.feature_selector(sample,sample_targets,SDn,num_selected)
train_data = train_data[best_feats]
test_data = test_data[best_feats]
val_data = val_data[best_feats]
#%% data visualization
import data_visualization
data_visualization.visualizer(train_data)
#%% models fitting
if SDn==0:
    import reg_models
    # linear regression
    model = reg_models.fit_LR(train_data,train_targets)
    LR_pred_test_targets = model.predict(test_data)
    # polynomial regression
    model = reg_models.fit_PLR(train_data,train_targets,val_data,val_targets)
    PLR_pred_test_targets = model.predict(test_data)
    # NN
    model = reg_models.fit_NN(train_data,train_targets)
    NN_pred_test_targets = model.predict(test_data)
    # printing
    from sklearn import metrics
    import numpy as np
    
    print('Mean Absolute Error For LR:', metrics.mean_absolute_error(test_targets, LR_pred_test_targets))
    print('Mean Absolute Error For PLR:', metrics.mean_absolute_error(test_targets, PLR_pred_test_targets))
    print('Mean Absolute Error For NN:', metrics.mean_absolute_error(test_targets, NN_pred_test_targets))

    print('Mean Squared Error For LR:', metrics.mean_squared_error(test_targets, LR_pred_test_targets))
    print('Mean Squared Error For PLR:', metrics.mean_squared_error(test_targets, PLR_pred_test_targets))
    print('Mean Squared Error For NN:', metrics.mean_squared_error(test_targets, NN_pred_test_targets))

    print('Root Mean Squared Error  For LR:', np.sqrt(metrics.mean_squared_error(test_targets, LR_pred_test_targets)))
    print('Root Mean Squared Error For PLR:', np.sqrt(metrics.mean_squared_error(test_targets, PLR_pred_test_targets)))
    print('Root Mean Squared Error For NN:', np.sqrt(metrics.mean_squared_error(test_targets, NN_pred_test_targets)))
    
    print('R2  For LR:', metrics.r2_score(test_targets, LR_pred_test_targets))
    print('R2  For PLR:', metrics.r2_score(test_targets, PLR_pred_test_targets))
    print('R2  For NN:', metrics.r2_score(test_targets, NN_pred_test_targets))

else:
    from sklearn import metrics
    # Outlier detection
    import outlier_detection
    train_data,train_targets = outlier_detection.remove_outliers(train_data,train_targets)
    # data balancing 
    import data_balancing
    train_data_balanced,train_targets_balanced = data_balancing.data_balancer(train_data,train_targets)
    # import classif_models
    import classif_models
    model = classif_models.Log_reg_fit(train_data,train_targets)
    Log_reg_pred_test_targets = model.predict(test_data)
    model = classif_models.Log_reg_fit(train_data_balanced,train_targets_balanced)
    Log_reg_pred_test_targets_balanc = model.predict(test_data)
    model = classif_models.Log_reg_L2_fit(train_data,train_targets)
    Log_reg_L2_pred_test_targets = model.predict(test_data)
    model = classif_models.Log_reg_L2_fit(train_data_balanced,train_targets_balanced)
    Log_reg_L2_pred_test_targets_balanc = model.predict(test_data)
    # classif_models.run_classif_models()
    precision_Log = metrics.precision_score(test_targets,Log_reg_pred_test_targets)
    print('Acc for Logistic regression: ',metrics.accuracy_score(test_targets,Log_reg_pred_test_targets))
    print('Precision for Logistic regression: ',precision_Log)
    recall_Log = metrics.recall_score(test_targets,Log_reg_pred_test_targets)
    print('Recall for Logistic regression: ',recall_Log)
    print('F1 for Logistic regression: ',2 * (precision_Log * recall_Log) / (precision_Log + recall_Log))
    
    precision_Log_L2 = metrics.precision_score(test_targets,Log_reg_L2_pred_test_targets)
    print('Acc for Logistic regression with L2: ',metrics.accuracy_score(test_targets,Log_reg_L2_pred_test_targets))
    print('Precision for Logistic regression with L2: ',precision_Log_L2)
    recall_Log_L2 = metrics.recall_score(test_targets,Log_reg_L2_pred_test_targets)
    print('Recall for Logistic regression with L2: ',recall_Log_L2)
    print('F1 for Logistic regression with L2: ',2 * (precision_Log_L2 * recall_Log_L2) / (precision_Log_L2 + recall_Log_L2))
    
    precision_Log_balanc = metrics.precision_score(test_targets,Log_reg_pred_test_targets_balanc)
    print('Acc for Logistic regression Balanced Data: ',metrics.accuracy_score(test_targets,Log_reg_pred_test_targets_balanc))
    print('Precision for Logistic regression Balanced Data: ',precision_Log_balanc)
    recall_Log_balanc = metrics.recall_score(test_targets,Log_reg_pred_test_targets_balanc)
    print('Recall for Logistic regression Balanced Data: ',recall_Log_balanc)
    print('F1 for Logistic regression Balanced Data: ',2 * (precision_Log_balanc * recall_Log_balanc) / (precision_Log_balanc + recall_Log_balanc))
    
    precision_Log_L2_balanc = metrics.precision_score(test_targets,Log_reg_L2_pred_test_targets_balanc)
    print('Acc for Logistic regression with L2 Balanced Data: ',metrics.accuracy_score(test_targets,Log_reg_L2_pred_test_targets_balanc))
    print('Precision for Logistic regression with L2 Balanced Data: ',precision_Log_L2_balanc)
    recall_Log_L2_balanc = metrics.recall_score(test_targets,Log_reg_L2_pred_test_targets_balanc)
    print('Recall for Logistic regression with L2 Balanced Data: ',recall_Log_L2_balanc)
    print('F1 for Logistic regression with L2 Balanced Data: ',2 * (precision_Log_L2_balanc * recall_Log_L2_balanc) / (precision_Log_L2_balanc + recall_Log_L2_balanc))
