# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 01:03:03 2022

@author: Eng.Mohammad Sakka
"""
# Linear regression
def fit_LR(train_data,train_targets):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(train_data, train_targets)
    return regressor

# polynomial linear regression with Ridge regularization
def fit_PLR(train_data,train_targets,val_data,val_targets):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures #to convert the original features into their higher order terms 
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    import numpy as np
    alphas = [0,0.1,1]
    losses = []
    for i in range(len(alphas)):

        polynomial_features = PolynomialFeatures(degree=2)
        linear_regression = Ridge(alpha =i )
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
        pipeline.fit(train_data, train_targets)

    # Evaluate the models using crossvalidation
        mse = mean_squared_error(pipeline.predict(val_data),val_targets)
        losses.append(mse)
    ind = np.argmin(losses)
    print(ind)
    best_alpha = alphas[ind]
    linear_regression = Ridge(alpha =best_alpha )
    pipeline.fit(train_data, train_targets)
    return pipeline

# Neural Network
def fit_NN(train_data,train_targets):
    from sklearn.neural_network import MLPRegressor
    regr = MLPRegressor( max_iter=100,hidden_layer_sizes=(20,),alpha = 0.1)
    regr.fit(train_data, train_targets)   
    return regr

 