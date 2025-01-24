#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

class BaselineModel:
    """A baseline model that predicts the mean of the training data."""
    def __init__(self, name="BaselineModel"):
        self.name = name  
    def fit(self, X_train, y_train):
        self.mean_value = np.mean(y_train)
    def predict(self, X_test):
        return np.repeat(self.mean_value, len(X_test))

rf_params = {
    'n_estimators': [100, 200, 500, 1000],  
    'min_samples_split': [2, 5, 10],        
    'max_depth': [None, 10, 20, 30]         
}

xgb_params = {
    'max_depth': [3, 5, 7, 9],               
    'n_estimators': [100, 200, 500, 1000],       
    'min_child_weight': [1, 2, 5, 10],       
    'learning_rate': [0.01, 0.005, 0.1]      
}

mlp_params = {
    'hidden_layer_sizes': [(64, 32), 
                           (128, 64, 32), 
                           (256, 128, 64), 
                           (128, 64), 
                           (256, 64)]
}


svr_params = {
    'C': [0.1, 1, 5, 10],              
    'tol': [0.0001, 0.001, 0.01],      
    'gamma': ['scale', 'auto']     
}

knn_params = {
    'n_neighbors': [5, 10, 15, 20, 30, 50],       
    'weights': ['uniform', 'distance']      
}

# Create model instances using the defined parameters
rf_variants = [
    RandomForestRegressor(
        n_estimators=n, 
        min_samples_split=s, 
        max_depth=d, 
        n_jobs=-1, 
        random_state=0
    ) 
    for n, s, d in zip(rf_params['n_estimators'], 
                      rf_params['min_samples_split'], 
                      rf_params['max_depth'])
]

xgb_variants = [
    XGBRegressor(
        max_depth=d, 
        n_estimators=n, 
        min_child_weight=mcw, 
        learning_rate=lr, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=0, 
        n_jobs=-1
    ) 
    for d, n, mcw, lr in zip(xgb_params['max_depth'], 
                           xgb_params['n_estimators'], 
                           xgb_params['min_child_weight'], 
                           xgb_params['learning_rate'])
]

mlp_variants = [
    MLPRegressor(
        hidden_layer_sizes=hls, 
        activation='relu', 
        solver='adam', 
        learning_rate='adaptive', 
        random_state=0, 
        max_iter=2000, 
        early_stopping=True
    ) 
    for hls in mlp_params['hidden_layer_sizes']
]

svr_variants = [
    SVR(
        kernel='rbf', 
        C=c, 
        tol=t, 
        gamma=g
    ) 
    for c, t, g in zip(svr_params['C'], 
                      svr_params['tol'], 
                      svr_params['gamma'])
]

knn_variants = [
    KNeighborsRegressor(
        n_neighbors=n, 
        weights=w
    ) 
    for n, w in zip(knn_params['n_neighbors'], 
                      knn_params['weights'])
]

models = {
    "Random Forest Variants": rf_variants,
    "XGBoost Variants": xgb_variants,
    "MLP Neural Network Variants": mlp_variants,
    "Support Vector Machine Variants": svr_variants,
    "K-Nearest Neighbors Variants": knn_variants,
    "Linear Regression": [LinearRegression()],
    "Baseline": [BaselineModel()]
}

# Save the dictionary to a file using pickle
with open('regression_variants.pkl', 'wb') as f:
    pickle.dump(models, f)