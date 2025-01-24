#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

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

svc_params = {
    'C': [0.1, 1, 5, 10],
    'tol': [0.0001, 0.001, 0.01],
    'gamma': ['scale', 'auto']
}

knn_params = {
    'n_neighbors': [5, 10, 15, 20, 30, 50],
    'weights': ['uniform', 'distance']
}

logistic_params = {
    'penalty': ['l2'],  
    'C': [0.1, 1, 5, 10],
    'solver': ['lbfgs', 'liblinear'], 
    'max_iter': [100, 200, 500]
}

# Create model instances using the defined parameters
rf_variants = [
    RandomForestClassifier(
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
    XGBClassifier(
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
    MLPClassifier(
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

svc_variants = [
    SVC(
        kernel='rbf', 
        C=c, 
        tol=t, 
        gamma=g,
        probability=True
    ) 
    for c, t, g in zip(svc_params['C'], 
                       svc_params['tol'], 
                       svc_params['gamma'])
]

knn_variants = [
    KNeighborsClassifier(
        n_neighbors=n, 
        weights=w
    ) 
    for n, w in zip(knn_params['n_neighbors'], 
                    knn_params['weights'])
]

logistic_variants = [
    LogisticRegression(
        penalty=p, 
        C=c, 
        solver=s, 
        max_iter=mi, 
        random_state=0
    ) 
    for p, c, s, mi in zip(logistic_params['penalty'], 
                           logistic_params['C'], 
                           logistic_params['solver'], 
                           logistic_params['max_iter'])
]

# Organize the models into a dictionary
models = {
    "Random Forest Variants": rf_variants,
    "XGBoost Variants": xgb_variants,
    "MLP Neural Network Variants": mlp_variants,
    "Support Vector Machine Variants": svc_variants,
    "K-Nearest Neighbors Variants": knn_variants,
    "Logistic Regression Variants": logistic_variants
}

# Save the dictionary to a file using pickle
with open('classification_variants.pkl', 'wb') as f:
    pickle.dump(models, f)