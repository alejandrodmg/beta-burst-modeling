#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import h5py
import numpy as np
import pickle
import logging
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Helper functions
def group_strings_by_prefix(strings):
    """Group strings by their prefix."""
    groups = {}
    for string in strings:
        prefix = get_rat_id(string)  # Extract the prefix
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(string)
    return list(groups.values())

def obj_to_string(array):
    """Convert char object to string."""
    return u''.join([chr(o[0]) for o in array])

def get_rat_id(s):
    """Extract the first sequence of digits before the first underscore."""
    match = re.match(r'(\d+)_', s)
    return match.group(1) if match else None

def load_data(file):
    """Load data from the given HDF5 file."""
    with h5py.File(file, 'r') as f:
        data = f['beta_RNN']
        rat_id = obj_to_string(f[data[0][0]][:])  # Column 1 = Rat ID
        eeg = f[data[2][0]][:]  # Column 3 = EEG
        eeg_time = f[data[3][0]][:].reshape(-1)  # Column 4 = EEG time
        velocity = f[data[4][0]][:].reshape(-1)  # Column 5 = Treadmill Velocity
        _ = f[data[5][0]][:].reshape(-1)  # Column 6 = Velocity Time (unused)
    return rat_id, eeg, eeg_time, velocity

def create_trial_folds(X, y, seed, n_trials):
    """Create trial-based data folds for cross-validation. The folds are 
    created at trial level to prevent leakage into the test set."""
    samples_per_trial = len(X) // n_trials
    trial_indices = np.arange(n_trials)
    trial_samples = np.repeat(trial_indices, samples_per_trial)
    rng = np.random.default_rng(seed)
    rng.shuffle(trial_indices)
    trial_X_chunks = [np.array(X)[trial_samples == trial] for trial in np.unique(trial_samples)]
    trial_y_chunks = [np.array(y)[trial_samples == trial] for trial in np.unique(trial_samples)]
    reordered_X = np.concatenate([trial_X_chunks[np.where(np.unique(trial_samples) == trial)[0][0]] for trial in trial_indices])
    reordered_y = np.concatenate([trial_y_chunks[np.where(np.unique(trial_samples) == trial)[0][0]] for trial in trial_indices])
    return reordered_X, reordered_y

class BaselineModel:
    """A baseline model that predicts the mean of the training data."""
    def __init__(self, name="BaselineModel"):
        self.name = name  
    def fit(self, X_train, y_train):
        self.mean_value = np.mean(y_train)
    def predict(self, X_test):
        return np.repeat(self.mean_value, len(X_test))

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    # Import hyperparameter variants for models
    with open('regression_variants.pkl', 'rb') as f:
        models = pickle.load(f)

    # Main loop for training and evaluation
    sessions = os.listdir('data')
    rat_groups = group_strings_by_prefix(sessions)
    seeds = [1, 2, 3, 4, 5]  # Seeds for cross-validation
    reg_models_scores = {}

    for rat_session in tqdm(rat_groups):
        rat_id = get_rat_id(rat_session[0])
        reg_models_scores[rat_id] = {}

        seqs = []
        time_range = 80 # 80 times steps equals 80 * 5ms = 400ms
        for sess in rat_session:
            _, beta, beta_time, velocity_events = load_data(f'data/{sess}')
            for v in velocity_events:
                idx_peak = np.argmin(beta_time <= v)
                seqs.append(beta.ravel()[idx_peak - time_range:idx_peak + time_range])

        total_seqs = len(seqs)
        logging.info(f"Rat ID: {rat_id}, Total trials: {total_seqs}")

        seq_size = 40 # 40 times steps equals 40 * 5ms = 200ms
        time_jump = 2 # 2 times steps equals 2 * 5ms = 10ms
        X, y = [], []
        for s in seqs:
            for i in range(seq_size, time_range, time_jump):
                X.append(s[i - seq_size:i])
                # Distance, in timesteps, to event
                y.append(time_range - i)

        # Train and evaluate each model
        for model_group, model_list in models.items():
            reg_models_scores[rat_id][model_group] = []
            for model in model_list:
                # Create a unique identifier for the model
                model_id = model.name if hasattr(model, "name") else type(model).__name__
                if hasattr(model, "get_params"):
                    params = model.get_params()
                    model_id += "_" + "_".join([f"{k}={v}" for k, v in sorted(params.items()) if isinstance(v, (int, float, str))])
                reg_models_scores[rat_id][model_group][model_id] = []
                for seed in seeds:
                    fX, fy = create_trial_folds(X, y, seed, total_seqs)
                    X_train, X_test, y_train, y_test = train_test_split(fX, fy, test_size=0.20, shuffle=False, random_state=seed)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2score = r2_score(y_test, y_pred)
                    reg_models_scores[rat_id][model_group][model_id].append(r2score)
