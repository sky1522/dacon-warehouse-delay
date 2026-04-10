"""
Phase NN: [TITLE]
- [핵심 변경 요약]
"""

import pandas as pd
import numpy as np
import gc
import pickle
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

DRIVE_CKPT_DIR = '/content/drive/MyDrive/dacon_ckpt'
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

PIPELINE_VERSION = "phaseNN_v1"


def save_ckpt(local_path, data, feature_cols=None):
    if feature_cols is not None:
        data['feature_cols'] = list(feature_cols)
        data['n_features'] = len(feature_cols)
    data['pipeline_version'] = PIPELINE_VERSION
    with open(local_path, 'wb') as f:
        pickle.dump(data, f)
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(local_path))
    if os.path.exists(os.path.dirname(drive_path)):
        shutil.copy(local_path, drive_path)
    print(f"  Saved: {local_path}", flush=True)


def load_ckpt(local_path, expected_features=None):
    drive_path = os.path.join(DRIVE_CKPT_DIR, os.path.basename(local_path))
    for path in [drive_path, local_path]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
            if ckpt.get('pipeline_version') != PIPELINE_VERSION:
                print(f"  {os.path.basename(path)}: version mismatch", flush=True)
                return None
            if expected_features is not None:
                cached_fc = ckpt.get('feature_cols')
                if cached_fc is not None and cached_fc != list(expected_features):
                    print(f"  {os.path.basename(path)}: feature_cols mismatch", flush=True)
                    return None
            return ckpt
    return None


# ##############################################################
# Part 0: Feature Engineering (copy from previous phase)
# ##############################################################
# TODO: Copy feature engineering from run_phase{PREV}_fe.py


# ##############################################################
# Part 1: NEW Features
# ##############################################################
# TODO: Add new features here


# ##############################################################
# Part 2: Training (8 models)
# ##############################################################
# TODO: Copy training loop from previous phase


# ##############################################################
# Part 3: Ensemble + Submission
# ##############################################################
# TODO: Nelder-Mead ensemble + submission generation


print("\n=== Phase NN Complete ===", flush=True)
