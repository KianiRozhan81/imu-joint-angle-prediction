import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler


def fit_scalers(train_metadata, imu_cols, ik_cols, save_path):
    imu_all, ik_all = [], []

    for _, row in train_metadata.iterrows():
        imu = pd.read_csv(row["imu_path"])[imu_cols].values
        ik  = pd.read_csv(row["ik_path"])[ik_cols].values

        n = min(len(imu), len(ik))
        imu_all.append(imu[:n])
        ik_all.append(ik[:n])

    imu_scaler = StandardScaler().fit(np.concatenate(imu_all, axis=0))
    ik_scaler  = StandardScaler().fit(np.concatenate(ik_all,  axis=0))

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "imu_scaler.pkl"), "wb") as f:
        pickle.dump(imu_scaler, f)
    with open(os.path.join(save_path, "ik_scaler.pkl"), "wb") as f:
        pickle.dump(ik_scaler, f)

    return imu_scaler, ik_scaler

def load_scalers(save_path):
    with open(os.path.join(save_path, "imu_scaler.pkl"), "rb") as f:
        imu_scaler = pickle.load(f)
    with open(os.path.join(save_path, "ik_scaler.pkl"), "rb") as f:
        ik_scaler = pickle.load(f)

    return imu_scaler, ik_scaler
