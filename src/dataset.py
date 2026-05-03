# metadata.csv
#      ↓
# subject_split()        # train/val/test by subject ID
#      ↓
# fit_scalers()          # on train only → saves to disk
#      ↓
# IMUDataset()           # for each split
#      ↓
# DataLoader()           # batches for training

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class IMUDataset(Dataset):
    def __init__(self, metadata, window_size, stride,
                 imu_cols, ik_cols,
                 imu_scaler=None, ik_scaler=None):

        self.windows = []

        for _, row in metadata.iterrows():
            imu = pd.read_csv(row["imu_path"])[imu_cols].values.astype(np.float32)
            ik  = pd.read_csv(row["ik_path"])[ik_cols].values.astype(np.float32)

            # trim to same length in case of off-by-one from export
            n = min(len(imu), len(ik))
            imu, ik = imu[:n], ik[:n]

            # normalize
            if imu_scaler is not None:
                imu = imu_scaler.transform(imu).astype(np.float32)
            if ik_scaler is not None:
                ik = ik_scaler.transform(ik).astype(np.float32)

            # sliding windows
            for start in range(0, n - window_size + 1, stride):
                end = start + window_size
                self.windows.append((imu[start:end], ik[start:end]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        imu_win, ik_win = self.windows[idx]
        return torch.from_numpy(imu_win), torch.from_numpy(ik_win)