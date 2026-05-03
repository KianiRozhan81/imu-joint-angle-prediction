import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

subject_id = ['AB06', 'AB07', 'AB08', 
              'AB09', 'AB10', 'AB11', 'AB12', 'AB13', 
              'AB14', 'AB15', 'AB16', 'AB17', 'AB18', 'AB19', 'AB20',
              'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB29','AB30']
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base_path = os.path.join(project_root, 'data', 'matlab_exported')
output_path = os.path.join(project_root, 'data', 'processed')

window_size = 200
stride = 100
rows = []

for subject in subject_id:
    # example of folder name C:\Users\DELTUS\Box\Courses\DeepLearning\Project\data\matlab_exported\AB15\1\treadmill\imu
    imu_trials_path = os.path.join(base_path, subject , '1', 'treadmill', 'imu')
    ik_trials_path = os.path.join(base_path, subject , '1', 'treadmill', 'ik')

    if not os.path.isdir(imu_trials_path) or not os.path.isdir(ik_trials_path):
        print(subject, "Missing expected folder(s):", imu_trials_path, "or", ik_trials_path)
        continue

    imu_trial_files = glob.glob(os.path.join(imu_trials_path, '*.csv'))
    ik_trial_files = glob.glob(os.path.join(ik_trials_path, '*.csv'))
    print(subject, "IMU:", len(imu_trial_files) , "IK:", len(ik_trial_files))
    for imu_file, ik_file in zip(imu_trial_files, ik_trial_files):
        trial_name = os.path.basename(imu_file).replace(".csv", "")

        rows.append({
            "subject_id": subject,
            "trial_id": trial_name,
            "imu_path": imu_file,
            "ik_path": ik_file
        })

metadata = pd.DataFrame(rows)

print(metadata.head())
print("Total paired trials:", len(metadata))

os.makedirs(output_path, exist_ok=True)
metadata.to_csv(os.path.join(output_path, "metadata.csv"), index=False)

metadata = pd.read_csv(os.path.join(output_path, "metadata.csv"))

X_all = []
y_all = []

for _, row in metadata.iterrows():
    imu = pd.read_csv(row["imu_path"])
    ik = pd.read_csv(row["ik_path"])

    print(row["subject_id"], row["trial_id"])
    print("IMU:", imu.shape, "IK:", ik.shape)
    break