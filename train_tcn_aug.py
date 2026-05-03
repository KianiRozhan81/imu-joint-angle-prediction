import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.scalers      import load_scalers
from utils.splits       import subject_split
from utils.augmentation import build_sensor_groups, augment_window
from src.models.tcn     import TCNModel
from src.train          import train
from src.evaluate       import evaluate_metrics


# ----- Augmented Dataset ----- #

class IMUDatasetAug(Dataset):
    """
    Same as IMUDataset but with on-the-fly augmentation for training.
    Val/test use this with augment=False.
    """
    def __init__(self, metadata, window_size, stride,
                 imu_cols, ik_cols,
                 imu_scaler=None, ik_scaler=None,
                 augment=False, aug_mode="rotation",
                 jitter_sigma=0.05, rot_deg_std=20.0):

        self.augment       = augment
        self.aug_mode      = aug_mode
        self.jitter_sigma  = jitter_sigma
        self.rot_deg_std   = rot_deg_std
        self.sensor_groups = build_sensor_groups(imu_cols)
        self.windows       = []

        for _, row in metadata.iterrows():
            imu = pd.read_csv(row["imu_path"])[imu_cols].values.astype(np.float32)
            ik  = pd.read_csv(row["ik_path"])[ik_cols].values.astype(np.float32)

            n = min(len(imu), len(ik))
            imu, ik = imu[:n], ik[:n]

            if imu_scaler is not None:
                imu = imu_scaler.transform(imu).astype(np.float32)
            if ik_scaler is not None:
                ik  = ik_scaler.transform(ik).astype(np.float32)

            for start in range(0, n - window_size + 1, stride):
                end = start + window_size
                self.windows.append((imu[start:end], ik[start:end]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        imu_win, ik_win = self.windows[idx]
        if self.augment and self.aug_mode != "none":
            imu_win = augment_window(
                imu_win,
                sensor_groups=self.sensor_groups,
                jitter_sigma=self.jitter_sigma,
                rot_deg_std=self.rot_deg_std,
                mode=self.aug_mode,
            )
        return torch.from_numpy(imu_win.copy()), torch.from_numpy(ik_win.copy())


# ----- Column config ----- #

IMU_COLS = [
    "foot_Accel_X",  "foot_Accel_Y",  "foot_Accel_Z",
    "foot_Gyro_X",   "foot_Gyro_Y",   "foot_Gyro_Z",
    "shank_Accel_X", "shank_Accel_Y", "shank_Accel_Z",
    "shank_Gyro_X",  "shank_Gyro_Y",  "shank_Gyro_Z",
    "thigh_Accel_X", "thigh_Accel_Y", "thigh_Accel_Z",
    "thigh_Gyro_X",  "thigh_Gyro_Y",  "thigh_Gyro_Z",
    "trunk_Accel_X", "trunk_Accel_Y", "trunk_Accel_Z",
    "trunk_Gyro_X",  "trunk_Gyro_Y",  "trunk_Gyro_Z",
]

IK_COLS = [
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
]


if __name__ == "__main__":

    # ----- Reproducibility ----- #

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ----- Config ----- #

    project_root   = os.path.abspath(os.path.dirname(__file__))
    metadata_path  = os.path.join(project_root, "data", "processed", "metadata.csv")
    scaler_path    = os.path.join(project_root, "data", "processed", "scalers")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    results_dir    = os.path.join(project_root, "results")

    VAL_SUBJECTS  = ["AB28", "AB29", "AB30"]
    TEST_SUBJECTS = ["AB25", "AB27"]

    WINDOW_SIZE  = 200
    STRIDE       = 100
    BATCH_SIZE   = 64
    NUM_EPOCHS   = 200
    LR           = 1e-3
    NUM_WORKERS  = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- 1. Load metadata and split ----- #

    metadata = pd.read_csv(metadata_path)
    train_meta, val_meta, test_meta = subject_split(
        metadata, VAL_SUBJECTS, TEST_SUBJECTS)

    print(f"Train subjects: {train_meta['subject_id'].nunique()} "
          f"| Val: {val_meta['subject_id'].nunique()} "
          f"| Test: {test_meta['subject_id'].nunique()}")

    # ----- 2. Load existing scalers ----- #

    imu_scaler, ik_scaler = load_scalers(scaler_path)
    print("Scalers loaded.")

    # ----- 3. Build datasets ----- #

    train_ds = IMUDatasetAug(
        train_meta, WINDOW_SIZE, STRIDE,
        IMU_COLS, IK_COLS, imu_scaler, ik_scaler,
        augment=True, aug_mode="rotation",
        jitter_sigma=0.05, rot_deg_std=20.0)

    val_ds  = IMUDatasetAug(
        val_meta, WINDOW_SIZE, STRIDE,
        IMU_COLS, IK_COLS, imu_scaler, ik_scaler,
        augment=False)

    test_ds = IMUDatasetAug(
        test_meta, WINDOW_SIZE, STRIDE,
        IMU_COLS, IK_COLS, imu_scaler, ik_scaler,
        augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    input_size  = len(IMU_COLS)   # 24
    output_size = len(IK_COLS)    # 5

    # ----- 4. Train TCN with augmentation ----- #

    print("\n" + "="*50)
    print("Training TCN + Augmentation")
    print("="*50)

    tcn_aug_model  = TCNModel(input_size=input_size, output_size=output_size)
    tcn_aug_losses = train(
        tcn_aug_model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LR,
        checkpoint_path=os.path.join(checkpoint_dir, "tcn_aug_best.pt"),
        device=device,
    )

    # ----- 5. Evaluate on test set ----- #

    print("\n" + "="*50)
    print("Test Set Evaluation — TCN + Augmentation")
    print("="*50)

    tcn_aug_model.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "tcn_aug_best.pt"),
        map_location=device, weights_only=True))

    tcn_aug_rmse, tcn_aug_preds, tcn_aug_targets = evaluate_metrics(
        tcn_aug_model, test_loader, ik_scaler, IK_COLS, device)

    # ----- 6. Save results ----- #

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "tcn_aug_train_losses.npy"), tcn_aug_losses[0])
    np.save(os.path.join(results_dir, "tcn_aug_val_losses.npy"),   tcn_aug_losses[1])
    np.save(os.path.join(results_dir, "tcn_aug_preds.npy"),        tcn_aug_preds)
    np.save(os.path.join(results_dir, "tcn_aug_targets.npy"),      tcn_aug_targets)

    print(f"\nResults saved → {results_dir}")