import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.scalers       import load_scalers
from utils.splits        import subject_split
from src.dataset         import IMUDataset
from src.models.patchtst import PatchTST
from src.train           import train
from src.evaluate        import evaluate_metrics


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

    WINDOW_SIZE = 200
    STRIDE      = 100
    BATCH_SIZE  = 64
    NUM_EPOCHS  = 200
    LR          = 1e-3
    NUM_WORKERS = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- 1. Load metadata and split ----- #

    metadata = pd.read_csv(metadata_path)
    train_meta, val_meta, test_meta = subject_split(
        metadata, VAL_SUBJECTS, TEST_SUBJECTS)

    print(f"Train subjects: {train_meta['subject_id'].nunique()} "
          f"| Val: {val_meta['subject_id'].nunique()} "
          f"| Test: {test_meta['subject_id'].nunique()}")

    # --- 2. Load existing scalers (fitted on train during main.py) ---

    imu_scaler, ik_scaler = load_scalers(scaler_path)
    print("Scalers loaded.")

    # --- 3. Build datasets ----#

    train_ds = IMUDataset(train_meta, WINDOW_SIZE, STRIDE,
                          IMU_COLS, IK_COLS, imu_scaler, ik_scaler)
    val_ds   = IMUDataset(val_meta,   WINDOW_SIZE, STRIDE,
                          IMU_COLS, IK_COLS, imu_scaler, ik_scaler)
    test_ds  = IMUDataset(test_meta,  WINDOW_SIZE, STRIDE,
                          IMU_COLS, IK_COLS, imu_scaler, ik_scaler)

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

    # ----- 4. Train PatchTST ----- #

    print("\n" + "="*50)
    print("Training PatchTST")
    print("="*50)

    patchtst_model  = PatchTST(input_size=input_size, output_size=output_size)
    patchtst_losses = train(
        patchtst_model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LR,
        checkpoint_path=os.path.join(checkpoint_dir, "patchtst_best.pt"),
        device=device,
    )

    # ----- 5. Evaluate on test set ----- #

    print("\n" + "="*50)
    print("Test Set Evaluation — PatchTST")
    print("="*50)

    patchtst_model.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "patchtst_best.pt"),
        map_location=device, weights_only=True))

    patchtst_rmse, patchtst_preds, patchtst_targets = evaluate_metrics(
        patchtst_model, test_loader, ik_scaler, IK_COLS, device)

    # ----- 6. Save results ----- #

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "patchtst_train_losses.npy"),
            patchtst_losses[0])
    np.save(os.path.join(results_dir, "patchtst_val_losses.npy"),
            patchtst_losses[1])
    np.save(os.path.join(results_dir, "patchtst_preds.npy"),   patchtst_preds)
    np.save(os.path.join(results_dir, "patchtst_targets.npy"), patchtst_targets)

    print(f"\nResults saved → {results_dir}")