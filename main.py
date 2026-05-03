import os
import random
import numpy as np
import pandas as pd
import torch

from utils.scalers   import fit_scalers, load_scalers
from utils.splits    import subject_split
from src.dataset     import IMUDataset
from src.models.lstm import LSTMModel
from src.models.tcn  import TCNModel
from src.train       import train
from src.evaluate    import evaluate_metrics
from torch.utils.data import DataLoader


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

    # --- Config --- #

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
    NUM_WORKERS = 0   # set to 2 on Colab/Linux

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load metadata ---#

    metadata = pd.read_csv(metadata_path)
    print(f"Total trials: {len(metadata)}")

    # --- Split by subject ---#

    train_meta, val_meta, test_meta = subject_split(
        metadata, VAL_SUBJECTS, TEST_SUBJECTS)

    print(f"Train subjects: {train_meta['subject_id'].nunique()} "
          f"| Val: {val_meta['subject_id'].nunique()} "
          f"| Test: {test_meta['subject_id'].nunique()}")

    #--- Fit scalers on train only ----#

    imu_scaler, ik_scaler = fit_scalers(
        train_meta, IMU_COLS, IK_COLS, scaler_path)

    # ---- Build datasets and dataloaders ---- #

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

    # --- Train LSTM ---#

    print("\n" + "="*50)
    print("Training LSTM")
    print("="*50)

    lstm_model  = LSTMModel(input_size=input_size, output_size=output_size)
    lstm_losses = train(
        lstm_model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LR,
        checkpoint_path=os.path.join(checkpoint_dir, "lstm_best.pt"),
        device=device,
    )

    #---- Train TCN --- #

    print("\n" + "="*50)
    print("Training TCN")
    print("="*50)

    tcn_model  = TCNModel(input_size=input_size, output_size=output_size)
    tcn_losses = train(
        tcn_model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LR,
        checkpoint_path=os.path.join(checkpoint_dir, "tcn_best.pt"),
        device=device,
    )

    #---- Evaluate both on test set --- #

    print("\n" + "="*50)
    print("Test Set Evaluation — LSTM")
    print("="*50)

    lstm_model.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "lstm_best.pt"),
        map_location=device, weights_only=True))
    lstm_rmse, lstm_preds, lstm_targets = evaluate_metrics(
        lstm_model, test_loader, ik_scaler, IK_COLS, device)

    print("\n" + "="*50)
    print("Test Set Evaluation — TCN")
    print("="*50)

    tcn_model.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "tcn_best.pt"),
        map_location=device, weights_only=True))
    tcn_rmse, tcn_preds, tcn_targets = evaluate_metrics(
        tcn_model, test_loader, ik_scaler, IK_COLS, device)

    # ----- Save results ----- #

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "lstm_train_losses.npy"), lstm_losses[0])
    np.save(os.path.join(results_dir, "lstm_val_losses.npy"),   lstm_losses[1])
    np.save(os.path.join(results_dir, "tcn_train_losses.npy"),  tcn_losses[0])
    np.save(os.path.join(results_dir, "tcn_val_losses.npy"),    tcn_losses[1])
    np.save(os.path.join(results_dir, "lstm_preds.npy"),        lstm_preds)
    np.save(os.path.join(results_dir, "lstm_targets.npy"),      lstm_targets)
    np.save(os.path.join(results_dir, "tcn_preds.npy"),         tcn_preds)
    np.save(os.path.join(results_dir, "tcn_targets.npy"),       tcn_targets)

    print(f"\nResults saved → {results_dir}")
