import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.scalers       import fit_scalers
from utils.splits        import subject_split
from src.dataset         import IMUDataset
from src.models.lstm     import LSTMModel
from src.models.tcn      import TCNModel
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



    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # --- Config --- #

    # project root is wherever this script lives
    project_root      = os.path.abspath(os.path.dirname(__file__))
    metadata_path     = os.path.join(project_root, "data", "processed", "metadata.csv")
    scaler_base_path  = os.path.join(project_root, "data", "processed", "scalers_lc")
    checkpoint_dir    = os.path.join(project_root, "checkpoints")
    results_dir       = os.path.join(project_root, "results")

    VAL_SUBJECTS  = ["AB28", "AB29", "AB30"]
    TEST_SUBJECTS = ["AB25", "AB27"]

    FRACTIONS   = [0.25, 0.50, 0.75, 1.00]
    WINDOW_SIZE = 200
    STRIDE      = 100
    BATCH_SIZE  = 64
    NUM_EPOCHS  = 100     # shorter than main — goal is trend, not best RMSE
    LR          = 1e-3
    NUM_WORKERS = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- Load metadata and split ----- #

    metadata = pd.read_csv(metadata_path)
    train_meta, val_meta, test_meta = subject_split(
        metadata, VAL_SUBJECTS, TEST_SUBJECTS)

    train_subjects = train_meta["subject_id"].unique().tolist()
    input_size     = len(IMU_COLS)
    output_size    = len(IK_COLS)

    print(f"Train subjects: {len(train_subjects)} "
          f"| Val: {val_meta['subject_id'].nunique()} "
          f"| Test: {test_meta['subject_id'].nunique()}")

    # ----- Learning curve loop ----- #

    results = {f: {} for f in FRACTIONS}

    for frac in FRACTIONS:
        n_subjects = max(1, int(len(train_subjects) * frac))

        # fixed subset controlled by seed
        random.seed(SEED)
        subset_subjects = random.sample(train_subjects, n_subjects)
        subset_meta     = train_meta[
            train_meta["subject_id"].isin(subset_subjects)]

        print(f"\n{'='*60}")
        print(f"Fraction {frac:.0%} — {n_subjects} training subjects: "
              f"{subset_subjects}")
        print(f"{'='*60}")

        # fit scalers on this subset only
        scaler_path = os.path.join(scaler_base_path, f"frac_{int(frac*100)}")
        imu_scaler, ik_scaler = fit_scalers(
            subset_meta, IMU_COLS, IK_COLS, scaler_path)

        # build datasets
        train_ds = IMUDataset(subset_meta, WINDOW_SIZE, STRIDE,
                              IMU_COLS, IK_COLS, imu_scaler, ik_scaler)
        val_ds   = IMUDataset(val_meta,    WINDOW_SIZE, STRIDE,
                              IMU_COLS, IK_COLS, imu_scaler, ik_scaler)
        test_ds  = IMUDataset(test_meta,   WINDOW_SIZE, STRIDE,
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

        # train each model
        models = {
            "LSTM":     LSTMModel(input_size, output_size),
            "TCN":      TCNModel(input_size, output_size),
            "PatchTST": PatchTST(input_size, output_size),
        }

        for model_name, model in models.items():
            print(f"\n--- {model_name} | fraction {frac:.0%} ---")

            # use fraction-specific checkpoint — never overwrites main checkpoints
            ckpt = os.path.join(
                checkpoint_dir,
                f"{model_name.lower()}_frac{int(frac*100)}.pt")

            train(model, train_loader, val_loader,
                  num_epochs=NUM_EPOCHS, lr=LR,
                  checkpoint_path=ckpt, device=device)

            model.load_state_dict(torch.load(
                ckpt, map_location=device, weights_only=True))

            rmse_per_joint, _, _ = evaluate_metrics(
                model, test_loader, ik_scaler, IK_COLS, device)

            mean_rmse = np.mean(list(rmse_per_joint.values()))
            results[frac][model_name] = mean_rmse
            print(f"{model_name} mean RMSE @ {frac:.0%}: {mean_rmse:.2f}°")

    # ----- Save results to CSV and print table ----#

    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for frac, model_results in results.items():
        row = {"fraction": frac}
        row.update(model_results)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "learning_curves.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nLearning curve results saved → {csv_path}")
    print(df.to_string(index=False))

    # ----- Plot -----

    n_subjects_axis = (df["fraction"] * len(train_subjects)).astype(int)
    colors = {"LSTM": "#1f77b4", "TCN": "#ff7f0e", "PatchTST": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color in colors.items():
        if model_name in df.columns:
            ax.plot(n_subjects_axis, df[model_name],
                    marker="o", label=model_name,
                    color=color, linewidth=2)

    ax.set_xlabel("Number of training subjects", fontsize=12)
    ax.set_ylabel("Mean RMSE (°)", fontsize=12)
    ax.set_title("Learning Curves — RMSE vs Training Set Size", fontsize=13)
    ax.set_xticks(n_subjects_axis)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Learning curve plot saved → {plot_path}")