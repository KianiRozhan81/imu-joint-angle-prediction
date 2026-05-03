# 1. run model on test loader
# 2. collect all predictions and ground truth
# 3. denormalize both using ik_scaler
# 4. compute RMSE per joint
# 5. print results table



import numpy as np
import torch


def evaluate_metrics(model, loader, ik_scaler, ik_cols, device=None):
    """
    Runs model on loader, denormalizes predictions and ground truth,
    computes RMSE per joint in degrees.

    Returns:
        rmse_per_joint : dict  {joint_name: rmse_in_degrees}
        all_preds      : np.array [n_samples * window_size, n_joints]
        all_targets    : np.array [n_samples * window_size, n_joints]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)                         # [batch, seq_len, n_joints]

            preds_list.append(pred.cpu().numpy())
            targets_list.append(y.numpy())

    # stack all batches → [n_windows, seq_len, n_joints]
    all_preds   = np.concatenate(preds_list,   axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    # flatten windows → [n_windows * seq_len, n_joints]
    n_windows, seq_len, n_joints = all_preds.shape
    all_preds   = all_preds.reshape(-1, n_joints)
    all_targets = all_targets.reshape(-1, n_joints)

    # denormalize back to degrees using ik_scaler
    all_preds   = ik_scaler.inverse_transform(all_preds)
    all_targets = ik_scaler.inverse_transform(all_targets)

    # RMSE per joint
    rmse_per_joint = {}
    for i, joint in enumerate(ik_cols):
        rmse = np.sqrt(np.mean((all_preds[:, i] - all_targets[:, i]) ** 2))
        rmse_per_joint[joint] = rmse

    # print results table
    print(f"\n{'Joint':<20} {'RMSE (°)':>10}")
    print("-" * 32)
    for joint, rmse in rmse_per_joint.items():
        print(f"{joint:<20} {rmse:>10.2f}")
    mean_rmse = np.mean(list(rmse_per_joint.values()))
    print("-" * 32)
    print(f"{'Mean':<20} {mean_rmse:>10.2f}")

    return rmse_per_joint, all_preds, all_targets