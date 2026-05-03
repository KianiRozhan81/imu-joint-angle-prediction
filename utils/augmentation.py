import re
import numpy as np


def build_sensor_groups(feature_names):

    pat = re.compile(
        r"^(?P<sensor>.+?)_(?P<kind>Acc|Accel|Gyr|Gyro)_(?P<axis>X|Y|Z)$",
        flags=re.IGNORECASE
    )
    raw = {}
    for idx, col in enumerate(feature_names):
        m = pat.match(col)
        if m is None:
            continue
        sensor   = m.group("sensor")
        kind_key = "acc" if m.group("kind").lower() in ("acc", "accel") else "gyr"
        axis     = m.group("axis").upper()
        raw.setdefault(sensor, {"acc": {}, "gyr": {}})
        raw[sensor][kind_key][axis] = idx

    cleaned = {}
    for sensor, d in raw.items():
        out = {}
        for k in ("acc", "gyr"):
            triad = d[k]
            if all(ax in triad for ax in ("X", "Y", "Z")):
                out[k] = [triad["X"], triad["Y"], triad["Z"]]
        if out:
            cleaned[sensor] = out
    return cleaned


def random_rotation_matrix(deg_std=20.0):
    """
    Generates a random 3D rotation matrix by sampling small
    rotation angles around each axis from N(0, deg_std).
    """
    ang = np.deg2rad(np.random.normal(0.0, deg_std, 3)).astype(np.float32)
    ax, ay, az = ang
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0,  0 ], [0,  cx, -sx], [0,  sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0,  1,   0 ], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz,  0], [0,   0,  1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def augment_window(x, sensor_groups,
                   jitter_sigma=0.05, rot_deg_std=20.0,
                   mode="rotation_jitter"):
    """
    Augments a single IMU window.

    Args:
        x             : np.ndarray [window_size, n_features]
        sensor_groups : output of build_sensor_groups()
        jitter_sigma  : std of Gaussian noise added to all channels
        rot_deg_std   : std of rotation angle in degrees per axis
        mode          : one of "none" | "jitter" | "rotation" | "rotation_jitter"

    Returns:
        augmented x   : np.ndarray [window_size, n_features]
    """
    x = x.copy()

    if mode in ("rotation", "rotation_jitter"):
        for sensor, groups in sensor_groups.items():
            R = random_rotation_matrix(rot_deg_std)
            for k in ("acc", "gyr"):
                if k in groups:
                    idx = groups[k]           # [x_idx, y_idx, z_idx]
                    x[:, idx] = x[:, idx] @ R.T

    if mode in ("jitter", "rotation_jitter"):
        x += np.random.normal(0.0, jitter_sigma, x.shape).astype(np.float32)

    return x.astype(np.float32)