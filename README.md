# imu-joint-angle-prediction
# IMU-Based Joint Angle Prediction

**24-788 Introduction to Deep Learning — Spring 2026**  
**Carnegie Mellon University**  
**Rozhan Kiani** (rkiani) · **Babak Tarivirdilouyasl** (btarivir)

---

## What is this project?

Measuring joint angles during walking normally requires a full motion capture lab, expensive equipment, controlled environments, and a lot of setup. We asked a simpler question: can we get the same information from cheap, wearable IMU sensors?

In this project, we trained three deep learning models to predict hip, knee, and ankle joint angles from IMU data collected during treadmill walking. We compared a standard LSTM baseline against two more recent architectures, a Temporal Convolutional Network (TCN) and a Patch-based Transformer (PatchTST), to see which one generalizes best across subjects.

---

## Models

| Model | Type | Reference |
|-------|------|-----------|
| BiLSTM | Baseline | Course material |
| TCN | Variant 1 | Bai et al., 2018 |
| PatchTST | Variant 2 | Nie et al., 2023 |

---

## Bonus Contributions

This project exceeds the 2-person scope requirement by delivering two additional analyses:

**1. Learning Curves (Training Dynamics)**  
We trained all three models on subsets of 25%, 50%, 75%, and 100% of the training subjects and measured test RMSE at each level. This analysis directly addresses a clinically relevant question: how much labeled data is needed before these models generalize reliably across subjects?

**2. Rotation Augmentation (Model Mechanics)**  
Motivated by Um et al. (2017), we applied random 3D rotation augmentation to IMU signals during TCN training to simulate sensor misalignment between subjects. Each IMU sensor's accelerometer and gyroscope triads were independently rotated by a random angle sampled from a zero-mean Gaussian during each training step. We compare TCN with and without augmentation to assess its effect on cross-subject generalization.

---

## Dataset

23 able-bodied subjects (AB06–AB30) walking on a treadmill. Each subject has:
- **IMU data** — 4 sensors (foot, shank, thigh, trunk), each with accelerometer + gyroscope = 24 features at 200 Hz
- **Joint angles** — hip flexion, hip adduction, hip rotation, knee angle, ankle angle (right side) from inverse kinematics
- **Gait cycle events** — heel strike and toe off timing

**Download the data here:**  
[Google Drive](https://drive.google.com/drive/folders/1trD4-GB9OCNVug2qI5CdnYDbA279FuAn?usp=sharing)

Place it at `data/matlab_exported/` with this structure:
```
data/matlab_exported/
    AB06/1/treadmill/imu/
    AB06/1/treadmill/ik/
    AB06/1/treadmill/gcRight/
    AB07/...
```

---

## Project Structure

```
imu-joint-angle-prediction/
│
├── main.py                  # Train LSTM and TCN
├── train_patchtst.py        # Train PatchTST
├── train_tcn_aug.py         # Train TCN with rotation augmentation
├── learning_curves.py       # RMSE vs training set size analysis
│
├── src/
│   ├── dataset.py           # Sliding window dataset
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Per-joint RMSE evaluation
│   └── models/
│       ├── lstm.py          # Bidirectional LSTM
│       ├── tcn.py           # Temporal Convolutional Network
│       └── patchtst.py      # Patch Time Series Transformer
│
├── utils/
│   ├── scalers.py           # StandardScaler fit and load
│   ├── splits.py            # Subject-level data splitting
│   └── augmentation.py      # IMU rotation augmentation (Um et al., 2017)
│
└── notebooks/
    └── reproduce_results.ipynb   # Reproduces all figures and metrics
```

---

## Setup

```bash
conda create -n DLcourse python=3.9
conda activate DLcourse

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib scipy jupyter
```

---

## How to Reproduce Results

You don't need to retrain anything — just load the checkpoints and run the notebook.

### Step 1 — Get the data
Download from the Google Drive link above and place at `data/matlab_exported/`.

### Step 2 — Build metadata
```bash
python src/build_metadata.py
```

### Step 3 — Train models (optional — checkpoints already saved)
```bash
python main.py               # LSTM + TCN
python train_patchtst.py     # PatchTST
python train_tcn_aug.py      # TCN + rotation augmentation
python learning_curves.py    # Learning curve analysis
```

### Step 4 — Reproduce all figures and metrics
```bash
jupyter notebook notebooks/reproduce_results.ipynb
```
Run all cells. The notebook loads saved checkpoints and regenerates:
- Per-joint RMSE table
- Training curves
- Gait cycle plots (mean ± std band)
- Learning curves
- Augmentation comparison

---

## Data Splits

We split by subject — never by window — to prevent data leakage:

| Split | Subjects |
|-------|----------|
| Train | 18 subjects |
| Val   | AB28, AB29, AB30 |
| Test  | AB25, AB27 |

---

## Results

| Joint | LSTM (°) | TCN (°) | PatchTST (°) |
|-------|----------|---------|--------------|
| Hip Flexion | 6.38 | 7.21 | 6.97 |
| Hip Adduction | 5.04 | 4.66 | 4.88 |
| Hip Rotation | 7.05 | 6.66 | 8.68 |
| Knee Angle | 5.87 | 6.54 | 6.01 |
| Ankle Angle | 4.16 | 3.39 | 4.37 |
| **Mean** | **5.70** | **5.69** | **6.18** |

TCN and LSTM perform comparably overall. TCN edges out on ankle and hip rotation — joints with more rhythmic, periodic patterns. LSTM holds its own on knee and hip flexion. PatchTST underperforms slightly, likely because the 200-timestep windows are too short for patch-based attention to provide an advantage.

---

## References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
- Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- Um, T. T., et al. (2017). Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks. [arXiv:1706.00527](https://arxiv.org/abs/1706.00527)

---

## AI Tool Disclosure

We used Claude (Anthropic) to help with code structure and debugging. All modeling decisions, analysis, and written report content are our own.



