# IMU-Based Joint Angle Prediction

**24-788 Introduction to Deep Learning — Spring 2026**  
**Carnegie Mellon University**  
**Rozhan Kiani** (rkiani) · **Babak Tarivirdilouyasl** (btarivir)

---

## What is this project?

Measuring joint angles during walking normally requires a full motion capture lab — expensive equipment, controlled environments, and a lot of setup. We asked a simpler question: can we get the same information from cheap, wearable IMU sensors?

In this project we trained three deep learning models to predict hip, knee, and ankle joint angles from IMU data collected during treadmill walking. We compared a standard bidirectional LSTM baseline against two more recent architectures — a Temporal Convolutional Network (TCN) and a Patch-based Transformer (PatchTST) — to see which one generalizes best across subjects.

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
We trained all three models on subsets of 25%, 50%, 75%, and 100% of training subjects and measured test RMSE at each level. This directly addresses a clinically relevant question: how much labeled data is needed before models generalize reliably across subjects?

**2. Rotation Augmentation (Model Mechanics)**  
Motivated by Um et al. (2017), we applied random 3D rotation augmentation to IMU signals during TCN training to simulate sensor misalignment between subjects. We compare TCN with and without augmentation to assess its effect on cross-subject generalization.

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

TCN and LSTM perform comparably overall. TCN is stronger at ankle and hip rotation — joints with more rhythmic periodic patterns. PatchTST underperforms slightly, likely because the 200-timestep windows are too short for patch-level attention to provide an advantage over simpler architectures.

---

## Project Structure

```
imu-joint-angle-prediction/
│
├── main.py                  # Train LSTM and TCN (200 epochs)
├── train_patchtst.py        # Train PatchTST (200 epochs)
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
├── checkpoints/             # Saved model weights (included in repo)
├── results/                 # Saved losses, predictions, and figures
│
└── notebooks/
    └── reproduce_results.ipynb   # Reproduces all figures and metrics
```

---

## Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/KianiRozhan81/imu-joint-angle-prediction.git
cd imu-joint-angle-prediction
```

### Step 2 — Create and activate the environment

```bash
conda create -n DLcourse python=3.9
conda activate DLcourse
```

### Step 3 — Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib scipy jupyter
```

> If you don't have a CUDA GPU, install the CPU version instead:
> ```bash
> pip install torch torchvision torchaudio
> ```

### Step 4 — Download the dataset

The raw data is hosted on Google Drive. Download the `data` folder from:

**[Google Drive Dataset Link](https://drive.google.com/drive/folders/1trD4-GB9OCNVug2qI5CdnYDbA279FuAn?usp=sharing)**

The Drive contains a `data/` folder with two subfolders:
- `matlab_exported/` — raw IMU and IK CSV files organized by subject
- `processed/` — metadata CSV and fitted scalers

Place both folders inside the `data/` directory of the cloned repo so the structure looks like:

```
imu-joint-angle-prediction/
└── data/
    ├── matlab_exported/
    │   ├── AB06/
    │   │   └── 1/treadmill/
    │   │       ├── imu/
    │   │       ├── ik/
    │   │       └── gcRight/
    │   ├── AB07/
    │   └── ...
    └── processed/
        ├── metadata.csv
        └── scalers/
```

---

## Reproducing Results

All model checkpoints and precomputed results are already included in the repo. You do not need to retrain anything.

### Reproduce all figures and metrics

```bash
jupyter notebook notebooks/reproduce_results.ipynb
```

Run all cells. The notebook will generate:
- Per-joint RMSE table for all three models
- Training and validation loss curves
- Gait cycle plots (mean ± std band over 0–100% gait cycle)
- Learning curves (RMSE vs number of training subjects)
- TCN vs TCN + augmentation comparison

### Retrain from scratch (optional)

If you want to retrain the models yourself:

```bash
python main.py               # Train LSTM + TCN
python train_patchtst.py     # Train PatchTST
python train_tcn_aug.py      # Train TCN with rotation augmentation
python learning_curves.py    # Learning curve analysis
```

> Training was done on an NVIDIA GPU. Each model takes approximately 30–60 minutes for 200 epochs.

---

## Data Splits

Splits are subject-level — never by window — to prevent data leakage:

| Split | Subjects |
|-------|----------|
| Train | 18 subjects (AB06–AB24, excluding val/test) |
| Val   | AB28, AB29, AB30 |
| Test  | AB25, AB27 |

---

## References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)
- Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- Um, T. T., et al. (2017). Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks. [arXiv:1706.00527](https://arxiv.org/abs/1706.00527)

---

## AI Tool Disclosure

We used Claude (Anthropic) to help with code structure and debugging. All modeling decisions, analysis, and written report content are our own.
