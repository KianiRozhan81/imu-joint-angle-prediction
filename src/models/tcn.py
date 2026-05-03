# Input [batch, window_size, n_imu_features]
#     ↓
# Bidirectional LSTM  (one or more layers)
#     ↓
# Dropout  (regularization)
#     ↓
# Linear layer
#     ↓
# Output [batch, window_size, n_ik_outputs]

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class TemporalBlock(nn.Module):
    """
    One residual block of the TCN.
    Contains two dilated causal convolutions with the same dilation factor.
    Uses weight normalization and dropout for regularization.
    A residual (skip) connection adds the input back to the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, dropout=0.2):
        super().__init__()

        # padding = (kernel_size - 1) * dilation ensures causal convolution
        # i.e. prediction at t only uses timesteps <= t
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation))

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation))

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # residual connection: if channels differ, use 1x1 conv to match
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: [batch, channels, seq_len]

        out = self.conv1(x)
        # trim extra padding on the right to keep sequence length the same
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + self.residual(x))


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (Bai et al. 2018).
    Stacks TemporalBlocks with exponentially increasing dilation factors
    so that deeper layers have a larger receptive field.

    Reference: https://arxiv.org/abs/1803.01271
    """
    def __init__(self, input_size, output_size,
                 num_channels=None, kernel_size=3, dropout=0.2):
        super().__init__()

        # default: 4 blocks with 128 channels each
        if num_channels is None:
            num_channels = [128, 128, 128, 128]

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation  = 2 ** i          # 1, 2, 4, 8 ...
            in_ch     = input_size if i == 0 else num_channels[i - 1]
            out_ch    = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc      = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x:   [batch, seq_len, input_size]
        # TCN expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        out = self.network(x)           # [batch, num_channels[-1], seq_len]
        out = out.permute(0, 2, 1)      # [batch, seq_len, num_channels[-1]]
        out = self.fc(out)              # [batch, seq_len, output_size]
        return out