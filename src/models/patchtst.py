import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Divides the input sequence into non-overlapping patches
    and projects each patch to d_model dimensions.
    """
    def __init__(self, patch_len, d_model, n_features, dropout=0.3):
        super().__init__()
        self.patch_len = patch_len
        # linear projection: patch_len * n_features → d_model
        self.projection = nn.Linear(patch_len * n_features, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        batch, seq_len, n_features = x.shape

        # trim sequence so it divides evenly into patches
        n_patches = seq_len // self.patch_len
        x = x[:, :n_patches * self.patch_len, :]

        # reshape into patches
        x = x.reshape(batch, n_patches, self.patch_len * n_features)

        # project to d_model
        x = self.projection(x)          # [batch, n_patches, d_model]
        return self.dropout(x), n_patches


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer (Nie et al. 2023).
    Divides the input sequence into patches and applies
    self-attention across patches for sequence-to-sequence prediction.

    Reference: https://arxiv.org/abs/2211.14730
    """
    def __init__(self, input_size, output_size,
                 patch_len=10, d_model=128, n_heads=8,
                 n_layers=3, dropout=0.3):
        super().__init__()

        self.patch_len   = patch_len
        self.output_size = output_size

        # patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len, d_model, input_size, dropout)

        # positional encoding (learnable)
        # max 500 patches to be safe
        self.pos_embedding = nn.Embedding(500, d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)

        # project from patch space back to timestep space
        # each patch covers patch_len timesteps
        self.fc = nn.Linear(d_model, patch_len * output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch, seq_len, _ = x.shape

        # patch embedding
        x_patches, n_patches = self.patch_embedding(x)
        # x_patches: [batch, n_patches, d_model]

        # add positional encoding
        positions = torch.arange(n_patches, device=x.device)
        x_patches = x_patches + self.pos_embedding(positions)

        # transformer encoder
        out = self.transformer(x_patches)   # [batch, n_patches, d_model]
        out = self.dropout(out)

        # project each patch back to patch_len timesteps
        out = self.fc(out)                  # [batch, n_patches, patch_len * output_size]

        # reshape to [batch, seq_len, output_size]
        out = out.reshape(batch, n_patches * self.patch_len, self.output_size)

        # if original seq_len was longer, pad with zeros
        if out.shape[1] < seq_len:
            pad = torch.zeros(batch, seq_len - out.shape[1],
                              self.output_size, device=x.device)
            out = torch.cat([out, pad], dim=1)

        return out