import torch
import torch.nn as nn
import numpy as np
from functools import partial
from tqdm import tqdm

from mae import MAE

# Use cm positions from mne.get_montage() - these are only included for reference.
POSITIONS = {
    "Fp1": (-3.09, 11.46, 2.79),
    "Fp2": (2.84, 11.53, 2.77),
    "F3": (-5.18, 8.67, 7.87),
    "F4": (5.03, 8.74, 7.73),
    "F7": (-7.19, 7.31, 2.58),
    "F8": (7.14, 7.45, 2.51),
    "T3": (-8.60, 1.49, 3.12),
    "T4": (8.33, 1.53, 3.10),
    "C3": (-6.71, 2.34, 10.45),
    "C4": (6.53, 2.36, 10.37),
    "T5": (-8.77, 1.29, -0.77),
    "T6": (8.37, 1.17, -0.77),
    "P3": (-5.50, -4.42, 9.99),
    "P4": (5.36, -4.43, 10.05),
    "O1": (-3.16, -8.06, 5.48),
    "O2": (2.77, -8.05, 5.47),
    "Fz": (-0.12, 9.33, 10.26),
    "Cz": (-0.14, 2.76, 14.02),
    "Pz": (-0.17, -4.52, 12.67),
    "A2": (8.39, 0.20, -2.69),
}


class MANAS1(nn.Module):
    def __init__(self, checkpoint_path, num_classes=2, flat_dim=512):
        super().__init__()

        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        self.mae = MAE(fs=200, embed_dim=512, encoder_depth=12, encoder_heads=8, decoder_depth=4, decoder_heads=8, mask_ratio=0.55)
        self.mae.load_state_dict(ckpt["model_state_dict"])

        self.patch_embed = self.mae.patch_embed
        self.pos_enc = self.mae.pos_enc
        self.encoder = self.mae.encoder
        self.patch_size = self.mae.patch_size
        self.step = self.mae.step

        self.flat_dim = flat_dim

        # # The Head
        # self.final_layer = nn.Sequential(
        #     nn.Flatten(),
        #     nn.RMSNorm(self.flat_dim),  # Tutorial uses RMSNorm
        #     nn.Dropout(0.1),
        #     nn.Linear(self.flat_dim, num_classes),
        # )

    def prepare_coords(self, xyz, num_patches):
        B, C, _ = xyz.shape
        device = xyz.device
        time_idx = torch.arange(num_patches, device=device).float()
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)
        time = time_idx.view(1, 1, num_patches, 1).expand(B, C, -1, -1)
        return torch.cat([spat, time], dim=-1).flatten(1, 2)

    def forward(self, x, pos):
        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]

        tokens = self.patch_embed.linear(patches).flatten(1, 2)

        coords = self.prepare_coords(pos, num_patches)
        pe = self.pos_enc(coords)

        x_enc = tokens + pe
        latents, _ = self.encoder(x_enc)

        # add final layer for classification
        return latents