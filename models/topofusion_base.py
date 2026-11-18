#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=min(8, ch), num_channels=ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=min(8, ch), num_channels=ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class GlobalConditioning(nn.Module):
    """
    Proyecta time_features + ghg_features a un vector de condicionamiento global.
    Luego se inyecta como bias espacial [B, C, 1, 1].
    """

    def __init__(self, in_dim: int, out_ch: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, x):
        # x: [B, in_dim]
        return self.net(x).unsqueeze(-1).unsqueeze(-1)


class ModelECMIPSwinUNetTopography(nn.Module):
    """
    Model E v1.

    Entradas:
      dynamic        [B, 24, 22, 22]
      coarse_static  [B, 1, 22, 22]
      fine_static    [B, 6, 64, 64]
      time_features  [B, 3]
      ghg_features   [B, 4]

    Salida:
      tas_pred       [B, 1, 64, 64] en °C
    """

    def __init__(
        self,
        dynamic_channels: int = 24,
        coarse_static_channels: int = 1,
        fine_static_channels: int = 6,
        time_dim: int = 3,
        ghg_dim: int = 4,
        base_channels: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.dynamic_channels = dynamic_channels
        self.coarse_static_channels = coarse_static_channels
        self.fine_static_channels = fine_static_channels
        self.time_dim = time_dim
        self.ghg_dim = ghg_dim
        self.base_channels = base_channels

        in_coarse = dynamic_channels + coarse_static_channels

        # Encoder 22x22
        self.enc_in = ConvBlock(in_coarse, base_channels, dropout=dropout)
        self.enc_res1 = ResidualBlock(base_channels, dropout=dropout)
        self.enc_res2 = ResidualBlock(base_channels, dropout=dropout)

        # Condicionamiento global: tiempo + GHG
        self.condition = GlobalConditioning(time_dim + ghg_dim, base_channels)

        # Upsampling 22x22 -> 44x44 -> 64x64
        self.up1 = nn.Sequential(
            nn.Upsample(size=(44, 44), mode="bilinear", align_corners=False),
            ConvBlock(base_channels, base_channels, dropout=dropout),
            ResidualBlock(base_channels, dropout=dropout),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
            ConvBlock(base_channels, base_channels, dropout=dropout),
            ResidualBlock(base_channels, dropout=dropout),
        )

        # Rama topográfica fina 64x64
        self.fine_encoder = nn.Sequential(
            ConvBlock(fine_static_channels, base_channels // 2, dropout=dropout),
            ResidualBlock(base_channels // 2, dropout=dropout),
        )

        # Fusión 64x64
        self.fusion = nn.Sequential(
            ConvBlock(base_channels + base_channels // 2, base_channels, dropout=dropout),
            ResidualBlock(base_channels, dropout=dropout),
            ResidualBlock(base_channels, dropout=dropout),
        )

        # Head final
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        dynamic: torch.Tensor,
        coarse_static: torch.Tensor,
        fine_static: torch.Tensor,
        time_features: torch.Tensor,
        ghg_features: torch.Tensor,
    ) -> torch.Tensor:
        # Validaciones mínimas
        if dynamic.ndim != 4:
            raise ValueError(f"dynamic debe ser [B,C,22,22], got {dynamic.shape}")

        if coarse_static.ndim != 4:
            raise ValueError(f"coarse_static debe ser [B,C,22,22], got {coarse_static.shape}")

        if fine_static.ndim != 4:
            raise ValueError(f"fine_static debe ser [B,C,64,64], got {fine_static.shape}")

        # 22x22
        x = torch.cat([dynamic, coarse_static], dim=1)

        x = self.enc_in(x)
        x = self.enc_res1(x)

        cond = torch.cat([time_features, ghg_features], dim=1)
        cond = self.condition(cond)
        x = x + cond

        x = self.enc_res2(x)

        # 64x64
        x = self.up1(x)
        x = self.up2(x)

        fine = self.fine_encoder(fine_static)

        x = torch.cat([x, fine], dim=1)
        x = self.fusion(x)

        out = self.head(x)

        return out


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


if __name__ == "__main__":
    model = ModelECMIPSwinUNetTopography()

    print(model)
    print(count_parameters(model))

    batch = {
        "dynamic": torch.randn(2, 24, 22, 22),
        "coarse_static": torch.randn(2, 1, 22, 22),
        "fine_static": torch.randn(2, 6, 64, 64),
        "time_features": torch.randn(2, 3),
        "ghg_features": torch.randn(2, 4),
    }

    with torch.no_grad():
        y = model(**batch)

    print("Output:", y.shape)
