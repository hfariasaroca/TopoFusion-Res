#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import torch
import torch.nn as nn


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
    def __init__(self, in_dim: int, out_ch: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x).unsqueeze(-1).unsqueeze(-1)


class ModelE2ResidualHead(nn.Module):
    """
    Model E2.

    Misma base que Model E phase1, pero agrega una cabeza residual aprendible:

        tas_final = tas_base + residual

    La última capa de residual_head se inicializa en cero, por lo que al cargar
    los pesos de phase1_best el modelo parte equivalente a Model E phase1.
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
        max_residual_delta: float = 2.5,
    ):
        super().__init__()

        self.dynamic_channels = dynamic_channels
        self.coarse_static_channels = coarse_static_channels
        self.fine_static_channels = fine_static_channels
        self.time_dim = time_dim
        self.ghg_dim = ghg_dim
        self.base_channels = base_channels
        self.max_residual_delta = float(max_residual_delta)

        in_coarse = dynamic_channels + coarse_static_channels

        # Misma base Model E
        self.enc_in = ConvBlock(in_coarse, base_channels, dropout=dropout)
        self.enc_res1 = ResidualBlock(base_channels, dropout=dropout)
        self.enc_res2 = ResidualBlock(base_channels, dropout=dropout)

        self.condition = GlobalConditioning(time_dim + ghg_dim, base_channels)

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

        self.fine_encoder = nn.Sequential(
            ConvBlock(fine_static_channels, base_channels // 2, dropout=dropout),
            ResidualBlock(base_channels // 2, dropout=dropout),
        )

        self.fusion = nn.Sequential(
            ConvBlock(base_channels + base_channels // 2, base_channels, dropout=dropout),
            ResidualBlock(base_channels, dropout=dropout),
            ResidualBlock(base_channels, dropout=dropout),
        )

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
        )

        # Nueva cabeza residual.
        # Usa features finales, estáticos finos y predicción base.
        residual_in = base_channels + fine_static_channels + 1

        self.residual_head = nn.Sequential(
            nn.Conv2d(residual_in, base_channels // 2, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(num_groups=min(8, base_channels // 2), num_channels=base_channels // 2),
            nn.GELU(),
            ResidualBlock(base_channels // 2, dropout=dropout),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
        )

        self._init_weights()
        self._init_residual_head_zero()

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

    def _init_residual_head_zero(self):
        # La última conv de residual_head parte en cero.
        # Así residual = 0 y tas_final = tas_base al inicio.
        last = self.residual_head[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def freeze_base(self):
        """
        Congela toda la base y deja entrenable solo residual_head.
        """
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("residual_head")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def trainable_parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "total_millions": total / 1e6,
            "trainable_millions": trainable / 1e6,
        }

    def forward_features(
        self,
        dynamic: torch.Tensor,
        coarse_static: torch.Tensor,
        fine_static: torch.Tensor,
        time_features: torch.Tensor,
        ghg_features: torch.Tensor,
    ):
        x = torch.cat([dynamic, coarse_static], dim=1)

        x = self.enc_in(x)
        x = self.enc_res1(x)

        cond = torch.cat([time_features, ghg_features], dim=1)
        cond = self.condition(cond)

        x = x + cond
        x = self.enc_res2(x)

        x = self.up1(x)
        x = self.up2(x)

        fine = self.fine_encoder(fine_static)

        x = torch.cat([x, fine], dim=1)
        x = self.fusion(x)

        return x

    def forward(
        self,
        dynamic: torch.Tensor,
        coarse_static: torch.Tensor,
        fine_static: torch.Tensor,
        time_features: torch.Tensor,
        ghg_features: torch.Tensor,
        return_parts: bool = False,
    ):
        features = self.forward_features(
            dynamic=dynamic,
            coarse_static=coarse_static,
            fine_static=fine_static,
            time_features=time_features,
            ghg_features=ghg_features,
        )

        tas_base = self.head(features)

        residual_input = torch.cat([features, fine_static, tas_base], dim=1)
        residual_raw = self.residual_head(residual_input)

        residual = self.max_residual_delta * torch.tanh(residual_raw / self.max_residual_delta)

        tas_final = tas_base + residual

        if return_parts:
            return {
                "tas_final": tas_final,
                "tas_base": tas_base,
                "residual": residual,
                "residual_raw": residual_raw,
            }

        return tas_final


def load_phase1_weights_into_e2(model: ModelE2ResidualHead, checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)

    allowed_missing = [k for k in missing if k.startswith("residual_head")]

    if len(allowed_missing) != len(missing):
        raise RuntimeError(
            "Faltan pesos inesperados al cargar phase1 en E2:\n"
            + "\n".join(missing)
        )

    if unexpected:
        raise RuntimeError(
            "Pesos inesperados al cargar phase1 en E2:\n"
            + "\n".join(unexpected)
        )

    return {
        "missing": missing,
        "unexpected": unexpected,
        "source_epoch": ckpt.get("epoch"),
        "source_metrics": ckpt.get("metrics", {}),
    }


if __name__ == "__main__":
    model = ModelE2ResidualHead()
    print(model.trainable_parameter_summary())

    batch = {
        "dynamic": torch.randn(2, 24, 22, 22),
        "coarse_static": torch.randn(2, 1, 22, 22),
        "fine_static": torch.randn(2, 6, 64, 64),
        "time_features": torch.randn(2, 3),
        "ghg_features": torch.randn(2, 4),
    }

    with torch.no_grad():
        out = model(**batch, return_parts=True)

    for k, v in out.items():
        print(k, tuple(v.shape), float(v.mean()), float(v.std()))
