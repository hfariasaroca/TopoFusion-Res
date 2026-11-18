#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from dataset_model_E import (
    ModelEDataset,
    load_config,
    resolve_dataset_dir,
    CONFIG_PATH,
)

FINE_TOPO_CHANNELS = (0, 4, 5)


class ModelEAblationDataset(ModelEDataset):

    def __init__(
        self,
        dataset_dir: str | Path,
        split: Literal["train", "val", "test"],
        no_fine_topo: bool = False,
    ):
        super().__init__(dataset_dir=dataset_dir, split=split)

        self.no_fine_topo = bool(no_fine_topo)

        if self.no_fine_topo:
            self.fine_static = self.fine_static.copy()
            self.fine_static[list(FINE_TOPO_CHANNELS)] = 0.0


def create_dataloader(
    split: Literal["train", "val", "test"],
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    config_path: str | Path = CONFIG_PATH,
) -> DataLoader:

    cfg = load_config(Path(config_path))
    dataset_dir = resolve_dataset_dir(cfg)

    mode = os.environ.get(
        "MODEL_E_ABLATION_MODE",
        "FULL"
    ).upper()

    if mode not in {"FULL", "NO_FINE_TOPO"}:
        raise ValueError(
            f"MODEL_E_ABLATION_MODE inválido: {mode}"
        )

    no_fine_topo = mode == "NO_FINE_TOPO"

    dataset = ModelEAblationDataset(
        dataset_dir=dataset_dir,
        split=split,
        no_fine_topo=no_fine_topo,
    )

    if shuffle is None:
        shuffle = split == "train"

    seed = int(
        os.environ.get(
            "MODEL_E_SEED",
            "20260813"
        )
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
