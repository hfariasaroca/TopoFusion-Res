#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/config_model_E.json"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    return json.loads(config_path.read_text())


def resolve_dataset_dir(cfg: dict) -> Path:
    dataset_path = Path(cfg["dataset_path"])
    if dataset_path.is_absolute():
        return dataset_path
    return ROOT / dataset_path


class ModelEDataset(Dataset):
    """
    Dataset oficial para Model E.

    Entrega por muestra:

    dynamic        [24, 22, 22]
    coarse_static  [1, 22, 22]
    fine_static    [6, 64, 64]
    time_features  [3]
    ghg_features   [4]
    target         [1, 64, 64]
    mask           [1, 64, 64]
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: Literal["train", "val", "test"],
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split

        if split == "train":
            self.inputs_path = self.dataset_dir / "train_inputs.npy"
            self.targets_path = self.dataset_dir / "train_targets.npy"
            self.time_features_path = self.dataset_dir / "time_features_train.npy"
            self.ghg_features_path = self.dataset_dir / "ghg_features_train.npy"
            self.time_path = self.dataset_dir / "train_time.npy"

        elif split == "val":
            self.inputs_path = self.dataset_dir / "val_inputs.npy"
            self.targets_path = self.dataset_dir / "val_targets.npy"
            self.time_features_path = self.dataset_dir / "time_features_val.npy"
            self.ghg_features_path = self.dataset_dir / "ghg_features_val.npy"
            self.time_path = self.dataset_dir / "val_time.npy"

        elif split == "test":
            self.inputs_path = self.dataset_dir / "test_inputs_ssp370.npy"
            self.targets_path = self.dataset_dir / "test_targets_ssp370.npy"
            self.time_features_path = self.dataset_dir / "time_features_test_ssp370.npy"
            self.ghg_features_path = self.dataset_dir / "ghg_features_test_ssp370.npy"
            self.time_path = self.dataset_dir / "test_time_ssp370.npy"

        else:
            raise ValueError(f"split inválido: {split}")

        required = [
            self.inputs_path,
            self.targets_path,
            self.time_features_path,
            self.ghg_features_path,
            self.time_path,
            self.dataset_dir / "coarse_static_22x22.npy",
            self.dataset_dir / "static_features_64x64.npy",
            self.dataset_dir / "target_valid_mask_64x64.npy",
        ]

        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Faltan archivos:\n" + "\n".join(missing))

        # Grandes: mmap
        self.inputs = np.load(self.inputs_path, mmap_mode="r")
        self.targets = np.load(self.targets_path, mmap_mode="r")
        self.time_features = np.load(self.time_features_path, mmap_mode="r")
        self.ghg_features = np.load(self.ghg_features_path, mmap_mode="r")
        self.times = np.load(self.time_path, mmap_mode="r")

        # Pequeños: memoria normal
        self.coarse_static = np.load(self.dataset_dir / "coarse_static_22x22.npy").astype(np.float32)
        self.fine_static = np.load(self.dataset_dir / "static_features_64x64.npy").astype(np.float32)
        self.mask = np.load(self.dataset_dir / "target_valid_mask_64x64.npy").astype(np.float32)

        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError(f"inputs y targets tienen largo distinto: {self.inputs.shape[0]} vs {self.targets.shape[0]}")

        if self.inputs.shape[0] != self.time_features.shape[0]:
            raise ValueError("inputs y time_features tienen largo distinto")

        if self.inputs.shape[0] != self.ghg_features.shape[0]:
            raise ValueError("inputs y ghg_features tienen largo distinto")

        if self.inputs.shape[1:] != (24, 22, 22):
            raise ValueError(f"inputs shape inesperado: {self.inputs.shape}")

        if self.targets.shape[1:] != (64, 64):
            raise ValueError(f"targets shape inesperado: {self.targets.shape}")

        if self.coarse_static.shape != (1, 22, 22):
            raise ValueError(f"coarse_static shape inesperado: {self.coarse_static.shape}")

        if self.fine_static.shape != (6, 64, 64):
            raise ValueError(f"fine_static shape inesperado: {self.fine_static.shape}")

        if self.mask.shape != (64, 64):
            raise ValueError(f"mask shape inesperado: {self.mask.shape}")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # np.array(..., copy=True) evita warning de arrays read-only por mmap.
        dynamic = torch.from_numpy(np.array(self.inputs[idx], dtype=np.float32, copy=True))
        target = torch.from_numpy(np.array(self.targets[idx], dtype=np.float32, copy=True)).unsqueeze(0)

        time_features = torch.from_numpy(np.array(self.time_features[idx], dtype=np.float32, copy=True))
        ghg_features = torch.from_numpy(np.array(self.ghg_features[idx], dtype=np.float32, copy=True))

        coarse_static = torch.from_numpy(self.coarse_static.copy())
        fine_static = torch.from_numpy(self.fine_static.copy())
        mask = torch.from_numpy(self.mask.copy()).unsqueeze(0)

        return {
            "dynamic": dynamic,
            "coarse_static": coarse_static,
            "fine_static": fine_static,
            "time_features": time_features,
            "ghg_features": ghg_features,
            "target": target,
            "mask": mask,
            "index": torch.tensor(idx, dtype=torch.long),
        }


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

    dataset = ModelEDataset(dataset_dir=dataset_dir, split=split)

    if shuffle is None:
        shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader


if __name__ == "__main__":
    cfg = load_config()
    dataset_dir = resolve_dataset_dir(cfg)

    print("Dataset dir:", dataset_dir)

    for split in ["train", "val", "test"]:
        ds = ModelEDataset(dataset_dir, split=split)
        print(split, "len:", len(ds))

        sample = ds[0]
        for k, v in sample.items():
            if torch.is_tensor(v):
                print(f"  {k:14s}: {tuple(v.shape)} {v.dtype}")
        print()
