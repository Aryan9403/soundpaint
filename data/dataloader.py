"""
DataLoader factory for AudioTokenDataset.
"""

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import AudioTokenDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders from config dict.

    Args:
        config: dict with keys: token_dir, seq_len, pad_token_id, val_split,
                seed, batch_size, num_workers, pin_memory

    Returns:
        (train_loader, val_loader)
    """
    dataset = AudioTokenDataset(
        token_dir=config["token_dir"],
        seq_len=config["seq_len"],
        pad_token_id=config["pad_token_id"],
    )

    val_size = max(1, int(len(dataset) * config.get("val_split", 0.05)))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=config.get("pin_memory", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=config.get("pin_memory", True),
        drop_last=False,
    )

    print(f"Dataset: {len(dataset)} total | {len(train_ds)} train | {len(val_ds)} val")
    return train_loader, val_loader
