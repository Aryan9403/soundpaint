"""
Dataset for audio tokens produced by data/prepare.py.

Each .pt file contains a 1D LongTensor of codebook-0 tokens.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset


class AudioTokenDataset(Dataset):
    """
    Loads all .pt token files from token_dir into memory.

    Returns (input_ids, target_ids) pairs:
        input_ids  = tokens[:-1]   shape (seq_len,)
        target_ids = tokens[1:]    shape (seq_len,)

    Sequences are truncated or right-padded with pad_token_id to seq_len.
    """

    def __init__(
        self,
        token_dir: str,
        seq_len: int = 2580,
        pad_token_id: int = 1024,
    ):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        token_dir = Path(token_dir)
        pt_files = sorted(token_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {token_dir}")

        self.data = []
        for f in pt_files:
            tokens = torch.load(f, weights_only=True)  # (L,)
            tokens = self._pad_or_truncate(tokens)
            self.data.append(tokens)

        print(f"Loaded {len(self.data)} token sequences from {token_dir}")

    def _pad_or_truncate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Ensure tokens has length seq_len + 1 (we slice for input/target)."""
        target_len = self.seq_len + 1
        L = tokens.shape[0]
        if L >= target_len:
            return tokens[:target_len]
        else:
            pad = torch.full((target_len - L,), self.pad_token_id, dtype=torch.long)
            return torch.cat([tokens, pad])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx]  # (seq_len + 1,)
        input_ids = tokens[:-1]   # (seq_len,)
        target_ids = tokens[1:]   # (seq_len,)
        return input_ids, target_ids
