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

    On each __getitem__ call a random window of seq_len+1 tokens is sampled
    from the full sequence, providing data diversity across epochs.
    Sequences shorter than seq_len+1 are right-padded with pad_token_id.
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
            self.data.append(tokens)  # store full length

        print(f"Loaded {len(self.data)} token sequences from {token_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx]  # full length
        L = tokens.shape[0]
        target_len = self.seq_len + 1

        if L <= target_len:
            # pad if needed
            if L < target_len:
                pad = torch.full((target_len - L,), self.pad_token_id, dtype=torch.long)
                tokens = torch.cat([tokens, pad])
        else:
            # random start within valid range
            start = torch.randint(0, L - target_len, (1,)).item()
            tokens = tokens[start : start + target_len]

        input_ids = tokens[:-1]   # (seq_len,)
        target_ids = tokens[1:]   # (seq_len,)
        return input_ids, target_ids
