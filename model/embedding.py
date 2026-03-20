"""
Audio token embedding: token embedding + learned positional embedding.
"""

import torch
import torch.nn as nn


class AudioEmbedding(nn.Module):
    """
    Combines token embedding and learned positional embedding.

    Args:
        vocab_size: number of tokens (e.g. 1028)
        d_model: embedding dimension
        max_seq_len: maximum sequence length for positional embeddings
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) LongTensor

        Returns:
            embeddings: (B, T, d_model)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        token_emb = self.token_emb(input_ids)   # (B, T, d_model)
        pos_emb = self.pos_emb(positions)        # (1, T, d_model)
        return token_emb + pos_emb
