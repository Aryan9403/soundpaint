"""
MusicMambaLM: Mamba-based language model for audio token generation.

Architecture:
    input_ids (B, T)
    → AudioEmbedding (B, T, d_model)
    → N x Mamba block
    → LayerNorm
    → lm_head (linear, no bias)
    → logits (B, T, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from model.embedding import AudioEmbedding


class MusicMambaLM(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        d_model = config["d_model"]
        n_layers = config["n_layers"]
        vocab_size = config["vocab_size"]
        seq_len = config["seq_len"]

        self.embedding = AudioEmbedding(vocab_size, d_model, seq_len)

        # Import here so the module can be imported without mamba_ssm installed
        # (will fail at runtime if not installed, not at import time)
        from mamba_ssm import Mamba

        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding weights with lm_head
        self.lm_head.weight = self.embedding.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.token_emb.weight, std=0.02)
        nn.init.normal_(self.embedding.pos_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) LongTensor

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embedding(input_ids)  # (B, T, d_model)

        for layer in self.layers:
            x = x + layer(x)           # residual connection

        x = self.norm(x)               # (B, T, d_model)
        logits = self.lm_head(x)       # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.95,
        top_k: int = 250,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt_ids: (1, T) or (T,) LongTensor — starting context
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            eos_token_id: stop early if this token is generated

        Returns:
            generated: (1, T + max_new_tokens) LongTensor
        """
        self.eval()
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # (1, T)

        seq_len = self.config["seq_len"]
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to seq_len context window
            input_ids = generated[:, -seq_len:]

            logits = self.forward(input_ids)  # (1, T, vocab_size)
            logits = logits[:, -1, :]         # last token → (1, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_vals = torch.topk(logits, top_k_val).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_vals, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated
