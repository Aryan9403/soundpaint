"""
MusicLM: Causal Transformer language model for audio token generation.

Pure PyTorch — no external dependencies beyond torch itself.

Architecture:
    input_ids (B, T)
    → AudioEmbedding (B, T, d_model)
    → N x CausalTransformerBlock
    → LayerNorm
    → lm_head (linear, no bias)
    → logits (B, T, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from model.embedding import AudioEmbedding


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class MusicLM(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config.get("n_heads", 8)
        vocab_size = config["vocab_size"]
        seq_len = config["seq_len"]
        dropout = config.get("dropout", 0.1)

        self.embedding = AudioEmbedding(vocab_size, d_model, seq_len)
        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
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
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.95,
        top_k: int = 250,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        seq_len = self.config["seq_len"]
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            input_ids = generated[:, -seq_len:]
            logits = self.forward(input_ids)[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_vals = torch.topk(logits, top_k_val).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_vals, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated
