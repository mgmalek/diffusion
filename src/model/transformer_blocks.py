from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.cond_blocks import (LearnedPositionalEmbedding2d, TimeEmbedding2d,
                               make_cond_block)


class MLP(nn.Sequential):
    def __init__(self, inc: int, midc: int, dropout_p: float):
        super().__init__(
            nn.Linear(inc, midc),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(midc, inc),
            nn.Dropout(p=dropout_p),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout_p: float):
        super().__init__()
        assert dim % n_heads == 0, (dim, n_heads)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.o_drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        qkv = rearrange(
            qkv,
            "b n (qkv h c) -> qkv b h n c",
            qkv=3,
            h=self.n_heads,
            c=self.dim // self.n_heads,
        )
        q, k, v = torch.unbind(qkv, dim=0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        x = rearrange(x, "b h n c -> b n (h c)")
        x = self.o_proj(x)
        x = self.o_drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout_p: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout_p=dropout_p)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dropout_p=dropout_p)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        in_height: int,
        in_width: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        t: int,
        cond_config: Optional[Dict] = None,
    ):
        super().__init__()
        assert dim % n_heads == 0, (dim, n_heads)
        self.pos_emb = LearnedPositionalEmbedding2d(dim, in_height, in_width)
        self.blocks = nn.Sequential(
            *[TransformerEncoderBlock(dim=dim, n_heads=n_heads, dropout_p=dropout_p) for _ in range(n_blocks)]
        )
        self.time_emb = nn.Sequential(
            TimeEmbedding2d(dim, t),
            nn.Conv2d(dim, dim, 1),
        )
        self.cond_emb = make_cond_block(dim, cond_config)

    def forward(self, x, t, cond):
        x = self.pos_emb(x) + self.time_emb(t)
        if self.cond_emb is not None:
            x = self.cond_emb(x, cond)

        _, _, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.blocks(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
