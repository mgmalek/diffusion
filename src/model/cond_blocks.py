from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from model.utils import NoWeightDecayModule


class LearnedPositionalEmbedding2d(NoWeightDecayModule):
    def __init__(self, inc: int, inh: int, inw: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(inc, inh, inw))
        torch.nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        return x + self.pos_emb


class TimeEmbedding2d(NoWeightDecayModule):
    def __init__(self, dim: int, t: int):
        super().__init__()
        assert dim % 2 == 0, dim
        t_pos = torch.arange(t, dtype=torch.float)
        t_pos = rearrange(t_pos, "t -> t 1")

        d_pos = torch.arange(dim // 2, dtype=torch.float)
        d_pos = rearrange(d_pos, "d -> 1 d")

        pow = (2 * d_pos) / (dim // 2)
        denom = 10000**pow
        t_emb = torch.cat((torch.sin(t_pos / denom), torch.cos(t_pos / denom)), dim=1)  # (t, d)
        t_emb = rearrange(t_emb, "t d -> t d 1 1")
        self.t_emb = nn.Parameter(t_emb, requires_grad=False)

    def forward(self, t):
        return self.t_emb[t]


class EltwiseAddClassEmbedding2d(NoWeightDecayModule):
    def __init__(self, inc: int, num_classes: int):
        super().__init__()
        self.cls_emb = nn.Parameter(torch.empty(num_classes, inc, 1, 1))
        torch.nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x, cond):
        assert cond is not None
        return x + self.cls_emb[cond]


class NoCondBlock(nn.Module):
    def forward(self, x, cond):
        return x


def make_cond_block(inc: int, config: Optional[Dict]) -> nn.Module:
    assert config is not None, breakpoint()

    config = deepcopy(config)
    if config is None:
        return NoCondBlock()

    cond_type = config.pop("type")
    if cond_type == "eltwise_add":
        return EltwiseAddClassEmbedding2d(inc=inc, **config)
    else:
        raise ValueError(f"Invalid {cond_type=}")
