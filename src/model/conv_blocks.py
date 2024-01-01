from collections import OrderedDict
from typing import Dict, Optional

import torch.nn as nn

from model.cond_blocks import TimeEmbedding2d, make_cond_block


class ConvBNReLU(nn.Sequential):
    def __init__(self, inc, outc, kernel_size, bn=True, activation=True, **kwargs):
        super().__init__(
            OrderedDict(
                conv=nn.Conv2d(inc, outc, kernel_size, **kwargs),
                norm=nn.BatchNorm2d(outc) if bn else nn.Identity(),
                relu=nn.ReLU() if activation else nn.Identity(),
            )
        )


class ResBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        midc: int,
        outc: int,
        t: int,
        dropout_p: float,
        final_activation: bool = True,
        cond_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBNReLU(inc, midc, kernel_size=1),
            nn.Dropout2d(p=dropout_p),
            ConvBNReLU(midc, midc, kernel_size=3, padding=1),
            ConvBNReLU(
                midc,
                outc,
                kernel_size=1,
                bn=final_activation,
                activation=final_activation,
            ),
        )
        self.residual = nn.Conv2d(inc, outc) if (inc != outc) else nn.Identity()
        self.time_emb = nn.Sequential(
            TimeEmbedding2d(outc, t),
            nn.Conv2d(outc, outc, 1),
        )
        self.cond_emb = make_cond_block(outc, cond_config)

    def forward(self, x, t, cond):
        x = self.convs(x) + self.residual(x) + self.time_emb(t)
        if self.cond_emb is not None:
            x = self.cond_emb(x, cond)
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        n_blocks: int,
        t: int,
        dropout_p: float,
        cond_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.down_conv = nn.Conv2d(inc, outc, kernel_size=3, stride=2, padding=1)
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(outc, outc // 4, outc, t=t, dropout_p=dropout_p, cond_config=cond_config)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, t, cond):
        x = self.down_conv(x)
        for block in self.res_blocks:
            x = block(x, t, cond)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        n_blocks: int,
        t: int,
        dropout_p: float,
        cond_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResBlock(inc, inc // 4, inc, t=t, dropout_p=dropout_p, cond_config=cond_config) for _ in range(n_blocks)]
        )
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(inc, outc, kernel_size=2, stride=2),
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t, cond):
        for block in self.res_blocks:
            x = block(x, t, cond)
        x = self.up_conv(x)
        return x
