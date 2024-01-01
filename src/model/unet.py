from functools import partial
from math import log2
from typing import Dict, List, Optional

import torch.nn as nn
from more_itertools import first, pairwise

from model.conv_blocks import DownBlock, UpBlock
from model.transformer_blocks import TransformerEncoder


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        channels: List[int],
        n_blocks_per_level: int,
        n_levels: int,
        attn_heads: int,
        attn_blocks: int,
        dropout_p: float,
        attn_latent_h: int,
        t: int,
        cond_config: Optional[Dict] = None,
    ):
        super().__init__()
        self._sanity_check_params(in_height, in_width, channels, n_levels)

        _make_transformer_encoder = partial(
            TransformerEncoder,
            n_heads=attn_heads,
            n_blocks=attn_blocks,
            dropout_p=dropout_p,
            t=t,
            cond_config=cond_config,
        )

        proj_outc = first(channels)
        self.in_proj = nn.Conv2d(in_channels, proj_outc, 1)

        latent_height = in_height
        latent_width = in_width
        added_down_attn = False
        added_up_attn = False

        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        for inc, outc in pairwise(channels):
            self.down_blocks.append(
                DownBlock(inc, outc, n_blocks_per_level, t, dropout_p=dropout_p, cond_config=cond_config)
            )
            latent_height //= 2
            latent_width //= 2

            # NOTE: Ho et al. 2020 always adds attention blocks at the 16x16 resolution
            if latent_height == attn_latent_h:
                assert not added_down_attn
                self.down_blocks.append(
                    _make_transformer_encoder(dim=outc, in_height=latent_height, in_width=latent_width)
                )
                added_down_attn = True

        assert added_down_attn

        # Upsample Blocks
        self.up_blocks = nn.ModuleList()
        for inc, outc in pairwise(channels[::-1]):
            if latent_height == attn_latent_h:
                assert not added_up_attn
                self.up_blocks.append(
                    _make_transformer_encoder(dim=inc, in_height=latent_height, in_width=latent_width)
                )
                added_up_attn = True

            self.up_blocks.append(
                UpBlock(inc, outc, n_blocks_per_level, t, dropout_p=dropout_p, cond_config=cond_config)
            )

            latent_height *= 2
            latent_width *= 2

        assert added_up_attn

        self.out_proj = nn.Conv2d(proj_outc, in_channels, 1)
        nn.init.zeros_(self.out_proj.weight)

    @staticmethod
    def _sanity_check_params(in_height, in_width, channels, n_levels) -> None:
        # For simplicity when calculating spatial dims of intermediate features,
        # we require input spatial dims to be a power of 2
        def _is_power_of_2(v):
            return abs(log2(v) - round(log2(v))) < 1e-6

        assert _is_power_of_2(in_height), in_height
        assert _is_power_of_2(in_width), in_width
        assert len(channels) == n_levels, (len(channels), n_levels)

    def forward(self, x, t, cond=None):
        t = t - 1  # we never sample with t=0 so the min value of t is 1

        x = self.in_proj(x)

        # Downsample Blocks
        down_feats = []
        for down_block in self.down_blocks:
            x = down_block(x, t, cond)
            down_feats.append(x)

        # Upsample Blocks
        for up_block in self.up_blocks:
            x_residual = down_feats.pop()
            x = x + x_residual
            x = up_block(x, t, cond)

        x = self.out_proj(x)

        return x
