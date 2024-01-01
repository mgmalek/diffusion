from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from einops import rearrange
from more_itertools import flatten


def deep_apply(x: Any, fn: Callable) -> Any:
    if isinstance(x, dict):
        return {k: deep_apply(v, fn) for k, v in x.items()}
    elif isinstance(x, list):
        return [deep_apply(v, fn) for v in x]
    elif isinstance(x, tuple):
        return tuple([deep_apply(v, fn) for v in x])
    return fn(x)


to_cpu = partial(deep_apply, fn=lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
to_cuda_nonblocking = partial(deep_apply, fn=lambda x: x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x)


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """All keys in the the state dict of torch compiled models have the prefix '_orig_mod.'
    so we remove this key to normalize param names in the state dict"""

    def _normalize_key(k: str) -> str:
        return k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k

    return {_normalize_key(k): v for k, v in state_dict.items()}


def save_checkpoint(model: nn.Module, checkpoint_dir: Path, iter_num: int) -> None:
    ckpt_path = (checkpoint_dir / f"ckpt_{iter_num}.pt").resolve()
    print(f"Saving checkpoint to {str(ckpt_path)}")
    state_dict = _normalize_state_dict(model.state_dict())
    torch.save(state_dict, ckpt_path)


def visualize_rollout(grid_shape: Tuple[int, int], x_rollout: torch.Tensor, inference_steps: List[int]):
    x_rollout = (x_rollout + 1) / 2
    grid_h, grid_w = grid_shape
    assert grid_h * grid_w >= len(x_rollout), (grid_h, grid_w, x_rollout)
    fig, axs = plt.subplots(grid_h, grid_w, figsize=(16, 16), dpi=100, squeeze=False)
    for ax, curr_xt, curr_step in zip(flatten(axs), x_rollout, inference_steps):
        curr_xt = rearrange(curr_xt, "c h w -> h w c")
        if curr_xt.size(dim=-1) == 1:
            curr_xt = curr_xt.squeeze(dim=-1)
        curr_xt = torch.clamp(curr_xt, min=0, max=1)
        ax.set_title(f"t = {curr_step}")
        ax.imshow(curr_xt.float().detach().cpu().numpy())
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axs
