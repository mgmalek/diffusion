import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from config import load_config_from_yaml
from datasets import get_image_size
from ddim import DDIM
from model.models import load_model
from utils import visualize_rollout


def main(config_path: Path, iter_num: int, cond: Optional[int], config_overrides: List[str]):
    config = load_config_from_yaml(config_path, config_overrides)

    ddim = DDIM.from_config(config)
    model = load_model(config)
    model.eval()

    checkpoint_path = config.checkpoint_dir / f"ckpt_{iter_num}.pt"
    print(f"Loading checkpoint from {checkpoint_path.resolve()}")
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    if config.compile:
        model = torch.compile(model, dynamic=False, fullgraph=True, mode="reduce-overhead")

    print(f"Generating {config.inference_batch_size} images")
    image_size = get_image_size(config.dataset_name)
    with torch.inference_mode():
        xT = torch.randn(config.inference_batch_size, *image_size, device=config.device)
        if cond is not None:
            cond = cond + 1  # class 0 is the null token
            cond = torch.tensor([cond], dtype=torch.long, device=xT.device)
        _, x_rollouts = ddim.sample(xT, cond, model, config.pred_type)

    config.inference_vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving generated images to {config.inference_vis_dir.resolve()}")
    for i, x_rollout in tqdm(enumerate(x_rollouts), total=len(x_rollouts)):
        _fig, _axs = visualize_rollout(config.inference_grid_shape, x_rollout, inference_steps=ddim.inference_steps)
        plt.tight_layout()
        plt.savefig(config.inference_vis_dir / f"example_{iter_num}_{i:04d}.png")
        plt.close()

    print("Finished saving generated images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to config .yaml file")
    parser.add_argument("--iter-num", type=int, help="Iteration number of the checkpoint to use for generation")
    parser.add_argument("--cond", type=int, help="Class idx to use for classifier-free guidance conditioning")
    base_args, config_overrides = parser.parse_known_args()
    main(**base_args.__dict__, config_overrides=config_overrides)
