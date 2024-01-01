import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from more_itertools import zip_equal
from torch.utils.tensorboard import SummaryWriter

from config import Config
from datasets import get_image_size
from ddim import DDIM, TrainingSample
from utils import visualize_rollout


def dump_train_vis(writer: SummaryWriter, iter_num: int, sample: TrainingSample):
    """Dump a visualisation of a training example and the model's corresponding prediction to tensorboard"""

    print("Dumping training examples to disk...")

    batch_size = len(sample.x0)
    rgb_train_imgs = []
    for i in range(min(batch_size, 4)):
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"t={sample.t[i].item()}")

        plot_data = [
            ("x0", sample.x0[i].cpu()),
            ("xt", sample.xt[i].cpu()),
            ("eps", sample.eps[i].cpu()),
            ("pred", sample.pred[i].cpu()),
        ]
        for ax, (label, data) in zip_equal(axs, plot_data):
            ax.set_title(label)
            data = (data + 1) / 2
            data = rearrange(data, "c h w -> h w c")
            if data.size(dim=-1) == 1:
                data = data.squeeze(-1)

            ax.imshow(data.float().numpy(), vmin=-2, vmax=2)

        plt.tight_layout()
        fig.canvas.draw()
        s, (width, height) = fig.canvas.print_to_buffer()
        plt.close()
        rgb_arr = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        rgb_arr = rgb_arr[..., :3]
        rgb_train_imgs.append(rgb_arr)

    writer.add_images("Training Example", np.stack(rgb_train_imgs), iter_num, dataformats="NHWC")


def dump_inference_vis(writer: SummaryWriter, iter_num: int, config: Config, model: nn.Module, ddim: DDIM):
    """Generate several (unconditional) samples and dump them to tensorboard"""

    print(f"Generating {config.inference_batch_size} images...")

    image_size = get_image_size(config.dataset_name)
    xT = torch.randn(config.inference_batch_size, *image_size, device=config.device)
    cond = torch.zeros(config.inference_batch_size, device=config.device, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        _, x_rollouts = ddim.sample(xT, cond, model, config.pred_type)
    model.train()

    rgb_rollout_imgs = []
    for x_rollout in x_rollouts:
        fig, _axs = visualize_rollout(config.inference_grid_shape, x_rollout, inference_steps=ddim.inference_steps)
        plt.tight_layout()
        fig.canvas.draw()
        s, (width, height) = fig.canvas.print_to_buffer()
        plt.close()
        rgb_arr = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        rgb_arr = rgb_arr[..., :3]
        rgb_rollout_imgs.append(rgb_arr)

    writer.add_images(
        "Generated Image",
        np.stack(rgb_rollout_imgs, axis=0),
        iter_num,
        dataformats="NHWC",
    )


def log_step(
    ddim: DDIM,
    model: nn.Module,
    config: Config,
    sample: TrainingSample,
    iter_num: int,
    iter_duration_ns: float,
    writer: SummaryWriter,
):
    iter_duration_s = iter_duration_ns / 1e9
    samples_per_sec = len(sample.x0) / iter_duration_s
    writer.add_scalar("Perf/iter_duration_s", iter_duration_s, iter_num)
    writer.add_scalar("Perf/samples_per_sec", samples_per_sec, iter_num)
    writer.add_scalar("Loss/train", sample.loss, iter_num)

    if iter_num % config.log_every == 0:
        print(f"Iteration: {iter_num:8d}        Step Duration: {iter_duration_s:6.3f}")

    if iter_num % config.train_vis_every == 0:
        dump_train_vis(writer, iter_num, sample)

    if iter_num % config.inference_vis_every == 0:
        dump_inference_vis(writer, iter_num, config, model, ddim)
