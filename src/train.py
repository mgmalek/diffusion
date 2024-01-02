import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.profiler import ProfilerActivity, profile
from torch.utils.tensorboard import SummaryWriter

from config import Config, load_config_from_yaml
from datasets import get_train_dataloader, iter_dl_with_prefetch
from ddim import DDIM, TrainingSample
from logger import log_step
from model.models import load_model, load_optimizer
from utils import save_checkpoint


def train_step(
    iter_dl,
    ddim: DDIM,
    config: Config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> TrainingSample:
    x0_unscaled, cond = next(iter_dl)

    sample = ddim.construct_training_sample(x0_unscaled, cond)

    with torch.autocast(device_type=config.device, enabled=config.amp, dtype=torch.bfloat16):
        pred = model(sample.xt, sample.t, sample.cond)

        if config.pred_type == "x0":
            loss = F.mse_loss(pred, sample.x0)
        else:
            raise ValueError(f"Invalid {config.pred_type = }")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    sample.set_pred(pred.detach())
    sample.set_loss(loss.item())

    return sample


def profile_model(train_step_kwargs: Dict, checkpoint_dir: Path) -> None:
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=4),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(checkpoint_dir),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        for _ in range(10):
            train_step(**train_step_kwargs)
            prof.step()


def main(config_path: Path, config_overrides: List[str]):
    # Load config
    config = load_config_from_yaml(config_path, config_overrides)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing logs to {str(config.checkpoint_dir)}")

    # Initialize scheduler, model and optimizer
    ddim = DDIM.from_config(config)
    model = load_model(config)
    optimizer = load_optimizer(model, config)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.max_iters)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # NOTE: there's currently a PyTorch issue where profiling a model that uses CUDA graphs will hang.
    # Entering a profile context prior to compiling the model appears to be a workaround for this issue.
    # GitHub issue link: https://github.com/pytorch/pytorch/issues/75504
    with profile(activities=[ProfilerActivity.CPU]):
        pass

    if config.compile:
        # NOTE: we set dynamic=False because otherwise we'll fail to recompile inside `log_step` when generating samples.
        # The recompile fails when tracing einops operations because dynamo uses SymInt objects as part of the tensor
        # shape. However, SymInt objects are not hashable and einops recipes are wrapped in an LRU cache which requires all
        # elements of the tensor shape to be hashable.
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead", dynamic=False)

    # Initialize dataloader
    train_dl = get_train_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    iter_dl = iter_dl_with_prefetch(iter(train_dl), device=config.device)

    # Initialize tensorboard logging
    writer = SummaryWriter(config.checkpoint_dir)

    train_step_kwargs = dict(
        iter_dl=iter_dl, ddim=ddim, config=config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
    )

    # Profile the model
    if config.profile:
        profile_model(train_step_kwargs, checkpoint_dir=config.checkpoint_dir)
        print("Done profiling the model, exiting...")
        return

    # Training Loop
    for iter_num in range(1, config.max_iters + 1):
        iter_start_time = time.time_ns()
        sample = train_step(**train_step_kwargs)
        iter_duration_ns = time.time_ns() - iter_start_time

        log_step(
            ddim=ddim,
            model=model,
            config=config,
            sample=sample,
            iter_num=iter_num,
            iter_duration_ns=iter_duration_ns,
            writer=writer,
        )

        if iter_num % config.save_checkpoint_every == 0:
            save_checkpoint(model, config.checkpoint_dir, iter_num)

    save_checkpoint(model, config.checkpoint_dir, iter_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to config .yaml file")
    base_args, config_overrides = parser.parse_known_args()
    main(**base_args.__dict__, config_overrides=config_overrides)
