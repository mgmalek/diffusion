from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from more_itertools import pairwise
from tqdm.auto import tqdm

from config import Config


@dataclass
class TrainingSample:
    x0: torch.Tensor
    xt: torch.Tensor
    t: torch.Tensor
    eps: torch.Tensor
    cond: Optional[torch.Tensor]
    pred: Optional[torch.Tensor] = None
    loss: Optional[float] = None

    def set_pred(self, pred: torch.Tensor):
        assert (pred is not None) and (self.pred is None)
        self.pred = pred

    def set_loss(self, loss: float):
        assert (loss is not None) and (self.loss is None)
        self.loss = loss


class ClassifierFreeGuidanceConfig(NamedTuple):
    w: float
    null_token_prob: float
    null_token_val: int


def get_linear_schedule(t_max: int, beta_min: float, beta_max: float, **kwargs) -> torch.Tensor:
    """Linear noise schedule as used in Ho et al. 2020"""
    beta = torch.linspace(beta_min, beta_max, steps=t_max, **kwargs)
    alpha = torch.cumprod(1 - beta, dim=0) ** 0.5
    return alpha


def get_cosine_schedule(t_max: int, **kwargs) -> torch.Tensor:
    s = 8e-3
    t = torch.arange(0, t_max + 1, dtype=torch.float32, **kwargs)
    ft = torch.cos((t / t_max + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_sq = ft / ft[0]
    alpha_sq = alpha_sq[:-1]
    alpha = torch.sqrt(alpha_sq)
    return alpha


class DDIM:
    def __init__(
        self,
        t_max: int,
        schedule_type: str,
        schedule_params: Dict,
        num_inference_steps: int,
        data_scale: float,
        device: str,
        cfg_config: Optional[Dict] = None,
    ):
        self.t_max = t_max
        self.data_scale = data_scale

        self.cfg_config = None
        if cfg_config is not None:
            self.cfg_config = ClassifierFreeGuidanceConfig(**cfg_config)

        alpha = self.get_schedule_alpha(t_max, schedule_type, schedule_params, device)
        self.alpha = F.pad(alpha, pad=(1, 0), value=1.0)
        self.sigma = torch.sqrt(1 - self.alpha**2)
        self.gamma = None  # we assume deterministic sampling by default

        inference_steps = np.linspace(0, t_max, num=num_inference_steps + 1, endpoint=True, dtype=np.int32)
        self.inference_steps = inference_steps[::-1]

    @classmethod
    def from_config(cls, config: Config) -> "DDIM":
        return cls(
            t_max=config.t,
            schedule_type=config.schedule_type,
            schedule_params=config.schedule_params,
            num_inference_steps=config.num_inference_steps,
            data_scale=config.data_scale,
            cfg_config=config.cfg_config,
            device=config.device,
        )

    def scale_data(self, x: torch.Tensor):
        return x * self.data_scale

    def unscale_data(self, x: torch.Tensor):
        return x / self.data_scale

    @staticmethod
    def get_schedule_alpha(t_max: int, schedule_type: str, schedule_params: Dict, device: str) -> torch.Tensor:
        schedule_fn = {
            "linear": get_linear_schedule,
            "cosine": get_cosine_schedule,
        }[schedule_type]
        alpha = schedule_fn(t_max=t_max, device=device, **schedule_params)
        return alpha

    def construct_training_sample(self, x0_unscaled: torch.Tensor, cond: Optional[torch.Tensor]) -> TrainingSample:
        x0 = self.scale_data(x0_unscaled)
        t = torch.randint(low=1, high=self.t_max + 1, size=(len(x0),), device=x0.device)
        eps = torch.randn_like(x0)

        if self.cfg_config:
            cond_mask = torch.rand(len(cond), device=cond.device) > self.cfg_config.null_token_prob
            cond = torch.where(cond_mask, cond, torch.full_like(cond, fill_value=self.cfg_config.null_token_val))

        alpha = self.alpha[t]
        alpha = rearrange(alpha, "b -> b 1 1 1")
        sigma = torch.sqrt(1 - alpha**2)

        xt = alpha * x0 + sigma * eps

        return TrainingSample(x0=x0, xt=xt, t=t, eps=eps, cond=cond)

    def sample(
        self,
        xT: torch.Tensor,
        cond: Optional[torch.Tensor],
        model: nn.Module,
        pred_type: str,
    ) -> torch.Tensor:
        x_timesteps = []
        x_timesteps.append(xT)

        xt = xT

        for t_curr, t_next in tqdm(pairwise(self.inference_steps), total=len(self.inference_steps) - 1):
            alpha_t_curr = self.alpha[t_curr]
            alpha_t_next = self.alpha[t_next]
            sigma_t_curr = self.sigma[t_curr]
            sigma_t_next = self.sigma[t_next]

            t = torch.full((len(xt),), fill_value=t_curr, device=xt.device)

            # Run model inference
            if self.cfg_config is not None:
                # NOTE: we have a .clone() here because we forward the model twice, and when using CUDA graphs the
                # output tensor from the first call to the model will be overwritten by the second call to the model,
                # so we need to clone the first output to retain its data
                pred = model(xt, t, torch.zeros(len(xt), dtype=torch.long, device=xt.device))
                pred = pred.clone()

                cond_pred = model(xt, t, cond) if (cond is not None) else None
            else:
                pred = model(xt, t, cond)
                cond_pred = None

            # Convert model prediction into x0
            if pred_type == "x0":
                x0 = pred
                x0_cond = cond_pred if (cond_pred is not None) else None
            else:
                raise NotImplementedError

            if (self.cfg_config is not None) and (cond_pred is not None):
                # Apply classifier-free guidance reweighting
                w = self.cfg_config.w
                x0 = (1 + w) * x0_cond - w * x0

            # Sample the next xt image
            if self.gamma is None:
                xt = alpha_t_next * x0 + sigma_t_next / sigma_t_curr * (xt - alpha_t_curr * x0)
            else:
                gamma_t_curr = self.gamma[t_curr]
                xt_mean = alpha_t_next * x0 + torch.sqrt(sigma_t_next**2 - gamma_t_curr**2) / sigma_t_curr * (
                    xt - alpha_t_curr * x0
                )
                xt = xt_mean + torch.randn_like(xt_mean) * gamma_t_curr

            x_timesteps.append(xt)

        x_rollouts = [torch.stack([x_ts[i] for x_ts in x_timesteps], dim=0) for i in range(len(xT))]

        xt = self.unscale_data(xt)
        x_rollouts = [self.unscale_data(x_) for x_ in x_rollouts]

        return xt, x_rollouts


class DDPM(DDIM):
    def __init__(
        self,
        t_max: int,
        schedule_type: str,
        schedule_params: Dict,
        data_scale: float,
        device: str,
    ):
        super().__init__(
            t_max=t_max,
            schedule_type=schedule_type,
            schedule_params=schedule_params,
            num_inference_steps=t_max,
            data_scale=data_scale,
            device=device,
        )

        alpha_t_sq = self.alpha[1:] ** 2
        alpha_t_prev_sq = self.alpha[:-1] ** 2
        sigma_t_sq = 1 - alpha_t_sq
        sigma_t_prev_sq = 1 - alpha_t_prev_sq
        gamma = (sigma_t_prev_sq / sigma_t_sq) * (1 - alpha_t_sq / alpha_t_prev_sq)
        self.gamma = F.pad(gamma, (1, 0), value=0.0)

    @classmethod
    def from_config(cls, config: Config) -> "DDPM":
        return cls(
            t_max=config.t,
            schedule_type=config.schedule_type,
            schedule_params=config.schedule_params,
            data_scale=config.data_scale,
            device=config.device,
        )
