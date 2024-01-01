import torch
import torch.nn as nn

from config import Config
from model.unet import UNet
from model.utils import NoWeightDecayModule


def load_model(config: Config) -> nn.Module:
    # Init model & optimizer
    if config.model_type == "unet":
        model = UNet(**config.model_kwargs)
    else:
        raise ValueError(f"Invalid {config.model_type = }")

    model = model.to(config.device)

    return model


def load_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    # We don't want to weight decay certain modules
    no_weight_decay_param_ids = set()

    for module_name, module in model.named_modules():
        if isinstance(
            module,
            (
                NoWeightDecayModule,
                nn.Embedding,
                nn.LayerNorm,
                nn.modules.batchnorm._NormBase,
            ),
        ):
            print(f"Not applying weight decay to all params in module {module_name}")
            no_weight_decay_param_ids |= set(id(p) for p in module.parameters())

    for param_name, param in model.named_parameters():
        if param_name.endswith(".bias"):
            print(f"Not applying weight decay to parameter {param_name}")
            no_weight_decay_param_ids.add(id(param))

    decay_params = [p for p in model.parameters() if id(p) not in no_weight_decay_param_ids]
    no_decay_params = [p for p in model.parameters() if id(p) in no_weight_decay_param_ids]
    param_groups = [
        dict(params=decay_params),
        dict(params=no_decay_params, weight_decay=0.0),
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)

    return optimizer
